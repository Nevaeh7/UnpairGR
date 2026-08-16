import argparse
import inspect
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
from typing import List

import torch
import torch.distributed as dist
import transformers

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers.trainer_callback import EarlyStoppingCallback

from utils import *
from collator import Collator


def remove_stale_model_artifacts(output_dir):

    stale_files = [
        "model.safetensors",
        "model.safetensors.index.json",
    ]
    for filename in stale_files:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            print(f"[checkpoint-cleanup] removed stale artifact: {path}")


def training_strategy_kwargs(strategy, training_arguments_cls=transformers.TrainingArguments):

    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    return {
        eval_key: strategy,
        "save_strategy": strategy,
    }


def supported_training_arguments_kwargs(kwargs, training_arguments_cls=transformers.TrainingArguments):

    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }


def supported_trainer_kwargs(kwargs, trainer_cls=transformers.Trainer):

    parameters = inspect.signature(trainer_cls.__init__).parameters
    trainer_kwargs = dict(kwargs)
    if (
        "tokenizer" in trainer_kwargs
        and "tokenizer" not in parameters
        and "processing_class" in parameters
    ):
        trainer_kwargs["processing_class"] = trainer_kwargs.pop("tokenizer")
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return trainer_kwargs
    return {
        key: value
        for key, value in trainer_kwargs.items()
        if key in parameters
    }


def train(args):

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    remove_stale_model_artifacts(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training in this repo requires CUDA/NCCL. On Mac MPS, run the script without torchrun.")
        device = get_preferred_device(f"cuda:{local_rank}", gpu_id=local_rank)
    else:
        device = get_preferred_device(args.device)
    args.device = str(device)
    if local_rank == 0:
        print(f"[device] using {device}")

    if args.load_model_name is not None:
        config = T5Config.from_pretrained(args.load_model_name)
        tokenizer = T5Tokenizer.from_pretrained(
            args.load_model_name,
            model_max_length=512,
        )
    else:
        config = T5Config.from_pretrained(args.base_model)
        tokenizer = T5Tokenizer.from_pretrained(
            args.base_model,
            model_max_length=512,
        )

    if args.tie_encoder_decoder:
        config.tie_encoder_decoder = True

    args.deepspeed = None
    gradient_checkpointing= False

    tasks = args.tasks.split(",")
    soft_prompts = {}
    for i, task in enumerate(tasks):
        token_ids = list(range(100 * (i + 1), 100 * (i + 1) + args.prompt_num))
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        soft_prompts[task] = ''.join(tokens)

    args.soft_prompts = soft_prompts

    train_data, valid_data = load_pretrain_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[-1].get_all_tokens())
    collator = Collator(args, tokenizer)

    if args.load_model_name is not None:
        model = T5ForConditionalGeneration.from_pretrained(args.load_model_name, config=config)
    else:
        model = T5ForConditionalGeneration(config)

    model.resize_token_embeddings(len(tokenizer))

    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print('train sequence')
        for dataset in train_data.datasets:
            print(dataset[100])
        print('valid sequence')
        for dataset in valid_data.datasets:
            print(dataset[100])
        print(model)
        print_trainable_parameters(model)


    if not ddp and is_cuda_device(device) and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # early_stop = EarlyStoppingCallback(early_stopping_patience=20)

    warmup_steps = int(len(train_data) / (world_size * args.per_device_batch_size * args.gradient_accumulation_steps))

    training_args_kwargs = {
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        # "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "constant_with_warmup",
        "fp16": args.fp16 and is_cuda_device(device),
        # "bf16": args.bf16,
        "logging_steps": args.logging_step,
        "optim": args.optim,
        "gradient_checkpointing": args.gradient_checkpointing,
        **training_strategy_kwargs(args.save_and_eval_strategy),
        "eval_steps": args.save_and_eval_steps,
        "save_steps": args.save_and_eval_steps,
        "output_dir": args.output_dir,
        # "save_total_limit": 5,
        "load_best_model_at_end": True,
        "save_safetensors": False,
        # "deepspeed": args.deepspeed,
        "ddp_find_unused_parameters": False if ddp else None,
        "report_to": "none",
        "eval_delay": 1 if args.save_and_eval_strategy == "epoch" else 2000,
    }

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_data,
        "eval_dataset": valid_data,
        "args": transformers.TrainingArguments(
            **supported_training_arguments_kwargs(training_args_kwargs)
        ),
        "tokenizer": tokenizer,
        "data_collator": collator,
        # "callbacks": [early_stop],
    }
    trainer = transformers.Trainer(**supported_trainer_kwargs(trainer_kwargs))
    model.config.use_cache = False


    try:
        trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
