import argparse
import inspect
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
from typing import List

import torch
import torch.distributed as dist
import transformers

from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

from utils import *
from collator import Collator, TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie


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


def evaluate_generation_metrics(args, checkpoint_path, tokenizer_path, valid_data, device):

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    valid_datasets = getattr(valid_data, "datasets", [valid_data])

    tokenizer = T5Tokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=512,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        checkpoint_path,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = True

    collator = TestCollator(args, tokenizer)
    summary = {
        "checkpoint_path": checkpoint_path,
        "datasets": {},
    }
    total_samples = 0
    total_metric_counts = {metric: 0.0 for metric in metrics}

    with torch.no_grad():
        for dataset in valid_datasets:
            dataset_name = f"{dataset.dataset}:{getattr(dataset, 'task', 'unknown')}:{dataset.mode}"
            if not hasattr(dataset, "get_all_items"):
                print(f"[generation-eval] skip unsupported dataset: {dataset_name}")
                continue

            if hasattr(dataset, "set_prompt"):
                dataset.set_prompt(args.valid_prompt_id)

            all_items = dataset.get_all_items()
            candidate_trie = Trie(
                [
                    [0] + tokenizer.encode(candidate)
                    for candidate in all_items
                ]
            )
            prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
            data_loader = DataLoader(
                dataset,
                batch_size=args.test_batch_size,
                collate_fn=collator,
                shuffle=False,
                num_workers=4,
                pin_memory=is_cuda_device(device),
            )

            metric_counts = {metric: 0.0 for metric in metrics}
            sample_count = 0

            for batch in data_loader:
                inputs = batch[0].to(device)
                targets = batch[1]
                sample_count += len(targets)

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                topk_res = get_topk_results(
                    predictions,
                    scores,
                    targets,
                    args.num_beams,
                    all_items=all_items if args.filter_items else None,
                )
                batch_metrics = get_metrics_results(topk_res, metrics)
                for metric, value in batch_metrics.items():
                    metric_counts[metric] += value

            if sample_count == 0:
                print(f"[generation-eval] dataset has no samples: {dataset_name}")
                continue

            dataset_metrics = {
                metric: metric_counts[metric] / sample_count
                for metric in metrics
            }
            summary["datasets"][dataset_name] = {
                "samples": sample_count,
                "metrics": dataset_metrics,
            }
            total_samples += sample_count
            for metric in metrics:
                total_metric_counts[metric] += metric_counts[metric]

    del model
    safe_empty_cache(device)

    if total_samples == 0:
        summary["aggregate"] = {}
        return summary

    summary["aggregate"] = {
        metric: total_metric_counts[metric] / total_samples
        for metric in metrics
    }
    return summary


def print_generation_summary(tag, summary):

    print("======================================================")
    print(f"{tag} generation results:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("======================================================")


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


def disabled_report_to_value():

    return "none"


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

    soft_prompts = ensure_soft_prompts(args, tokenizer)
    if local_rank == 0:
        for task, prompt in soft_prompts.items():
            print(task, prompt)

    train_data, valid_data = load_datasets(args)

    add_num = 0
    for dataset in train_data.datasets:
        add_num += tokenizer.add_tokens(dataset.get_new_tokens())

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
        print('train sequence')
        print(valid_data[100])
        print(model)
        print_trainable_parameters(model)


    if not ddp and is_cuda_device(device) and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args_kwargs = {
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "fp16": args.fp16 and is_cuda_device(device),
        # "bf16": args.bf16,
        "logging_steps": args.logging_step,
        "optim": args.optim,
        "gradient_checkpointing": args.gradient_checkpointing,
        **training_strategy_kwargs(args.save_and_eval_strategy),
        "eval_steps": args.save_and_eval_steps,
        "save_steps": args.save_and_eval_steps,
        "output_dir": args.output_dir,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "save_safetensors": False,
        # "deepspeed": args.deepspeed,
        "ddp_find_unused_parameters": False if ddp else None,
        "report_to": disabled_report_to_value(),
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
    }
    trainer = transformers.Trainer(**supported_trainer_kwargs(trainer_kwargs))
    model.config.use_cache = False


    try:
        trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

        final_checkpoint = os.path.join(args.output_dir, f"checkpoint-{trainer.state.global_step}")
        best_checkpoint = trainer.state.best_model_checkpoint
        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)
        del trainer
        del model

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        if local_rank == 0:
            safe_empty_cache(device)
            if args.skip_generation_eval:
                print("[generation-eval] skipped by --skip_generation_eval")
                return

            generation_eval = {}

            if os.path.isdir(final_checkpoint):
                generation_eval["final_epoch"] = evaluate_generation_metrics(
                    args=args,
                    checkpoint_path=final_checkpoint,
                    tokenizer_path=args.output_dir,
                    valid_data=valid_data,
                    device=device,
                )
                print_generation_summary("Final epoch", generation_eval["final_epoch"])
            else:
                print(f"[generation-eval] final checkpoint not found: {final_checkpoint}")

            if best_checkpoint and os.path.isdir(best_checkpoint):
                generation_eval["best_checkpoint"] = evaluate_generation_metrics(
                    args=args,
                    checkpoint_path=best_checkpoint,
                    tokenizer_path=args.output_dir,
                    valid_data=valid_data,
                    device=device,
                )
                print_generation_summary("Best checkpoint", generation_eval["best_checkpoint"])
            else:
                print(f"[generation-eval] best checkpoint not found: {best_checkpoint}")

            if generation_eval:
                results_path = os.path.join(args.output_dir, "generation_eval_summary.json")
                with open(results_path, "w") as f:
                    json.dump(generation_eval, f, indent=2, sort_keys=True)
                print(f"[generation-eval] saved summary to {results_path}")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser.add_argument(
        "--skip_generation_eval",
        action="store_true",
        default=False,
        help="Skip the built-in post-finetune generation evaluation.",
    )

    args = parser.parse_args()

    train(args)
