import json
import logging
import os
import random
import datetime
import argparse

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data import SeqRecDataset, FusionSeqRecDataset, ItemImageDataset, SynergisticDataset
from device_utils import (
    get_preferred_device,
    get_transformers_device_map,
    is_cuda_device,
    is_mps_device,
    safe_empty_cache,
)


def parse_prompt_ids(value, prompt_num=None):
    raw = str(value or "").strip()
    if raw.lower() == "all":
        if prompt_num is None or int(prompt_num) <= 0:
            raise ValueError("prompt_num must be positive when test_prompt_ids='all'")
        return list(range(int(prompt_num)))

    prompt_ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        prompt_ids.append(int(part))
    if not prompt_ids:
        raise ValueError("test_prompt_ids must contain at least one prompt id")
    return prompt_ids


def parse_task_list(value):
    return [task.strip() for task in str(value or "").split(",") if task.strip()]


def build_soft_prompts(tasks, prompt_num, tokenizer):
    soft_prompts = {}
    prompt_num = int(prompt_num)
    for i, task in enumerate(tasks):
        if task == 'fgfusionseqrec':
            token_ids = list(range(100 * (i + 1), 100 * (i + 1) + prompt_num * 4))
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt = []
            for k in range(4):
                p = ''.join(tokens[prompt_num * k : prompt_num * (k+1)])
                prompt.append(p)
        else:
            token_ids = list(range(100 * (i + 1), 100 * (i + 1) + prompt_num))
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt = ''.join(tokens)
        soft_prompts[task] = prompt
    return soft_prompts


def ensure_soft_prompts(args, tokenizer):
    tasks = parse_task_list(getattr(args, "tasks", "seqrec"))
    for attr in ("test_task", "valid_task"):
        for task in parse_task_list(getattr(args, attr, "")):
            if task not in tasks:
                tasks.append(task)
    args.soft_prompts = build_soft_prompts(
        tasks,
        getattr(args, "prompt_num", 4),
        tokenizer,
    )
    return args.soft_prompts


def parse_global_args(parser):


    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str, default="./config/ckpt",help="basic model path")

    parser.add_argument("--load_model_name", type=str, default=None,help="load pretrained model")

    parser.add_argument("--output_dir", type=str,
                        default="./log",
                        help="The output directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, cpu, mps, cuda, or cuda:<id>. auto prefers mps, then cuda, then cpu")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="",
                        help="data directory")
    parser.add_argument("--tasks", type=str, default="seqrec",
                        help="Downstream tasks, separate by comma")
    parser.add_argument('--pretrain_datasets', type=str, default="",
                            help=[])
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.json", help="the item indices file")
    parser.add_argument("--text_index_file", type=str, default=".index.json", help="the text item indices file")
    parser.add_argument("--image_index_file", type=str, default=".index.json", help="the item indices file")
    parser.add_argument("--fg_image_index_file", type=str, default=".index.json", help="the item indices file")
    parser.add_argument("--modal_mix_prob", type=float, default=0.0,
                        help="probability of swapping a seqrec history item into text/image semantic IDs during training")
    parser.add_argument("--modal_mix_eval_prob", type=float, default=0.0,
                        help="compatibility flag accepted by finetune wrappers; evaluation mixing is handled by explicit index files")
    parser.add_argument("--modal_mix_eval_seed", type=int, default=42,
                        help="compatibility flag accepted by finetune wrappers; evaluation masks use explicit files")
    parser.add_argument("--fusion_source_modal_mix_prob", type=float, default=0.0,
                        help="probability of swapping seqitem2image/seqitem2text training history items into text/image semantic IDs")

    parser.add_argument("--prompt_num", type=int, default=4,
                            help="soft prompt num")

    parser.add_argument("--code_num", type=int, default=256,
                            help="code num")

    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--train_data_mode", type=int, default=0,
                        help="0, 1")

    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")
    parser.add_argument("--only_valid_response", action="store_true", default=False,
                        help="whether only valid on responses")
    parser.add_argument("--only_valid_first_token", action="store_true", default=False,
                        help="whether only valid on responses")
    parser.add_argument("--tie_encoder_decoder", action="store_true", default=False,
                        help="whether only valid on responses")

    parser.add_argument("--train_prompt_sample_num", type=str, default="1",
                        help="the number of sampling prompts for each task")
    parser.add_argument("--train_data_sample_num", type=str, default="-1",
                        help="the number of sampling prompts for each task")

    # arguments related for evaluation
    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")
    parser.add_argument("--use_la", action="store_true", default=False)
    parser.add_argument("--use_lac", action="store_true", default=False)

    parser.add_argument("--add_prompt", action="store_true", default=False)

    return parser

def parse_train_args(parser):

    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
    parser.add_argument("--patient", type=int, default=10)
    parser.add_argument("--eval_num_beams", type=int, default=5)

    parser.add_argument("--valid_task", type=str, default="SeqRec")

    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for the contrastive loss (default: 0.1)")
    parser.add_argument("--mask_ratio", type=float, default=0.3, help="Ratio of tokens to mask in the dominant modality (default: 0.3)")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss (default: 0.07)")

    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="",
                        help="The checkpoint path")
    parser.add_argument("--filter_items", action="store_true", default=False,
                        help="whether filter illegal items")

    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")

    parser.add_argument("--save_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")

    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--eval_start", type=int, default=0,
                        help="0-based row offset for deterministic evaluation slicing")
    parser.add_argument("--eval_count", type=int, default=-1,
                        help="number of rows to evaluate after --eval_start; -1 keeps all remaining rows")
    parser.add_argument("--eval_stride", type=int, default=1,
                        help="take every Nth evaluation row before applying --eval_start/--eval_count")
    parser.add_argument("--eval_indices_file", type=str, default="",
                        help="optional JSON/text file with deterministic evaluation row indices")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="SeqRec")
    parser.add_argument("--eval_mode", type=str, default="test", choices=["valid", "test"],
                        help="Dataset split to evaluate with evaluate_recommender.py")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="print running metrics every N evaluation steps; 0 disables intermediate logs")
    parser.add_argument("--debug_first_batch", action="store_true", default=False,
                        help="print the first evaluation batch inputs and targets")
    parser.add_argument("--modal_mix_eval_manifest", default="", help=argparse.SUPPRESS)



    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args):

    tasks = args.tasks.split(",")

    # train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    # assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    # train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    # assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_prompt_sample_num = [1] * len(tasks)
    train_data_sample_num = [-1] * len(tasks)

    train_datasets = []
    for task, prompt_sample_num,data_sample_num in zip(tasks, train_prompt_sample_num, train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "seqtext":
            dataset = SeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "seqimage":
            dataset = SeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() in {"item2image", "image2item", "item2text", "text2item"}:
            dataset = ItemImageDataset(args, task=task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() in {"seqitem2image", "seqimage2item", "seqitem2text", "seqtext2item", "fusionseqrec"}:
            dataset = FusionSeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "masktext" or task.lower() == "maskimg":
            dataset = SynergisticDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
        else:
            raise NotImplementedError

        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_datasets = []
    valid_tasks = args.valid_task.lower().split(',')
    for valid_task in valid_tasks:
        # 如果验证任务是新定义的协同任务
        if valid_task in ['masktext', 'maskimg']:
            dataset = SynergisticDataset(args, task=valid_task, mode="valid")
        elif valid_task in {"seqitem2image", "seqimage2item", "seqitem2text", "seqtext2item", "fusionseqrec"}:
            dataset = FusionSeqRecDataset(args, task=valid_task, mode="valid")
        else:
            dataset = SeqRecDataset(args, task=valid_task, mode="valid")

        valid_datasets.append(dataset)
    valid_data = ConcatDataset(valid_datasets)

    # valid_data = SeqRecDataset(args, task=args.valid_task.lower(), mode="valid")

    return train_data, valid_data

def load_pretrain_datasets(args):

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [1] * len(tasks)
    train_data_sample_num = [-1] * len(tasks)

    pretrain_datasets = args.pretrain_datasets.split(',')

    train_datasets = []
    valid_datasets = []
    for dataset in pretrain_datasets:
        args.dataset = dataset
        for task, prompt_sample_num,data_sample_num in zip(tasks, train_prompt_sample_num, train_data_sample_num):
            if task.lower() == "seqrec":
                dataset = SeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

            elif task.lower() == "seqtext":
                dataset = SeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

            elif task.lower() == "seqimage":
                dataset = SeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

            elif task.lower() in {"item2image", "image2item", "item2text", "text2item"}:
                dataset = ItemImageDataset(args, task=task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

            elif task.lower() in {"seqitem2image", "seqimage2item", "seqitem2text", "seqtext2item", "fusionseqrec"}:
                dataset = FusionSeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

            elif task.lower() == "item2imagedel" or task.lower() == "image2itemdel":
                dataset = ItemImageDelDataset(args, task=task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

            elif task.lower() == "seqimagedel":
                dataset = SeqRecImageDelDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

            elif task.lower() == "seqitem2imagedel" or task.lower() == "seqimage2itemdel" or task.lower() == "fusionseqrecdel":
                dataset = FusionSeqRecDelDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

            else:
                raise NotImplementedError

            train_datasets.append(dataset)


        # valid_data = SeqRecDataset(args, mode="valid")
        valid_data = SeqRecDataset(args, task=args.valid_task.lower(), mode="valid")
        valid_datasets.append(valid_data)

    train_data = ConcatDataset(train_datasets)
    valid_data = ConcatDataset(valid_datasets)

    return train_data, valid_data

def load_test_dataset(args):

    test_task = args.test_task.lower()
    if test_task in {"seqrec", "seqtext", "seqimage"}:
        test_data = SeqRecDataset(args, task=test_task, mode=args.eval_mode, sample_num=args.sample_num)
        # test_data = SeqRecTestDataset(args, sample_num=args.sample_num)
    elif test_task in {"seqitem2image", "seqimage2item", "seqitem2text", "seqtext2item", "fusionseqrec"}:
        test_data = FusionSeqRecDataset(args, task=test_task, mode=args.eval_mode, sample_num=args.sample_num)
    else:
        raise NotImplementedError

    return test_data

def prefix_allowed_tokens_fn(candidate_trie, fallback_token_id=None):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        if not trie_out and fallback_token_id is not None:
            return [fallback_token_id]
        return trie_out

    return prefix_allowed_tokens

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )
