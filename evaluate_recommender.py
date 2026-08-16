import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from utils import *
from collator import TestCollatorSave
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie

def evaluate(args):

    set_seed(args.seed)
    print(vars(args))

    device = get_preferred_device(args.device, gpu_id=args.gpu_id)
    args.device = str(device)
    device_map = get_transformers_device_map(device, gpu_id=args.gpu_id)
    print(f"[device] using {device}")

    args.use_la = True

    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    ensure_soft_prompts(args, tokenizer)
    config = T5Config.from_pretrained(args.ckpt_path)

    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        **model_kwargs,
    )
    if device_map is None:
        model.to(device)

    prompt_ids = parse_prompt_ids(args.test_prompt_ids, prompt_num=getattr(args, "prompt_num", None))

    test_data = load_test_dataset(args)
    collator = TestCollatorSave(args, tokenizer)
    all_items = test_data.get_all_items()


    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate)
            for candidate in all_items
        ]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    # prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=False, num_workers=4, pin_memory=is_cuda_device(device))


    print("data num:", len(test_data))

    model.eval()

    metrics = args.metrics.split(",")
    all_outputs = []
    all_scores = []
    all_targets = []
    all_users = []
    save_dict = {}
    all_prompt_results = []
    with torch.no_grad():
        for prompt_id in prompt_ids:

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                users = batch[2]
                total += len(targets)
                if args.debug_first_batch and step == 0:
                    print(inputs)
                    print(targets)

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    # max_length=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
                output_ids = output["sequences"]
                if "sequences_scores" in output:
                    scores = output["sequences_scores"]
                else:
                    scores = torch.zeros(output_ids.shape[0], device=device)

                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                score_values = scores.detach().cpu().tolist()
                all_outputs.extend(output)
                all_scores.extend(score_values)
                all_targets.extend(targets)
                all_users.extend(users)
                save_dict["all_outputs"] = all_outputs
                save_dict["all_scores"] = all_scores
                save_dict["all_targets"] = all_targets
                save_dict["all_users"] = all_users
                # print(output)
                topk_res = get_topk_results(output, scores, targets, args.num_beams,
                                            all_items=all_items if args.filter_items else None)

                batch_metrics_res = get_metrics_results(topk_res, metrics)
                # print(batch_metrics_res)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                    temp={}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")
            ensure_dir(os.path.dirname(args.save_file) or ".")
            with open(args.save_file, "w") as f:
                json.dump(save_dict, f, indent=4)

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res)/len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")


    save_data={}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    ensure_dir(os.path.dirname(args.results_file) or ".")
    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the UnpairGR recommender")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    evaluate(args)
