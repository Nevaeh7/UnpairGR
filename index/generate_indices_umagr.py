import argparse
import collections
import copy
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.uma_gr import UMAGRTokenizer, code_prefixes
from device_utils import get_preferred_device, is_cuda_device


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        data_root,
        dataset,
        text_embedding_file,
        image_embedding_file,
        input_mode,
    ):
        self.mode = input_mode
        self.text_embeddings = None
        self.image_embeddings = None

        text_path = os.path.join(data_root, dataset, f"{dataset}{text_embedding_file}")
        image_path = os.path.join(data_root, dataset, f"{dataset}{image_embedding_file}")
        has_text = os.path.exists(text_path)
        has_image = os.path.exists(image_path)

        if input_mode == "auto":
            if has_text and has_image:
                self.mode = "paired"
            elif has_text:
                self.mode = "text"
            elif has_image:
                self.mode = "image"
            else:
                raise FileNotFoundError(
                    f"[{dataset}] missing both text/image embeddings under {data_root}"
                )

        if self.mode in {"paired", "text"}:
            if not has_text:
                raise FileNotFoundError(f"[{dataset}] missing text embeddings: {text_path}")
            self.text_embeddings = np.load(text_path)

        if self.mode in {"paired", "image"}:
            if not has_image:
                raise FileNotFoundError(f"[{dataset}] missing image embeddings: {image_path}")
            self.image_embeddings = np.load(image_path)

        if self.mode == "paired" and len(self.text_embeddings) != len(self.image_embeddings):
            raise ValueError(
                f"[{dataset}] text/image length mismatch: "
                f"{len(self.text_embeddings)} vs {len(self.image_embeddings)}"
            )

    def __len__(self):
        if self.text_embeddings is not None:
            return len(self.text_embeddings)
        return len(self.image_embeddings)

    def __getitem__(self, index):
        batch = {}
        if self.text_embeddings is not None:
            batch["text"] = torch.FloatTensor(self.text_embeddings[index])
        if self.image_embeddings is not None:
            batch["image"] = torch.FloatTensor(self.image_embeddings[index])
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Generate UMA-GR semantic IDs")
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--text_embedding_file", type=str, default=".emb-llama-td.npy")
    parser.add_argument("--image_embedding_file", type=str, default=".emb-ViT-L-14.npy")
    parser.add_argument("--output_file", type=str, default=".index_umagr.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fuse_mode", type=str, default="consensus", choices=["mean", "text", "image", "consensus"])
    parser.add_argument(
        "--input_mode",
        type=str,
        default="paired",
        choices=["paired", "text", "image", "auto"],
        help="Embeddings used when generating semantic IDs.",
    )
    return parser.parse_args()


def resolve_output_path(data_root, dataset, output_file):
    output_path = Path(output_file)
    if output_path.is_absolute() or output_path.parent != Path("."):
        return output_path
    return Path(data_root) / dataset / f"{dataset}{output_file}"


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str))


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)
    return [group for group in index2id.values() if len(group) > 1]


def build_model(ckpt_args):
    return UMAGRTokenizer(
        text_in_dim=ckpt_args.text_in_dim,
        image_in_dim=ckpt_args.image_in_dim,
        shared_dim=ckpt_args.shared_dim,
        trunk_layers=ckpt_args.trunk_layers,
        trunk_heads=ckpt_args.trunk_heads,
        trunk_dropout=ckpt_args.trunk_dropout,
        num_emb_list=ckpt_args.num_emb_list,
        quant_e_dim=ckpt_args.quant_e_dim,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
        use_linear=ckpt_args.use_linear,
        share_trunk=not getattr(ckpt_args, "no_shared_trunk", False),
        share_codebook=not getattr(ckpt_args, "no_shared_codebook", False),
        assignment_temperature=getattr(ckpt_args, "assignment_temperature", 0.3),
        lambda_consistency=getattr(ckpt_args, "lambda_consistency", 0.0),
        lambda_usage=getattr(ckpt_args, "lambda_usage", 0.0),
        reliability_floor=getattr(ckpt_args, "reliability_floor", 0.05),
        reliability_epsilon=getattr(ckpt_args, "reliability_epsilon", 1e-6),
    )


def solve_collisions(all_indices, all_distances):
    all_indices_str = [str(index) for index in all_indices]
    all_indices_str_set = set(all_indices_str)
    sort_distances_index = np.argsort(all_distances, axis=2)

    item_min_dis = defaultdict(list)
    for item, distances in enumerate(all_distances):
        for dis in distances:
            item_min_dis[item].append(np.min(dis))

    tt = 0
    level = all_distances.shape[1] - 1
    max_num = all_distances.shape[2]

    while True:
        if check_collision(all_indices_str) or tt == 2:
            break

        collision_item_groups = get_collision_item(all_indices_str)
        for collision_items in collision_item_groups:
            min_distances = [item_min_dis[item][level] for item in collision_items]
            min_index = np.argsort(np.array(min_distances))

            for i, m_index in enumerate(min_index):
                if i == 0:
                    continue

                item = collision_items[m_index]
                ori_code = copy.deepcopy(all_indices[item])

                num = i
                while str(ori_code) in all_indices_str_set and num < max_num:
                    ori_code[level] = int(sort_distances_index[item][level][num])
                    num += 1

                for alt in range(1, max_num):
                    if str(ori_code) in all_indices_str_set:
                        ori_code = copy.deepcopy(all_indices[item])
                        ori_code[level - 1] = int(sort_distances_index[item][level - 1][alt])

                    num = 0
                    while str(ori_code) in all_indices_str_set and num < max_num:
                        ori_code[level] = int(sort_distances_index[item][level][num])
                        num += 1

                    if str(ori_code) not in all_indices_str_set:
                        break

                all_indices[item] = ori_code
                all_indices_str[item] = str(ori_code)
                all_indices_str_set.add(str(ori_code))

        tt += 1

    return all_indices


def main():
    config = parse_args()
    device = get_preferred_device(config.device)
    config.device = str(device)
    ckpt = torch.load(config.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    ckpt_args = ckpt["args"]
    model = build_model(ckpt_args)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    prefixes = code_prefixes(len(ckpt_args.num_emb_list))

    for dataset in [d.strip() for d in config.datasets.split(",") if d.strip()]:
        data = EmbeddingDataset(
            config.data_root,
            dataset,
            config.text_embedding_file,
            config.image_embedding_file,
            config.input_mode,
        )
        loader = DataLoader(
            data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=is_cuda_device(device),
        )

        all_indices = []
        all_distances = []
        for batch in tqdm(loader, desc=f"Generate {dataset} [{data.mode}]"):
            text_batch = batch.get("text")
            image_batch = batch.get("image")
            if text_batch is not None:
                text_batch = text_batch.to(config.device)
            if image_batch is not None:
                image_batch = image_batch.to(config.device)
            indices, distances = model.get_indices(
                text_x=text_batch,
                image_x=image_batch,
                use_sk=False,
                fuse_mode=config.fuse_mode,
            )
            all_indices.extend(indices.cpu().numpy().tolist())
            all_distances.extend(distances.cpu().numpy().tolist())

        all_distances = np.array(all_distances)
        all_indices = solve_collisions(all_indices, all_distances)
        all_indices_str = [str(indices) for indices in all_indices]
        collision_counts = get_indices_count(all_indices_str)
        max_collision = max(collision_counts.values(), default=0)
        if not check_collision(all_indices_str):
            duplicate_groups = sum(
                1 for count in collision_counts.values() if count > 1
            )
            raise RuntimeError(
                f"{dataset}: collision repair left {duplicate_groups} duplicate "
                f"semantic-ID groups (maximum multiplicity {max_collision}); "
                "refusing to write an ambiguous index"
            )

        all_indices_dict = {}
        for item, indices in enumerate(all_indices):
            all_indices_dict[item] = [
                prefixes[level].format(int(code))
                for level, code in enumerate(indices)
            ]

        output_path = resolve_output_path(config.data_root, dataset, config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fp:
            json.dump(all_indices_dict, fp, indent=4)
        print(f"Saved {dataset} semantic IDs to {output_path} (input_mode={data.mode})")
        print("Max collisions after solve:", max_collision)


if __name__ == "__main__":
    main()
