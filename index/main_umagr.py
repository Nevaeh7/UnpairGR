import argparse
import logging
import random
from typing import Optional

import numpy as np
import torch

from uma_datasets import PairedEmbeddingDataset, SingleModalEmbeddingDataset
from models.uma_gr import UMAGRTokenizer
from trainer_umagr import UMAGRTrainer
from device_utils import get_preferred_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="UMA-GR tokenizer pre-training")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--learner", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--paired_datasets", type=str, default="Instruments,Arts,Games")
    parser.add_argument("--unpaired_image_datasets", type=str, default="")
    parser.add_argument("--unpaired_text_datasets", type=str, default="")
    parser.add_argument("--text_embedding_file", type=str, default=".emb-llama-td.npy")
    parser.add_argument("--image_embedding_file", type=str, default=".emb-ViT-L-14.npy")

    parser.add_argument("--shared_dim", type=int, default=256)
    parser.add_argument("--trunk_layers", type=int, default=3)
    parser.add_argument("--trunk_heads", type=int, default=4)
    parser.add_argument("--trunk_dropout", type=float, default=0.1)

    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256, 256])
    parser.add_argument("--quant_e_dim", type=int, default=256)
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.003])
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--use_linear", type=int, default=0)

    parser.add_argument("--alpha", type=float, default=1.0, help="quantization loss weight")
    parser.add_argument("--beta", type=float, default=0.5, help="unpaired loss weight")
    parser.add_argument("--lambda_align", type=float, default=0.1, help="alignment loss weight")
    parser.add_argument("--lambda_consistency", type=float, default=0.1, help="paired cross-view code consistency weight")
    parser.add_argument("--lambda_usage", type=float, default=0.02, help="codebook utilization regularization weight")
    parser.add_argument("--assignment_temperature", type=float, default=0.3, help="soft assignment temperature for consistency/utilization")
    parser.add_argument("--reliability_floor", type=float, default=0.05, help="minimum modality reliability when building consensus fused reps")
    parser.add_argument(
        "--reliability_epsilon",
        type=float,
        default=1e-6,
        help="epsilon in (kappa_m + epsilon) / (sum(kappa) + 2 * epsilon)",
    )
    parser.add_argument("--no_shared_trunk", action="store_true", help="ablation: use modality-specific trunks")
    parser.add_argument("--no_shared_codebook", action="store_true", help="ablation: use modality-specific codebooks")
    parser.add_argument("--no_unpaired_loss", action="store_true", help="ablation: disable unpaired streams and loss")
    parser.add_argument("--no_align_loss", action="store_true", help="ablation: set alignment loss weight to zero")

    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, mps, cuda, or cuda:<id>")
    parser.add_argument("--ckpt_dir", type=str, default="./log/umagr_tokenizer")
    parser.add_argument("--fuse_mode", type=str, default="consensus", choices=["mean", "text", "image", "consensus"])

    return parser.parse_args()


def main():
    seed = 2026
    set_seed(seed)

    args = parse_args()
    args.device = str(get_preferred_device(args.device))
    if args.no_unpaired_loss:
        args.beta = 0.0
        args.unpaired_image_datasets = ""
        args.unpaired_text_datasets = ""
    if args.no_align_loss:
        args.lambda_align = 0.0

    logging.basicConfig(level=logging.INFO)
    print(args)

    paired_dataset = PairedEmbeddingDataset(
        args.data_root,
        args.paired_datasets,
        args.text_embedding_file,
        args.image_embedding_file,
    )

    def build_single_modal_dataset(datasets: str, embedding_file: str, modality: str) -> Optional[SingleModalEmbeddingDataset]:
        if not datasets.strip():
            logging.warning("No unpaired %s datasets configured; skipping that training stream.", modality)
            return None
        return SingleModalEmbeddingDataset(
            args.data_root,
            datasets,
            embedding_file,
            modality=modality,
        )

    unpaired_image_dataset = build_single_modal_dataset(
        args.unpaired_image_datasets,
        args.image_embedding_file,
        modality="image",
    )
    unpaired_text_dataset = build_single_modal_dataset(
        args.unpaired_text_datasets,
        args.text_embedding_file,
        modality="text",
    )

    args.text_in_dim = paired_dataset.text_dim
    args.image_in_dim = paired_dataset.image_dim

    model = UMAGRTokenizer(
        text_in_dim=args.text_in_dim,
        image_in_dim=args.image_in_dim,
        shared_dim=args.shared_dim,
        trunk_layers=args.trunk_layers,
        trunk_heads=args.trunk_heads,
        trunk_dropout=args.trunk_dropout,
        num_emb_list=args.num_emb_list,
        quant_e_dim=args.quant_e_dim,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        use_linear=args.use_linear,
        assignment_temperature=args.assignment_temperature,
        lambda_consistency=args.lambda_consistency,
        lambda_usage=args.lambda_usage,
        reliability_floor=args.reliability_floor,
        reliability_epsilon=args.reliability_epsilon,
        share_trunk=not args.no_shared_trunk,
        share_codebook=not args.no_shared_codebook,
    )
    print(model)

    trainer = UMAGRTrainer(args, model)
    best_loss, best_collision = trainer.fit(
        paired_dataset=paired_dataset,
        unpaired_image_dataset=unpaired_image_dataset,
        unpaired_text_dataset=unpaired_text_dataset,
    )
    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision)


if __name__ == "__main__":
    main()
