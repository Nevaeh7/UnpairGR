import logging
import math
import os
from collections import defaultdict
from itertools import cycle
from time import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from utils import ensure_dir, get_local_time, set_color
from device_utils import is_cuda_device


class UMAGRTrainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = torch.device(args.device)
        self.pin_memory = is_cuda_device(self.device)
        self.ckpt_dir = args.ckpt_dir
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_val_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_ckpt = "best_umagr_model.pth"

        self.model = self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = None

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner.lower()
        if learner == "adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "adamw":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "sgd":
            return optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        self.logger.warning("Unknown optimizer %s, fallback to AdamW", self.learner)
        return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _build_scheduler(self, steps_per_epoch: int):
        total_steps = max(1, steps_per_epoch * self.epochs)
        warmup_steps = max(1, int(total_steps * self.args.warmup_ratio))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        return batch.to(self.device)

    def _save_checkpoint(self, epoch: int, val_loss: float, collision_rate: float):
        ckpt_path = os.path.join(self.ckpt_dir, self.best_ckpt)
        torch.save(
            {
                "args": self.args,
                "epoch": epoch,
                "best_loss": self.best_loss,
                "best_val_loss": self.best_val_loss,
                "best_collision_rate": self.best_collision_rate,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            ckpt_path,
            pickle_protocol=4,
        )
        self.logger.info("%s: %s", set_color("Saving current", "blue"), ckpt_path)

    @torch.no_grad()
    def validate_tokenizer(self, paired_dataset, batch_size: int = 2048):
        self.model.eval()
        loader = DataLoader(
            paired_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.pin_memory,
        )

        all_codes = []
        total_loss = 0.0
        num_batches = 0
        for batch in tqdm(loader, total=len(loader), ncols=100, desc=set_color("Validate", "pink")):
            batch = self._to_device(batch)
            loss, _ = self.model.compute_phase1_loss(
                paired_text=batch["text"],
                paired_image=batch["image"],
                alpha=self.args.alpha,
                beta=self.args.beta,
                lambda_align=self.args.lambda_align,
                use_sk=False,
            )
            total_loss += float(loss.detach().cpu())
            num_batches += 1
            indices, _ = self.model.get_indices(
                text_x=batch["text"],
                image_x=batch["image"],
                use_sk=False,
                fuse_mode=self.args.fuse_mode,
            )
            indices = indices.cpu().numpy().tolist()
            for code in indices:
                all_codes.append("-".join(str(int(x)) for x in code))

        freq_count = defaultdict(int)
        for code in all_codes:
            freq_count[code] += 1
        collision_rate = (len(all_codes) - len(freq_count)) / max(1, len(all_codes))
        max_value = max(freq_count.values())
        min_value = min(freq_count.values())
        val_loss = total_loss / max(1, num_batches)
        return val_loss, max_value, min_value, collision_rate

    def _build_loader(self, dataset, batch_size: int):
        if dataset is None or len(dataset) == 0:
            return None
        effective_batch_size = min(max(1, batch_size), len(dataset))
        drop_last = len(dataset) >= batch_size
        return DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )

    def fit(self, paired_dataset, unpaired_image_dataset, unpaired_text_dataset):
        active_streams = 1
        if unpaired_image_dataset is not None and len(unpaired_image_dataset) > 0:
            active_streams += 1
        if unpaired_text_dataset is not None and len(unpaired_text_dataset) > 0:
            active_streams += 1

        paired_batch = max(1, self.args.batch_size // active_streams)
        unpaired_batch = paired_batch

        paired_loader = self._build_loader(paired_dataset, paired_batch)
        if paired_loader is None:
            raise ValueError("paired_dataset must contain at least one example")
        image_loader = self._build_loader(unpaired_image_dataset, unpaired_batch)
        text_loader = self._build_loader(unpaired_text_dataset, unpaired_batch)

        steps_per_epoch = max(
            len(paired_loader),
            len(image_loader) if image_loader is not None else 0,
            len(text_loader) if text_loader is not None else 0,
        )
        self._build_scheduler(steps_per_epoch)

        paired_iter = cycle(paired_loader)
        image_iter = cycle(image_loader) if image_loader is not None else None
        text_iter = cycle(text_loader) if text_loader is not None else None

        for epoch_idx in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_quant = 0.0
            epoch_unpaired = 0.0
            epoch_align = 0.0
            epoch_consistency = 0.0
            epoch_usage = 0.0

            iter_data = tqdm(
                range(steps_per_epoch),
                total=steps_per_epoch,
                ncols=100,
                desc=set_color(f"Train {epoch_idx}", "pink"),
            )

            start_time = time()
            for _ in iter_data:
                paired_batch_data = self._to_device(next(paired_iter))
                unpaired_image_batch = self._to_device(next(image_iter)) if image_iter is not None else None
                unpaired_text_batch = self._to_device(next(text_iter)) if text_iter is not None else None

                self.optimizer.zero_grad()
                loss, stats = self.model.compute_phase1_loss(
                    paired_text=paired_batch_data["text"],
                    paired_image=paired_batch_data["image"],
                    unpaired_image=unpaired_image_batch,
                    unpaired_text=unpaired_text_batch,
                    alpha=self.args.alpha,
                    beta=self.args.beta,
                    lambda_align=self.args.lambda_align,
                    use_sk=True,
                )
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss += stats["loss_total"]
                epoch_quant += stats["loss_quant"]
                epoch_unpaired += stats["loss_unpaired"]
                epoch_align += stats["loss_align"]
                epoch_consistency += stats.get("loss_consistency", 0.0)
                epoch_usage += stats.get("loss_usage", 0.0)

            end_time = time()
            self.logger.info(
                "%s [time: %.2fs, train loss: %.4f, quant: %.4f, unpaired: %.4f, align: %.4f, consistency: %.4f, usage: %.4f]",
                set_color(f"epoch {epoch_idx} training", "green"),
                end_time - start_time,
                epoch_loss / steps_per_epoch,
                epoch_quant / steps_per_epoch,
                epoch_unpaired / steps_per_epoch,
                epoch_align / steps_per_epoch,
                epoch_consistency / steps_per_epoch,
                epoch_usage / steps_per_epoch,
            )

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

            if (epoch_idx + 1) % self.eval_step == 0:
                val_loss, max_value, min_value, collision_rate = self.validate_tokenizer(paired_dataset)
                self.logger.info(
                    "%s [val_loss: %.6f, collision_rate: %.6f, max_repeat: %d, min_repeat: %d]",
                    set_color(f"epoch {epoch_idx} evaluating", "green"),
                    val_loss,
                    collision_rate,
                    max_value,
                    min_value,
                )
                if collision_rate < self.best_collision_rate or (
                    math.isclose(collision_rate, self.best_collision_rate, rel_tol=0.0, abs_tol=1e-12)
                    and val_loss < self.best_val_loss
                ):
                    self.best_collision_rate = collision_rate
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch_idx, val_loss, collision_rate)

        return self.best_loss, self.best_collision_rate
