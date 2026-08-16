import math
import string
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rq import ResidualVectorQuantizer


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UMAGRTokenizer(nn.Module):
    """
    MAQE-style tokenizer for UMA-GR.

    Design adapted to the current repository:
    - inputs are frozen pre-extracted text/image embeddings
    - modality-specific projection layers map them to a shared dimension
    - a shared Transformer trunk processes both modalities
    - a single shared RQ-VAE codebook discretizes both modalities
    """

    def __init__(
        self,
        text_in_dim: int = 768,
        image_in_dim: int = 768,
        shared_dim: int = 256,
        trunk_layers: int = 3,
        trunk_heads: int = 4,
        trunk_dropout: float = 0.1,
        num_emb_list=None,
        quant_e_dim: Optional[int] = None,
        kmeans_init: bool = True,
        kmeans_iters: int = 100,
        sk_epsilons=None,
        sk_iters: int = 50,
        use_linear: int = 0,
        assignment_temperature: float = 0.3,
        lambda_consistency: float = 0.0,
        lambda_usage: float = 0.0,
        reliability_floor: float = 0.05,
        share_trunk: bool = True,
        share_codebook: bool = True,
        reliability_epsilon: float = 1e-6,
    ):
        super().__init__()
        reliability_epsilon = float(reliability_epsilon)
        if not math.isfinite(reliability_epsilon) or reliability_epsilon <= 0.0:
            raise ValueError("reliability_epsilon must be finite and greater than zero")
        if num_emb_list is None:
            num_emb_list = [256, 256, 256, 256]
        if sk_epsilons is None:
            sk_epsilons = [0.0, 0.0, 0.0, 0.003]
        if quant_e_dim is None:
            quant_e_dim = shared_dim

        self.text_in_dim = text_in_dim
        self.image_in_dim = image_in_dim
        self.shared_dim = shared_dim
        self.trunk_layers = trunk_layers
        self.trunk_heads = trunk_heads
        self.num_emb_list = num_emb_list
        self.quant_e_dim = quant_e_dim
        self.assignment_temperature = assignment_temperature
        self.lambda_consistency = lambda_consistency
        self.lambda_usage = lambda_usage
        self.reliability_floor = reliability_floor
        self.reliability_epsilon = reliability_epsilon
        self.share_trunk = share_trunk
        self.share_codebook = share_codebook
        self.register_buffer(
            "consistency_level_weights",
            torch.linspace(1.0, 0.4, steps=len(num_emb_list)),
            persistent=False,
        )

        self.text_projection = ProjectionMLP(text_in_dim, shared_dim)
        self.image_projection = ProjectionMLP(image_in_dim, shared_dim)
        self.text_reconstruction = ProjectionMLP(shared_dim, text_in_dim)
        self.image_reconstruction = ProjectionMLP(shared_dim, image_in_dim)

        def build_trunk():
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=shared_dim,
                nhead=trunk_heads,
                dim_feedforward=shared_dim * 4,
                dropout=trunk_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=trunk_layers)

        if self.share_trunk:
            self.shared_trunk = build_trunk()
        else:
            self.text_trunk = build_trunk()
            self.image_trunk = build_trunk()

        def build_codebook():
            return ResidualVectorQuantizer(
                num_emb_list,
                quant_e_dim,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                sk_epsilons=sk_epsilons,
                sk_iters=sk_iters,
                use_linear=use_linear,
            )

        if self.share_codebook:
            self.shared_codebook = build_codebook()
        else:
            self.text_codebook = build_codebook()
            self.image_codebook = build_codebook()

        if shared_dim != quant_e_dim:
            self.pre_quant = nn.Linear(shared_dim, quant_e_dim)
            self.post_quant = nn.Linear(quant_e_dim, shared_dim)
        else:
            self.pre_quant = nn.Identity()
            self.post_quant = nn.Identity()

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "text":
            z = self.text_projection(x)
        elif modality == "image":
            z = self.image_projection(x)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # One-token Transformer trunk. Default uses a shared trunk; the
        # no-shared-trunk ablation routes each modality through its own trunk.
        if self.share_trunk:
            trunk = self.shared_trunk
        elif modality == "text":
            trunk = self.text_trunk
        else:
            trunk = self.image_trunk
        r = trunk(z.unsqueeze(1)).squeeze(1)
        return r

    def _codebook_for(self, modality: str):
        if self.share_codebook:
            return self.shared_codebook
        if modality == "image":
            return self.image_codebook
        return self.text_codebook

    def quantize(self, r: torch.Tensor, use_sk: bool = True, modality: str = "pair"):
        q_in = self.pre_quant(r)
        q_out, q_loss, indices, distances = self._codebook_for(modality)(q_in, use_sk=use_sk)
        r_hat = self.post_quant(q_out)
        return r_hat, q_loss, indices, distances

    def encode_and_quantize(self, x: torch.Tensor, modality: str, use_sk: bool = True):
        r = self.encode(x, modality)
        r_hat, q_loss, indices, distances = self.quantize(r, use_sk=use_sk, modality=modality)
        return r, r_hat, q_loss, indices, distances

    def _distances_to_probs(self, distances: torch.Tensor) -> torch.Tensor:
        temperature = max(self.assignment_temperature, 1e-6)
        return torch.softmax(-distances / temperature, dim=-1)

    def _normalized_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(probs.size(-1), device=probs.device, dtype=probs.dtype))
        return entropy / max_entropy.clamp_min(1e-12)

    def _reliability_from_distances(self, distances: torch.Tensor) -> torch.Tensor:
        probs = self._distances_to_probs(distances)
        mean_entropy = self._normalized_entropy(probs).mean(dim=1, keepdim=True)
        reliability = (1.0 - mean_entropy).clamp_min(self.reliability_floor)
        return reliability

    def _js_divergence(self, probs_a: torch.Tensor, probs_b: torch.Tensor) -> torch.Tensor:
        mean_probs = 0.5 * (probs_a + probs_b)
        kl_a = (probs_a.clamp_min(1e-12) * (probs_a.clamp_min(1e-12).log() - mean_probs.clamp_min(1e-12).log())).sum(dim=-1)
        kl_b = (probs_b.clamp_min(1e-12) * (probs_b.clamp_min(1e-12).log() - mean_probs.clamp_min(1e-12).log())).sum(dim=-1)
        return 0.5 * (kl_a + kl_b)

    def _consistency_loss(self, pair_distances: torch.Tensor, text_distances: torch.Tensor, image_distances: torch.Tensor) -> torch.Tensor:
        pair_probs = self._distances_to_probs(pair_distances)
        text_probs = self._distances_to_probs(text_distances)
        image_probs = self._distances_to_probs(image_distances)

        pair_text_js = self._js_divergence(pair_probs, text_probs)
        pair_image_js = self._js_divergence(pair_probs, image_probs)
        level_loss = pair_text_js + pair_image_js
        weights = self.consistency_level_weights.to(level_loss.device, level_loss.dtype).view(1, -1)
        return (level_loss * weights).mean()

    def _usage_regularization(self, distance_views) -> torch.Tensor:
        if not distance_views:
            raise ValueError("distance_views must be non-empty")

        probs = [self._distances_to_probs(distances) for distances in distance_views]
        stacked = torch.cat(probs, dim=0)
        level_usage = stacked.mean(dim=0)
        num_codes = level_usage.size(-1)
        uniform = torch.full_like(level_usage, 1.0 / num_codes)
        usage_kl = (
            level_usage.clamp_min(1e-12)
            * (level_usage.clamp_min(1e-12).log() - uniform.clamp_min(1e-12).log())
        ).sum(dim=-1)
        weights = self.consistency_level_weights.to(usage_kl.device, usage_kl.dtype)
        return (usage_kl * weights).mean()

    def _paired_consensus_representation(
        self,
        text_x: torch.Tensor,
        image_x: torch.Tensor,
    ):
        r_text = self.encode(text_x, "text")
        r_image = self.encode(image_x, "image")

        q_text_in = self.pre_quant(r_text)
        q_image_in = self.pre_quant(r_image)
        _, _, _, text_distances = self._codebook_for("text")(q_text_in, use_sk=False)
        _, _, _, image_distances = self._codebook_for("image")(q_image_in, use_sk=False)

        reliability_text = self._reliability_from_distances(text_distances)
        reliability_image = self._reliability_from_distances(image_distances)
        normalization_dtype = (
            torch.float32
            if reliability_text.dtype in (torch.float16, torch.bfloat16)
            else reliability_text.dtype
        )
        reliability_text_norm = reliability_text.to(dtype=normalization_dtype)
        reliability_image_norm = reliability_image.to(dtype=normalization_dtype)
        reliability_sum = (
            reliability_text_norm
            + reliability_image_norm
            + 2.0 * self.reliability_epsilon
        )
        weight_text = (
            (reliability_text_norm + self.reliability_epsilon) / reliability_sum
        ).to(dtype=r_text.dtype)
        weight_image = (
            (reliability_image_norm + self.reliability_epsilon) / reliability_sum
        ).to(dtype=r_image.dtype)

        paired_rep = weight_text * r_text + weight_image * r_image
        return paired_rep, r_text, r_image, text_distances, image_distances, weight_text, weight_image

    def reconstruct_input(self, r_hat: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "text":
            return self.text_reconstruction(r_hat)
        if modality == "image":
            return self.image_reconstruction(r_hat)
        raise ValueError(f"Unsupported modality: {modality}")

    def pair_representation(
        self,
        text_x: torch.Tensor,
        image_x: torch.Tensor,
        fuse_mode: str = "mean",
    ) -> torch.Tensor:
        r_text = self.encode(text_x, "text")
        r_image = self.encode(image_x, "image")

        if fuse_mode == "mean":
            return 0.5 * (r_text + r_image)
        if fuse_mode == "text":
            return r_text
        if fuse_mode == "image":
            return r_image
        if fuse_mode == "consensus":
            paired_rep, *_ = self._paired_consensus_representation(text_x, image_x)
            return paired_rep
        raise ValueError(f"Unsupported fuse_mode: {fuse_mode}")

    def compute_phase1_loss(
        self,
        paired_text: Optional[torch.Tensor] = None,
        paired_image: Optional[torch.Tensor] = None,
        unpaired_image: Optional[torch.Tensor] = None,
        unpaired_text: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        lambda_align: float = 0.1,
        use_sk: bool = True,
    ):
        device = None
        for tensor in [paired_text, paired_image, unpaired_image, unpaired_text]:
            if tensor is not None:
                device = tensor.device
                break
        if device is None:
            raise ValueError("At least one input tensor must be provided.")

        zero = torch.zeros((), device=device)
        quant_loss = zero
        unpaired_loss = zero
        align_loss = zero
        consistency_loss = zero
        usage_loss = zero
        usage_distance_views = []

        if paired_text is not None and paired_image is not None:
            paired_rep, r_text, r_image, text_distances, image_distances, weight_text, weight_image = self._paired_consensus_representation(
                paired_text,
                paired_image,
            )
            rhat_pair, qloss_pair, _, _ = self.quantize(
                paired_rep,
                use_sk=use_sk,
                modality="pair",
            )
            _, _, _, pair_distances = self.quantize(paired_rep, use_sk=False, modality="pair")
            paired_recon = (
                F.mse_loss(self.reconstruct_input(rhat_pair, "text"), paired_text)
                + F.mse_loss(self.reconstruct_input(rhat_pair, "image"), paired_image)
            )
            quant_loss = paired_recon + qloss_pair
            align_loss = F.mse_loss(r_text, r_image)
            usage_distance_views.extend([pair_distances, text_distances, image_distances])
            if self.lambda_consistency > 0:
                consistency_loss = self._consistency_loss(pair_distances, text_distances, image_distances)

        if unpaired_image is not None:
            r_uimg, rhat_uimg, qloss_uimg, _, image_distances_unpaired = self.encode_and_quantize(
                unpaired_image, "image", use_sk=use_sk
            )
            unpaired_loss = (
                unpaired_loss
                + F.mse_loss(self.reconstruct_input(rhat_uimg, "image"), unpaired_image)
                + qloss_uimg
            )
            usage_distance_views.append(image_distances_unpaired)

        if unpaired_text is not None:
            r_utxt, rhat_utxt, qloss_utxt, _, text_distances_unpaired = self.encode_and_quantize(
                unpaired_text, "text", use_sk=use_sk
            )
            unpaired_loss = (
                unpaired_loss
                + F.mse_loss(self.reconstruct_input(rhat_utxt, "text"), unpaired_text)
                + qloss_utxt
            )
            usage_distance_views.append(text_distances_unpaired)

        if self.lambda_usage > 0 and usage_distance_views:
            usage_loss = self._usage_regularization(usage_distance_views)

        total = (
            alpha * quant_loss
            + beta * unpaired_loss
            + lambda_align * align_loss
            + self.lambda_consistency * consistency_loss
            + self.lambda_usage * usage_loss
        )
        stats = {
            "loss_total": float(total.detach().cpu()),
            "loss_quant": float(quant_loss.detach().cpu()),
            "loss_unpaired": float(unpaired_loss.detach().cpu()),
            "loss_align": float(align_loss.detach().cpu()),
            "loss_consistency": float(consistency_loss.detach().cpu()),
            "loss_usage": float(usage_loss.detach().cpu()),
        }
        return total, stats

    @torch.no_grad()
    def get_indices(
        self,
        text_x: Optional[torch.Tensor] = None,
        image_x: Optional[torch.Tensor] = None,
        use_sk: bool = False,
        fuse_mode: str = "mean",
    ):
        if text_x is not None and image_x is not None:
            rep = self.pair_representation(text_x, image_x, fuse_mode=fuse_mode)
            modality = "pair"
        elif text_x is not None:
            rep = self.encode(text_x, "text")
            modality = "text"
        elif image_x is not None:
            rep = self.encode(image_x, "image")
            modality = "image"
        else:
            raise ValueError("Either text_x or image_x must be provided.")

        _, _, indices, distances = self.quantize(rep, use_sk=use_sk, modality=modality)
        return indices, distances


def code_prefixes(num_levels: int):
    letters = string.ascii_lowercase
    if num_levels > len(letters):
        raise ValueError(f"Too many quantization levels: {num_levels}")
    return [f"<{letters[i]}_{{}}>" for i in range(num_levels)]
