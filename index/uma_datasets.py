import os
from typing import Iterable, List

import numpy as np
import torch
import torch.utils.data as data


def _normalize_datasets(datasets) -> List[str]:
    if isinstance(datasets, str):
        datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    return list(datasets)


class PairedEmbeddingDataset(data.Dataset):
    """
    Paired multimodal dataset built from pre-extracted frozen embeddings.

    This matches the UMA-GR paper's setup where frozen feature extractors
    provide modality-specific features and the tokenizer is trained on top of
    them.
    """

    def __init__(
        self,
        data_root: str,
        datasets: Iterable[str],
        text_embedding_file: str,
        image_embedding_file: str,
    ):
        self.data_root = data_root
        self.datasets = _normalize_datasets(datasets)

        text_embeddings_all = []
        image_embeddings_all = []
        self.dataset_count = []

        for dataset in self.datasets:
            text_path = os.path.join(data_root, dataset, f"{dataset}{text_embedding_file}")
            image_path = os.path.join(data_root, dataset, f"{dataset}{image_embedding_file}")

            text_emb = np.load(text_path)
            image_emb = np.load(image_path)
            assert len(text_emb) == len(image_emb), (
                f"[{dataset}] text/image length mismatch: {len(text_emb)} vs {len(image_emb)}"
            )

            text_embeddings_all.append(text_emb)
            image_embeddings_all.append(image_emb)
            self.dataset_count.append(len(text_emb))

        self.text_embeddings = np.concatenate(text_embeddings_all, axis=0)
        self.image_embeddings = np.concatenate(image_embeddings_all, axis=0)
        self._text_dim = self.text_embeddings.shape[-1]
        self._image_dim = self.image_embeddings.shape[-1]

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, index):
        return {
            "text": torch.FloatTensor(self.text_embeddings[index]),
            "image": torch.FloatTensor(self.image_embeddings[index]),
        }

    @property
    def text_dim(self):
        return self._text_dim

    @property
    def image_dim(self):
        return self._image_dim


class SingleModalEmbeddingDataset(data.Dataset):
    """
    Unpaired modality pool for Phase 1 MAQE pre-training.

    The dataset may come from the same paired domains or from additional
    related domains. Only one modality is exposed to the tokenizer.
    """

    def __init__(
        self,
        data_root: str,
        datasets: Iterable[str],
        embedding_file: str,
        modality: str,
    ):
        self.data_root = data_root
        self.datasets = _normalize_datasets(datasets)
        self.modality = modality

        embeddings_all = []
        self.dataset_count = []

        for dataset in self.datasets:
            emb_path = os.path.join(data_root, dataset, f"{dataset}{embedding_file}")
            emb = np.load(emb_path)
            embeddings_all.append(emb)
            self.dataset_count.append(len(emb))

        self.embeddings = np.concatenate(embeddings_all, axis=0)
        self._dim = self.embeddings.shape[-1]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return torch.FloatTensor(self.embeddings[index])

    @property
    def dim(self):
        return self._dim
