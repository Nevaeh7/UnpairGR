import torch


AUTO_DEVICE_VALUES = {"", "auto", "default", None}


def is_mps_available():
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def get_preferred_device(device=None, gpu_id=0):
    if isinstance(device, torch.device):
        return device

    if device not in AUTO_DEVICE_VALUES:
        return torch.device(device)

    if gpu_id == -1:
        return torch.device("cpu")

    if is_mps_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id if gpu_id is not None else 0)

    return torch.device("cpu")


def is_cuda_device(device):
    return torch.device(device).type == "cuda"


def set_seed(seed):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
