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


def is_mps_device(device):
    return torch.device(device).type == "mps"


def get_transformers_device_map(device, gpu_id=0, local_rank=0):
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None

    if resolved.index is not None:
        return {"": resolved.index}

    if local_rank is not None:
        return {"": local_rank}

    return {"": gpu_id}


def safe_empty_cache(device=None):
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_mps_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        return

    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif resolved.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
