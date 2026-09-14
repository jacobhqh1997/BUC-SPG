"""Device selection without hardware inventory, identifiers or environment dumps.

No changes to CUDA_VISIBLE_DEVICES. CUDA indices refer to process-visible GPUs,
not physical machine numbering. Explicit unavailable requests raise an error.
"""

from contextlib import nullcontext
import torch


def select_device(preference='auto'):
    """Accept auto, cpu, cuda, cuda:N or mps. Auto prefers CUDA, then MPS, CPU.

    Never chooses devices based on other users' processes or prints GPU details.
    """
    mps = getattr(torch.backends, 'mps', None)
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda', torch.cuda.current_device())
        if mps is not None and mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    device = torch.device(preference)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('Requested CUDA device is unavailable')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError('Requested CUDA index is unavailable')
    elif device.type == 'mps':
        if mps is None or not mps.is_available():
            raise ValueError('Requested MPS device is unavailable')
    elif device.type != 'cpu':
        raise ValueError('Supported device types: cpu, cuda, mps')
    return device


def move_to_device(value, device):
    """Recursively move tensor batches; other objects are returned unchanged.

    This is NOT a metadata anonymizer. Remove private fields upstream.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def autocast_context(device, enabled=False):
    """Optional CUDA bfloat16 only; otherwise keep FP32 (no scaler required).

    Compute survival logarithms outside this context in FP32.
    """
    device = torch.device(device)
    if enabled and device.type == 'cuda':
        with torch.cuda.device(device):
            supported = torch.cuda.is_bf16_supported()
        if supported:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()
