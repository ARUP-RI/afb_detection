import numpy as np
import torch
import pynvml


def get_gpu_mem_usage():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    mem_free_pcts = np.zeros(device_count)
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_free_pcts[i] = mem_info.free / mem_info.total
    pynvml.nvmlShutdown()
    return mem_free_pcts


def get_all_gpu_idxs(min_mem_avail=0.8):
    mem_free_pcts = get_gpu_mem_usage()
    return [i for i, pct in enumerate(mem_free_pcts) if pct >= min_mem_avail]


def get_single_gpu_idx(min_mem_avail=0.8):
    mem_free_pcts = get_gpu_mem_usage()
    idx = np.argmax(mem_free_pcts)
    if mem_free_pcts[idx] < min_mem_avail:
        raise RuntimeError(
            f"All GPUs have less than {min_mem_avail} fraction of memory available!"
        )
    return idx


def choose_single_gpu_if_available(min_mem_avail=0.8, dev=False):
    if dev:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{get_single_gpu_idx(min_mem_avail)}")
    else:
        device = torch.device("cpu")
    return device
