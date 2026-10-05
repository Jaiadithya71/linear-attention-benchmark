import datetime
import json
import platform
import subprocess
import sys
import torch


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def require_cuda(expect=None):
    """Fail loudly: no CPU fallback, no silently-weaker GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark never runs on CPU "
                           "(timing/memory would be meaningless). Connect a GPU runtime.")
    name = torch.cuda.get_device_name(0)
    if expect and expect.lower() not in name.lower():
        raise RuntimeError(f"Expected a GPU matching '{expect}' but got '{name}'. Refusing to run.")
    return name


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unavailable"


def gather(stage, started, finished, args):
    props = torch.cuda.get_device_properties(0)
    try:
        import transformers
        tv = transformers.__version__
    except Exception:
        tv = "not installed"
    return {
        "stage": stage,
        "started_utc": started,
        "finished_utc": finished,
        "gpu_name": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "transformers_version": tv,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") != "",
        "argv": sys.argv,
        "args": vars(args),
    }


def write_config(path, cfg):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
