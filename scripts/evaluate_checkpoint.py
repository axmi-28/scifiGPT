"""
Evaluate a nanoGPT checkpoint on the validation split.

Run from the project root:
    python scripts/evaluate_checkpoint.py --out_dir=out-gutenberg-scifi-mps --device=cpu
"""

import argparse
import io
import json
import math
import os
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import GPT, GPTConfig  # noqa: E402


def validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")


def tensor_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def load_checkpoint(checkpoint_path: str, device: str) -> tuple[GPT, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    gptconf = GPTConfig(**checkpoint["model_args"])
    with redirect_stdout(io.StringIO()):
        model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, checkpoint


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data_path: str,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str,
) -> float:
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    ctx = nullcontext()
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()

    return float(losses.mean().item())


def print_report(report: dict[str, Any]) -> None:
    print("Checkpoint evaluation")
    print(f"  checkpoint: {report['checkpoint_path']}")
    print(f"  dataset: {report['dataset']}")
    print(f"  split: {report['split']}")
    print(f"  iteration: {report['iteration']}")
    print(f"  validation loss: {report['validation_loss']:.4f}")
    print(f"  validation perplexity: {report['validation_perplexity']:.2f}")
    print(f"  best validation loss in checkpoint: {report['best_val_loss']:.4f}")
    print(f"  parameter count: {report['parameter_count']:,}")
    print("  model config:")
    for key, value in report["model_config"].items():
        print(f"    {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="out-gutenberg-scifi-mps")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit ckpt.pt path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args()

    validate_device(args.device)
    checkpoint_path = args.checkpoint or os.path.join(args.out_dir, "ckpt.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    torch.manual_seed(1337)
    model, checkpoint = load_checkpoint(checkpoint_path, args.device)

    config = checkpoint.get("config", {})
    dataset = config.get("dataset", "unknown")
    if dataset == "unknown":
        raise ValueError("Checkpoint config does not include a dataset name.")

    data_path = os.path.join("data", dataset, f"{args.split}.bin")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No {args.split} split found at {data_path}")

    model_config = checkpoint["model_args"]
    block_size = int(model_config["block_size"])
    val_loss = estimate_loss(
        model=model,
        data_path=data_path,
        batch_size=args.batch_size,
        block_size=block_size,
        eval_iters=args.eval_iters,
        device=args.device,
    )

    report = {
        "checkpoint_path": checkpoint_path,
        "dataset": dataset,
        "split": args.split,
        "iteration": int(checkpoint.get("iter_num", -1)),
        "validation_loss": val_loss,
        "validation_perplexity": math.exp(val_loss),
        "best_val_loss": tensor_to_float(checkpoint.get("best_val_loss")),
        "parameter_count": model.get_num_params(),
        "model_config": model_config,
        "training_config": config,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
