"""
Parse nanoGPT training logs into a small metrics CSV.

Run from the project root:
    python scripts/parse_training_log.py train_gutenberg_mps.log
"""

import argparse
import csv
import math
import os
import re
from typing import Optional


ITER_RE = re.compile(
    r"iter (?P<iteration>\d+): loss (?P<loss>[0-9.]+), "
    r"time (?P<time_ms>[0-9.]+)ms, mfu (?P<mfu_percent>-?[0-9.]+)%"
)
EVAL_RE = re.compile(
    r"step (?P<iteration>\d+): train loss (?P<train_loss>[0-9.]+), "
    r"val loss (?P<val_loss>[0-9.]+)"
)
LR_RE = re.compile(r"(?:^|[, ])(?:lr|learning rate)[: ]+(?P<learning_rate>[0-9.eE+-]+)")

FIELDNAMES = [
    "iteration",
    "train_loss",
    "eval_train_loss",
    "val_loss",
    "train_perplexity",
    "val_perplexity",
    "learning_rate",
    "time_ms",
    "mfu_percent",
]


def safe_exp(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return math.exp(value)


def parse_log(log_path: str) -> list[dict[str, Optional[float]]]:
    rows_by_iter: dict[int, dict[str, Optional[float]]] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            lr_match = LR_RE.search(line)

            iter_match = ITER_RE.search(line)
            if iter_match:
                iteration = int(iter_match.group("iteration"))
                row = rows_by_iter.setdefault(
                    iteration, {field: None for field in FIELDNAMES}
                )
                row["iteration"] = iteration
                row["train_loss"] = float(iter_match.group("loss"))
                row["time_ms"] = float(iter_match.group("time_ms"))
                row["mfu_percent"] = float(iter_match.group("mfu_percent"))
                if lr_match:
                    row["learning_rate"] = float(lr_match.group("learning_rate"))
                continue

            eval_match = EVAL_RE.search(line)
            if eval_match:
                iteration = int(eval_match.group("iteration"))
                row = rows_by_iter.setdefault(
                    iteration, {field: None for field in FIELDNAMES}
                )
                row["iteration"] = iteration
                row["eval_train_loss"] = float(eval_match.group("train_loss"))
                row["val_loss"] = float(eval_match.group("val_loss"))
                if lr_match:
                    row["learning_rate"] = float(lr_match.group("learning_rate"))

    rows = [rows_by_iter[key] for key in sorted(rows_by_iter)]
    for row in rows:
        row["train_perplexity"] = safe_exp(row["eval_train_loss"])
        row["val_perplexity"] = safe_exp(row["val_loss"])
    return rows


def default_output_path(log_path: str) -> str:
    stem = os.path.splitext(os.path.basename(log_path))[0]
    return os.path.join("metrics", f"{stem}_metrics.csv")


def write_csv(rows: list[dict[str, Optional[float]]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Optional[float]]]) -> str:
    if not rows:
        return "No metrics found."

    last_eval = next(
        (row for row in reversed(rows) if row.get("val_loss") is not None), None
    )
    last_train = next(
        (row for row in reversed(rows) if row.get("train_loss") is not None), None
    )

    parts = [f"Parsed {len(rows)} metric rows."]
    if last_train:
        parts.append(
            f"Latest minibatch train loss: {last_train['train_loss']:.4f} "
            f"at iter {int(last_train['iteration'])}."
        )
    if last_eval:
        parts.append(
            f"Latest eval train/val loss: {last_eval['eval_train_loss']:.4f}/"
            f"{last_eval['val_loss']:.4f} at iter {int(last_eval['iteration'])}."
        )
        parts.append(f"Latest validation perplexity: {last_eval['val_perplexity']:.2f}.")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to a nanoGPT training log")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to metrics/<log-name>_metrics.csv",
    )
    args = parser.parse_args()

    rows = parse_log(args.log_path)
    output_path = args.output or default_output_path(args.log_path)
    write_csv(rows, output_path)
    print(f"Wrote {output_path}")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
