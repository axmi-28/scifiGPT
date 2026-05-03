"""
Sanity check for the Gutenberg sci-fi nanoGPT dataset files.

Run from the project root:
    python scripts/check_gutenberg_setup.py
"""

import json
import os
import sys


DATA_DIR = os.path.join("data", "gutenberg_scifi")
TRAIN_PATH = os.path.join(DATA_DIR, "train.bin")
VAL_PATH = os.path.join(DATA_DIR, "val.bin")
INFO_PATH = os.path.join(DATA_DIR, "dataset_info.json")
NEXT_COMMAND = "python train.py config/train_gutenberg_scifi_tiny.py"


def require_nonempty_file(path: str) -> bool:
    if not os.path.exists(path):
        print(f"missing: {path}")
        return False
    if os.path.getsize(path) == 0:
        print(f"empty: {path}")
        return False
    print(f"ok: {path} ({os.path.getsize(path):,} bytes)")
    return True


def print_dataset_info() -> None:
    if not os.path.exists(INFO_PATH):
        print(f"dataset summary not found: {INFO_PATH}")
        return

    with open(INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)

    print("\nDataset summary")
    print(f"  dataset_name: {info.get('dataset_name', 'unknown')}")
    print(f"  books: {info.get('num_books_used', 'unknown')} total, {info.get('num_train_books', 'unknown')} train, {info.get('num_val_books', 'unknown')} val")
    print(f"  tokens: {info.get('total_tokens', 'unknown')} total, {info.get('train_tokens', 'unknown')} train, {info.get('val_tokens', 'unknown')} val")

    preview = info.get("preview") or []
    if preview:
        print("  preview:")
        for book in preview[:3]:
            print(f"    - {book.get('title', 'Unknown')} by {book.get('author', 'Unknown')}")


def main() -> int:
    print("Checking Gutenberg sci-fi setup...\n")
    ok = require_nonempty_file(TRAIN_PATH)
    ok = require_nonempty_file(VAL_PATH) and ok
    print_dataset_info()

    print(f"\nNext training command:\n  {NEXT_COMMAND}")
    if not ok:
        print("\nRun this first:\n  python data/gutenberg_scifi/prepare.py --max_books 100")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
