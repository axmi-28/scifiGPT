"""
Prepare a Gutenberg science fiction corpus for nanoGPT.

The output format matches nanoGPT's GPT-2 BPE datasets: train.bin and val.bin
contain uint16 token ids, and meta.pkl gives train.py the GPT-2 vocabulary size.
"""

import argparse
import ast
import json
import os
import pickle
import random
import re
from typing import Any, Optional

import numpy as np
import tiktoken
from datasets import DatasetDict, load_dataset
from tqdm import tqdm


DEFAULT_DATASET = "stas/gutenberg-100"
FULL_DATASET = "stevez80/Sci-Fi-Books-gutenberg"
VOCAB_SIZE = 50257


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Gutenberg sci-fi text for nanoGPT.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET, help=f"Hugging Face dataset name. Use {FULL_DATASET!r} for the full sci-fi corpus.")
    parser.add_argument("--val_fraction", type=float, default=0.05, help="Fraction of books held out for validation.")
    parser.add_argument("--max_books", type=int, default=None, help="Optional cap for fast debugging runs.")
    parser.add_argument("--min_chars", type=int, default=1000, help="Skip cleaned books shorter than this many characters.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic book shuffling.")
    return parser.parse_args()


def choose_split(dataset: Any):
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        first_split = next(iter(dataset.keys()))
        return dataset[first_split]
    return dataset


def find_column(column_names: list[str], candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in column_names:
            return candidate
    lower_to_original = {name.lower(): name for name in column_names}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def find_text_column(column_names: list[str]) -> str:
    text_column = find_column(column_names, ["Text", "text"])
    if text_column is None:
        raise ValueError(f"Could not find a text column in columns: {column_names}")
    return text_column


def decode_bytes_looking_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return str(value)

    text = value
    stripped = text.strip()
    if stripped.startswith(("b'", 'b"')):
        try:
            literal = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return text
        if isinstance(literal, bytes):
            return literal.decode("utf-8", errors="replace")
    return text


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove common Project Gutenberg wrappers with simple line-marker rules."""
    lines = text.splitlines()
    start_idx = 0
    end_idx = len(lines)

    start_markers = (
        "*** start of the project gutenberg",
        "*** start of this project gutenberg",
        "project gutenberg's",
    )
    end_markers = (
        "*** end of the project gutenberg",
        "*** end of this project gutenberg",
        "end of the project gutenberg",
    )

    for idx, line in enumerate(lines):
        lowered = line.strip().lower()
        if any(marker in lowered for marker in start_markers):
            start_idx = idx + 1
            break

    for idx in range(len(lines) - 1, -1, -1):
        lowered = lines[idx].strip().lower()
        if any(marker in lowered for marker in end_markers):
            end_idx = idx
            break

    if start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx])
    return text


def clean_book(text: Any) -> str:
    text = decode_bytes_looking_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_gutenberg_boilerplate(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_metadata(value: Any) -> str:
    text = decode_bytes_looking_text(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def render_book(book: dict[str, str]) -> str:
    title = book.get("title") or "Unknown"
    author = book.get("author") or "Unknown"
    return f"<|book|>\nTitle: {title}\nAuthor: {author}\n\n{book['text']}"


def encode_books(books: list[dict[str, str]], enc: tiktoken.Encoding, desc: str) -> np.ndarray:
    ids: list[int] = []
    for book in tqdm(books, desc=desc):
        # The visible book separator helps the model learn document boundaries.
        ids.extend(enc.encode_ordinary(render_book(book)))
        ids.append(enc.eot_token)
    return np.array(ids, dtype=np.uint16)


def write_tokens(path: str, tokens: np.ndarray) -> None:
    tokens.tofile(path)


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_fraction < 1:
        raise ValueError("--val_fraction must be in the range [0, 1).")

    out_dir = os.path.dirname(__file__)
    print(f"Loading dataset: {args.dataset_name}")
    raw_dataset = choose_split(load_dataset(args.dataset_name))

    column_names = list(raw_dataset.column_names)
    text_column = find_text_column(column_names)
    title_column = find_column(column_names, ["Title", "title"])
    author_column = find_column(column_names, ["Author", "author"])

    books: list[dict[str, str]] = []
    rows_seen = 0
    for row in tqdm(raw_dataset, desc="cleaning books"):
        rows_seen += 1
        text = clean_book(row[text_column])
        if len(text) < args.min_chars:
            continue

        title = clean_metadata(row[title_column]) if title_column else ""
        author = clean_metadata(row[author_column]) if author_column else ""
        books.append({"title": title, "author": author, "text": text})

        if args.max_books is not None and len(books) >= args.max_books:
            break

    if not books:
        raise RuntimeError(
            f"No books met --min_chars={args.min_chars}. "
            "Try lowering --min_chars or checking the dataset schema."
        )

    rng = random.Random(args.seed)
    rng.shuffle(books)

    if len(books) == 1:
        val_count = 0
    else:
        val_count = max(1, int(round(len(books) * args.val_fraction)))
        val_count = min(val_count, len(books) - 1)

    val_books = books[:val_count]
    train_books = books[val_count:]

    enc = tiktoken.get_encoding("gpt2")
    train_tokens = encode_books(train_books, enc, "tokenizing train books")
    val_tokens = encode_books(val_books, enc, "tokenizing val books")

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "meta.pkl")
    info_path = os.path.join(out_dir, "dataset_info.json")

    write_tokens(train_path, train_tokens)
    write_tokens(val_path, val_tokens)

    meta = {
        "vocab_size": VOCAB_SIZE,
        "tokenizer": "gpt2",
        "dataset": "gutenberg_scifi",
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    preview = [
        {"title": book.get("title") or "Unknown", "author": book.get("author") or "Unknown"}
        for book in books[:10]
    ]
    dataset_info = {
        "dataset_name": args.dataset_name,
        "rows_seen": rows_seen,
        "num_books_used": len(books),
        "num_train_books": len(train_books),
        "num_val_books": len(val_books),
        "train_tokens": int(train_tokens.size),
        "val_tokens": int(val_tokens.size),
        "total_tokens": int(train_tokens.size + val_tokens.size),
        "preview": preview,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print("\nGutenberg sci-fi dataset prepared")
    print(f"  books used: {len(books)} ({len(train_books)} train, {len(val_books)} val)")
    print(f"  train tokens: {train_tokens.size:,}")
    print(f"  val tokens: {val_tokens.size:,}")
    print(f"  wrote: {train_path}")
    print(f"  wrote: {val_path}")
    print(f"  wrote: {meta_path}")
    print(f"  wrote: {info_path}")


if __name__ == "__main__":
    main()
