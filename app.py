"""
Streamlit UI for inspecting and sampling a trained SciFi-GPT checkpoint.

Run from the project root:
    streamlit run app.py
"""

import glob
import math
import os
import pickle
from contextlib import nullcontext
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
import streamlit as st
import tiktoken
import torch

from model import GPT, GPTConfig
from scripts.parse_training_log import default_output_path, parse_log, summarize, write_csv


DEFAULT_OUT_DIR = "out-gutenberg-scifi-mps"
DEFAULT_LOG_PATH = "train_gutenberg_mps.log"
DEFAULT_METRICS_PATH = default_output_path(DEFAULT_LOG_PATH)


def validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine. Try device='cpu'.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine. Try device='cpu'.")


def load_tokenizer(
    checkpoint: dict,
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    meta = None
    dataset = checkpoint.get("config", {}).get("dataset")
    if dataset:
        meta_path = os.path.join("data", dataset, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

    if meta and meta.get("tokenizer") != "gpt2" and "stoi" in meta:
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda ids: "".join([itos[i] for i in ids])
        return encode, decode

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda ids: enc.decode(ids)
    return encode, decode


def tensor_to_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


@st.cache_data(show_spinner="Reading checkpoint metadata...")
def load_checkpoint_info(out_dir: str) -> Optional[dict]:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", {})
    parameter_count = sum(value.numel() for value in state_dict.values())
    stat = os.stat(ckpt_path)
    best_val_loss = tensor_to_float(checkpoint.get("best_val_loss"))

    return {
        "path": ckpt_path,
        "size_mb": stat.st_size / (1024 * 1024),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "iter_num": checkpoint.get("iter_num"),
        "best_val_loss": best_val_loss,
        "best_val_perplexity": math.exp(best_val_loss) if best_val_loss else None,
        "parameter_count": parameter_count,
        "config": checkpoint.get("config", {}),
        "model_args": checkpoint.get("model_args", {}),
    }


@st.cache_resource(show_spinner="Loading checkpoint...")
def load_model(out_dir: str, device: str):
    validate_device(device)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    encode, decode = load_tokenizer(checkpoint)
    return model, encode, decode


def generate_text(
    model: GPT,
    encode,
    decode,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: str,
) -> str:
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    device_type = "cuda" if "cuda" in device else "cpu"
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type)
    )

    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(y[0].tolist())


@st.cache_data(show_spinner=False)
def load_metrics(metrics_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(metrics_path):
        return None

    df = pd.read_csv(metrics_path)
    numeric_columns = [
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
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def latest_value(df: pd.DataFrame, column: str) -> Optional[float]:
    if column not in df:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def configured_lr(iteration: int, config: dict) -> Optional[float]:
    learning_rate = config.get("learning_rate")
    if learning_rate is None:
        return None
    if not config.get("decay_lr", True):
        return float(learning_rate)

    warmup_iters = int(config.get("warmup_iters", 0))
    lr_decay_iters = int(config.get("lr_decay_iters", config.get("max_iters", iteration)))
    min_lr = float(config.get("min_lr", learning_rate))
    learning_rate = float(learning_rate)

    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / (warmup_iters + 1)
    if iteration > lr_decay_iters:
        return min_lr

    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def chart(df: pd.DataFrame, columns: list[str], title: str) -> None:
    available = [column for column in columns if column in df.columns]
    if not available:
        return

    plot_df = df[["iteration", *available]].dropna(how="all", subset=available)
    if plot_df.empty:
        return

    st.subheader(title)
    st.line_chart(plot_df.set_index("iteration"))


def saved_sample_paths(out_dir: str) -> list[str]:
    candidates = glob.glob("samples/*.txt")
    candidates.extend(glob.glob(os.path.join(out_dir, "*.txt")))
    return sorted(set(candidates))


def render_metric_cards(df: pd.DataFrame) -> None:
    final_train_loss = latest_value(df, "eval_train_loss")
    if final_train_loss is None:
        final_train_loss = latest_value(df, "train_loss")
    final_val_loss = latest_value(df, "val_loss")
    final_perplexity = latest_value(df, "val_perplexity")
    if final_perplexity is None and final_val_loss is not None:
        final_perplexity = math.exp(final_val_loss)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Final Train Loss",
        f"{final_train_loss:.4f}" if final_train_loss is not None else "n/a",
    )
    col2.metric(
        "Final Validation Loss",
        f"{final_val_loss:.4f}" if final_val_loss is not None else "n/a",
    )
    col3.metric(
        "Final Validation Perplexity",
        f"{final_perplexity:.2f}" if final_perplexity is not None else "n/a",
    )


def render_checkpoint_info(info: Optional[dict]) -> None:
    st.subheader("Latest Checkpoint")
    if not info:
        st.info("No checkpoint found in the selected output directory.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Checkpoint Iteration", info.get("iter_num", "n/a"))
    col2.metric(
        "Best Val Loss",
        f"{info['best_val_loss']:.4f}" if info.get("best_val_loss") else "n/a",
    )
    col3.metric("Parameters", f"{info['parameter_count'] / 1e6:.2f}M")

    st.caption(
        f"`{info['path']}` · {info['size_mb']:.1f} MB · modified {info['modified']}"
    )
    with st.expander("Model and training config"):
        st.json({"model_args": info["model_args"], "config": info["config"]})


st.set_page_config(page_title="SciFi-GPT")
st.title("SciFi-GPT")
st.caption(
    "A small GPT-style decoder-only Transformer trained from scratch on Gutenberg science fiction."
)

with st.sidebar:
    out_dir = st.text_input("Checkpoint directory", DEFAULT_OUT_DIR)
    metrics_path = st.text_input("Metrics CSV", DEFAULT_METRICS_PATH)
    log_path = st.text_input("Training log", DEFAULT_LOG_PATH)

    if st.button("Parse log to metrics CSV"):
        if not os.path.exists(log_path):
            st.error(f"No log found at `{log_path}`.")
        else:
            rows = parse_log(log_path)
            write_csv(rows, metrics_path)
            load_metrics.clear()
            st.success(summarize(rows))

    st.divider()
    device = st.selectbox("Generation device", ["cpu", "mps", "cuda"], index=0)
    max_new_tokens = st.slider(
        "Max new tokens", min_value=16, max_value=1000, value=200, step=16
    )
    temperature = st.slider(
        "Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.05
    )
    top_k_value = st.number_input(
        "Top-k (0 disables)", min_value=0, max_value=1000, value=200, step=10
    )

checkpoint_info = load_checkpoint_info(out_dir)
metrics_df = load_metrics(metrics_path)

dashboard_tab, generate_tab = st.tabs(["Dashboard", "Generate"])

with dashboard_tab:
    render_checkpoint_info(checkpoint_info)

    if metrics_df is None:
        st.info(
            "No metrics CSV found yet. Use the sidebar button or run "
            f"`python scripts/parse_training_log.py {log_path}`."
        )
    else:
        render_metric_cards(metrics_df)
        chart(metrics_df, ["train_loss"], "Training Minibatch Loss")
        chart(metrics_df, ["val_loss"], "Validation Loss")
        chart(metrics_df, ["eval_train_loss", "val_loss"], "Train vs Validation Loss")
        chart(metrics_df, ["train_perplexity", "val_perplexity"], "Perplexity")

        config = checkpoint_info["config"] if checkpoint_info else {}
        if config:
            lr_df = metrics_df[["iteration"]].dropna().copy()
            lr_df["configured_learning_rate"] = lr_df["iteration"].apply(
                lambda value: configured_lr(int(value), config)
            )
            chart(lr_df, ["configured_learning_rate"], "Configured Learning Rate")
        elif "learning_rate" in metrics_df.columns:
            chart(metrics_df, ["learning_rate"], "Logged Learning Rate")

    sample_paths = saved_sample_paths(out_dir)
    st.subheader("Saved Generated Samples")
    if sample_paths:
        for path in sample_paths[:3]:
            with st.expander(path):
                with open(path, "r", encoding="utf-8") as f:
                    st.text(f.read())
    else:
        st.caption("No saved sample text files found. Use the Generate tab for live samples.")

with generate_tab:
    prompt = st.text_area("Prompt", value="The machine began to dream", height=120)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        st.info(
            f"No checkpoint found at `{ckpt_path}`. Train a model first, for example:\n\n"
            "`python train.py config/train_gutenberg_scifi_tiny.py`"
        )
    elif st.button("Generate", type="primary"):
        try:
            loaded = load_model(out_dir, device)
            if loaded is None:
                st.warning("Checkpoint disappeared while loading. Please check the output directory.")
            else:
                model, encode, decode = loaded
                top_k = None if top_k_value == 0 else int(top_k_value)
                output = generate_text(
                    model=model,
                    encode=encode,
                    decode=decode,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=device,
                )
                st.subheader("Generated Text")
                st.write(output)
        except Exception as exc:
            st.error(f"Could not generate text: {exc}")
