# SciFiGPT

SciFiGPT is a **small GPT-style decoder-only Transformer trained from scratch** on public-domain science fiction text aggregated from Project Gutenberg / Hugging Face mirrors. It sits on top of [Andrej Karpathy](https://github.com/karpathy/nanoGPT)’s **nanoGPT** and adds a reproducible data pipeline, training configs, sampling, metrics helpers, and a local **Streamlit** playground.

## Current limitations (honest)

- **Not GPT-2 scale**: compact width/depth and modest training budget.
- **Loss around ~4**: on an example Apple Silicon run, train/validation cross-entropy was about **3.98 / 4.01** at the logged checkpoint—useful for learning, not for production text.
- **Imperfect generations**: samples show some genre texture but limited long-range coherence; expect repetition and hallucination typical of small LMs.
- **Artifacts not in git**: `train.bin` / `val.bin` / `meta.pkl` and `out-*/ckpt.pt` are large and are ignored—you must run data prep and training locally (or supply your own checkpoints) before `sample.py` or `app.py` can load a trained weights file.

## What this demonstrates

- Dataset preparation from Hugging Face `datasets`
- GPT-2 BPE tokenization with `tiktoken`
- Causal self-attention in a decoder-only Transformer
- Training from scratch for next-token prediction
- Sampling with temperature and top-k filtering
- Optional checkpoint evaluation (validation loss / perplexity)
- Training-curve plotting from a plain text log → CSV
- A simple local UI with Streamlit

## Dataset

The default quick dataset is `stas/gutenberg-100`, useful for fast setup and debugging. For a larger science fiction corpus, use `stevez80/Sci-Fi-Books-gutenberg`.

`python data/gutenberg_scifi/prepare.py` writes nanoGPT-compatible files:

- `data/gutenberg_scifi/train.bin`
- `data/gutenberg_scifi/val.bin`
- `data/gutenberg_scifi/meta.pkl`
- `data/gutenberg_scifi/dataset_info.json` (small JSON summary committed for provenance; binaries are not)

## Setup

```bash
pip install torch numpy transformers datasets tiktoken tqdm streamlit pandas
```

Or:

```bash
pip install -r requirements-scifi-gpt.txt
```

Use a recent PyTorch build for your platform (**CPU**, **CUDA**, or **MPS** on Apple Silicon).

## Prepare data

Quick default dataset:

```bash
python data/gutenberg_scifi/prepare.py
```

Larger science fiction mix:

```bash
python data/gutenberg_scifi/prepare.py --dataset_name stevez80/Sci-Fi-Books-gutenberg
```

Smoke test:

```bash
python data/gutenberg_scifi/prepare.py --max_books 20
python scripts/check_gutenberg_setup.py
```

## Train

CPU (tiny config):

```bash
python train.py config/train_gutenberg_scifi_tiny.py
```

Apple Silicon (MPS):

```bash
python train.py config/train_gutenberg_scifi_mps.py
```

CUDA:

```bash
python train.py config/train_gutenberg_scifi_cuda.py
```

Checkpoints are written under the config’s `out_dir` (for example `out-gutenberg-scifi-mps`), which is gitignored.

## Sample

```bash
python sample.py --out_dir=out-gutenberg-scifi-tiny --device=cpu --start="The machine began to dream"
python sample.py --out_dir=out-gutenberg-scifi-mps --device=mps --start="Beyond the red planet"
```

Adjust `--out_dir` and `--device` to match whatever you trained.

## Streamlit UI

From the repo root (`nanoGPT/`):

```bash
streamlit run app.py
```

The UI expects `ckpt.pt` inside the selected output directory. It can plot metrics if you point it at a CSV produced from a training log (below).

## Metrics from a training log

Example log `train_gutenberg_mps.log` (from a local MPS run) ships in the repo for reproducibility of the parsing workflow. Convert it to CSV:

```bash
python scripts/parse_training_log.py train_gutenberg_mps.log
```

That writes `metrics/train_gutenberg_mps_metrics.csv`. In the app, choose the matching `out_dir` and metrics file.

## Evaluate a checkpoint

```bash
python scripts/evaluate_checkpoint.py --out_dir=out-gutenberg-scifi-mps --device=cpu
```

This reports validation loss, perplexity, parameter count, and basic checkpoint metadata. Increase `--eval_iters` for a less noisy estimate.

## Interpreting loss around 4

Roughly **4 nats** cross-entropy implies perplexity \(e^4 \approx 55\): the model has picked up local statistics and some genre flavor, but not reliable long-form semantics. That is expected here. Stronger results usually need more compute, data curation, and/or model capacity.

## Portfolio blurb (optional)

You can describe this as a **small nanoGPT-style LM trained from scratch on a Gutenberg science-fiction corpus**, with a visible pipeline from raw books → tokens → training → checkpoints → sampling → a minimal Streamlit demo—and clear caveats about quality.
