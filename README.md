# SciFiGPT

SciFiGPT is a small GPT-style decoder-only Transformer trained from scratch on a Gutenberg-style science-fiction corpus. It is built on Andrej Karpathy's nanoGPT codebase, which I adapted with custom data preparation, training configurations, evaluation tools, loss visualizations, and a Streamlit interface for generation.

## Demo

Screenshot or short demo link coming soon.

To run the local demo after preparing data and training a checkpoint:

```bash
streamlit run app.py
```

## What This Project Is

SciFiGPT is an educational language-modeling project that takes a public-domain science-fiction text corpus through the full workflow: dataset preparation, GPT-2 BPE tokenization, training a compact decoder-only Transformer from scratch, sampling from checkpoints, evaluating validation loss and perplexity, visualizing training behavior, and presenting the model in a Streamlit UI.

This is not a full GPT-2 reproduction and is not presented as production-quality text generation. The goal is to demonstrate the end-to-end mechanics of building and evaluating a small language model on a custom corpus.

## What I Changed From nanoGPT

This repository keeps nanoGPT's readable PyTorch Transformer model and training loop as the foundation, then adapts the project around SciFiGPT-specific data, experiments, and presentation.

My main additions include:

- Gutenberg-style science-fiction dataset preparation in `data/gutenberg_scifi/prepare.py`
- SciFiGPT-specific CPU, Apple Silicon MPS, and CUDA training configs
- local checkpoint sampling workflow for science-fiction prompts
- checkpoint evaluation script for validation loss, perplexity, and parameter counts
- training-log parsing and loss/perplexity CSV generation
- Streamlit generation and dashboard UI in `app.py`
- project documentation that presents SciFiGPT honestly as a nanoGPT-based portfolio project

## Why I Built It

I built SciFiGPT to understand the practical language-modeling pipeline beyond calling a hosted model API. The project focuses on how raw text becomes tokens, how a decoder-only Transformer is trained for next-token prediction, how loss and perplexity relate to generation quality, and how to communicate experimental results without overstating model capability.

## What This Project Demonstrates

- preparing a custom text corpus for language modeling
- tokenizing text with GPT-2 BPE
- training a decoder-only Transformer from scratch
- understanding loss, validation loss, and perplexity
- generating text autoregressively from checkpoints
- visualizing training behavior
- presenting an experimental model honestly

## Current Results

After an overnight local training run of roughly 5000 iterations, the model reached a loss of around 4.0. Generated text often has local grammar and a science-fiction-like tone, but it does not yet maintain reliable long-range coherence.

This is expected for a compact model trained locally with limited compute. The goal of this project is not to reproduce GPT-2-scale performance, but to demonstrate the full language-modeling pipeline from data preparation to training, evaluation, visualization, and interactive generation.

Roughly 4 nats of cross-entropy corresponds to perplexity around `e^4`, or about 55. That means the model has learned useful local token statistics and some genre texture, while still struggling with long-form structure, factual consistency, and stable story arcs.

## Installation

From the repository root:

```bash
pip install -r requirements-scifi-gpt.txt
```

Or install the main dependencies manually:

```bash
pip install torch numpy transformers datasets tiktoken tqdm streamlit pandas
```

Use a recent PyTorch build for your platform. The included configs cover CPU, CUDA, and Apple Silicon MPS.

## Dataset Preparation

The SciFiGPT dataset script writes files in nanoGPT's expected format:

- `data/gutenberg_scifi/train.bin`
- `data/gutenberg_scifi/val.bin`
- `data/gutenberg_scifi/meta.pkl`
- `data/gutenberg_scifi/dataset_info.json`

The binary token files and metadata pickle are generated artifacts and are ignored by git.

Prepare the default quick dataset:

```bash
python data/gutenberg_scifi/prepare.py
```

Prepare the larger science-fiction corpus:

```bash
python data/gutenberg_scifi/prepare.py --dataset_name stevez80/Sci-Fi-Books-gutenberg
```

Run a smaller smoke test:

```bash
python data/gutenberg_scifi/prepare.py --max_books 20
python scripts/check_gutenberg_setup.py
```

## Training

CPU tiny config:

```bash
python train.py config/train_gutenberg_scifi_tiny.py
```

Apple Silicon MPS config:

```bash
python train.py config/train_gutenberg_scifi_mps.py
```

Single-GPU CUDA config:

```bash
python train.py config/train_gutenberg_scifi_cuda.py
```

Checkpoints are written to the config's `out_dir`, such as `out-gutenberg-scifi-mps`. Checkpoint files can be large and are ignored by git.

## Sampling

Sample from a trained CPU checkpoint:

```bash
python sample.py --out_dir=out-gutenberg-scifi-tiny --device=cpu --start="The machine began to dream"
```

Sample from an Apple Silicon MPS checkpoint:

```bash
python sample.py --out_dir=out-gutenberg-scifi-mps --device=mps --start="Beyond the red planet"
```

Adjust `--out_dir`, `--device`, `--temperature`, `--top_k`, and `--max_new_tokens` to match the checkpoint and generation style you want to test.

## Streamlit UI

Run the local interface from the repository root:

```bash
streamlit run app.py
```

The UI loads a selected checkpoint directory, exposes generation controls, and can display parsed training metrics when a metrics CSV is available.

## Evaluation And Visualization

Evaluate a trained checkpoint:

```bash
python scripts/evaluate_checkpoint.py --out_dir=out-gutenberg-scifi-mps --device=cpu
```

Parse a nanoGPT training log into metrics:

```bash
python scripts/parse_training_log.py train_gutenberg_mps.log
```

This writes a CSV such as `metrics/train_gutenberg_mps_metrics.csv`, which can be used by the Streamlit UI to visualize loss and perplexity over training.

## Repository Structure

```text
.
├── app.py                              # Streamlit UI for generation and metrics
├── config/
│   ├── train_gutenberg_scifi_tiny.py   # compact CPU training config
│   ├── train_gutenberg_scifi_mps.py    # Apple Silicon training config
│   └── train_gutenberg_scifi_cuda.py   # single-GPU CUDA training config
├── data/gutenberg_scifi/
│   ├── prepare.py                      # Gutenberg-style corpus preparation
│   └── dataset_info.json               # small dataset provenance summary
├── metrics/                            # parsed training metrics
├── scripts/
│   ├── check_gutenberg_setup.py        # data setup sanity checks
│   ├── evaluate_checkpoint.py          # validation loss/perplexity report
│   └── parse_training_log.py           # training log to CSV parser
├── model.py                            # nanoGPT Transformer model
├── train.py                            # nanoGPT training loop
├── sample.py                           # checkpoint sampling script
└── requirements-scifi-gpt.txt
```

The repository also retains nanoGPT's original examples, including Shakespeare, OpenWebText, GPT-2 evaluation configs, and notebooks. These are useful references for understanding the inherited training code, but the top-level project focus is SciFiGPT.

## Attribution

This project is built on top of Andrej Karpathy's nanoGPT, a minimal implementation of GPT training in PyTorch. I used nanoGPT as the educational research codebase for the Transformer training loop and adapted it into a custom science-fiction language-modeling project.

Original nanoGPT repository:
https://github.com/karpathy/nanoGPT

My contributions in this repo include:

- Gutenberg-style science-fiction dataset preparation
- SciFiGPT-specific training configs
- local CPU/MPS/CUDA training setup
- checkpoint sampling workflow
- evaluation scripts
- loss/perplexity visualization
- Streamlit generation and dashboard UI
- README/documentation rewrite for this project

## nanoGPT Background

nanoGPT is a compact PyTorch codebase for training and finetuning GPT-style models. Its core files, especially `train.py` and `model.py`, provide a readable implementation of the training loop and Transformer architecture. The original nanoGPT project also includes examples for character-level Shakespeare training, OpenWebText preprocessing, GPT-2 evaluation configs, and GPT-2 checkpoint loading.

SciFiGPT uses that codebase as a starting point rather than claiming a from-scratch implementation of every training component.

## Honest Limitations

- The model is compact and trained with local compute, so generations are much weaker than modern hosted LLMs.
- Loss around 4.0 is useful for demonstrating learning, but not enough for reliable long-range coherence.
- The corpus is Gutenberg-style science fiction, so outputs can inherit old public-domain style, artifacts, and biases from the source text.
- Checkpoints, tokenized binary datasets, and local environment files are intentionally not committed.
- The Streamlit UI is a local demo interface, not a production deployment.
- Results depend heavily on training duration, hardware, dataset quality, and sampling settings.

## Future Improvements

- train longer runs with larger model sizes and better hardware
- improve corpus filtering, deduplication, and metadata cleanup
- compare multiple context lengths and model sizes
- add more systematic sample quality evaluation
- track experiments with a structured experiment logger
- publish a demo video or screenshot
- add optional hosted inference once a checkpoint is small enough to serve reliably

## License

This repository inherits nanoGPT's MIT License. See `LICENSE` for details.
