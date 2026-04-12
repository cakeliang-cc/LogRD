# LogRD

HDFS log anomaly detection with a BERT-style encoder over discrete log keys. Training and inference use a **precomputed template embedding matrix** (`.npy`) aligned with integer `contentId` values in the CSV sessions.

## Repository layout

| Path | Description |
|------|-------------|
| `HDFS/logbert.py` | CLI: `process` (build CSVs + embeddings), `train`, `predict` |
| `HDFS/hdfs_process.py` | Raw HDFS log → CSVs, `all_vocab.csv`, optional SimCSE vectors |
| `HDFS/PIPELINE_CLEAN_TEMPLATES.md` | Short pipeline notes (clean-based templates) |
| `bert_pytorch/` | Datasets, trainer, `data_clean.py` (`clean()`, legacy vector `load_HDFS`) |
| `Net/` | BERT backbone and log-specific head |
| `output/hdfs/` | Default artifacts: `train.csv`, `test_normal.csv`, `testAnomaly.csv`, `train_vocab.csv`, `all_vocab.csv`, `bert/` |
| `output/hdfs/raw/` | Default raw inputs: `HDFS.log`, `anomaly_label.csv` |
| `WhiteningNpy/` | Default embedding file: `HDFS.npy` (row `i` ↔ `contentId == i + 1`) |

**Git:** The repo intentionally **excludes data and generated artifacts** (see `.gitignore`: `output/`, `WhiteningNpy/`, `*.csv`, `*.npy`, weights, logs, etc.). Obtain HDFS raw files locally and run `python HDFS/logbert.py process` to recreate CSVs and embeddings.

## Requirements

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Install **PyTorch** matching your platform and CUDA stack from [pytorch.org](https://pytorch.org) if the wheels from PyPI are not suitable.

**Note:** Importing `bert_pytorch/data_clean.py` loads TensorFlow and several Hugging Face **TensorFlow** weights at module import time. For a lighter `process` path you can lazy-import or vendor only the `clean()` function.

## Usage

Run from the **repository root** (`logbert.py` adds the repo root to `sys.path`).

### 1. `process` — build datasets and embeddings

Reads `HDFS.log` and `anomaly_label.csv`, derives template IDs via `clean()`, writes CSVs and `all_vocab.csv`, and (by default) encodes each template with a supervised SimCSE checkpoint from Hugging Face into `WhiteningNpy/HDFS.npy`.

```bash
python HDFS/logbert.py process
```

Useful flags:

- `--log`, `--label`, `--out_dir` — override paths (defaults are relative to `HDFS/`)
- `--train_ratio` — fraction of **Normal** blocks for training (default `0.8`)
- `--split {sequential,random}` — how Normal blocks are split
- `--no-embed` — CSV + `all_vocab.csv` only; skip model download and `HDFS.npy`
- `--embed_model` — default `princeton-nlp/sup-simcse-bert-base-uncased`
- `--whitening_npy` — output path for the embedding matrix

### 2. `train`

Requires `output/hdfs/train.csv`, `train_vocab.csv`, and the numpy file at `options["embedding_path"]` (default `../WhiteningNpy/HDFS.npy`). The embedding dimension must match `options["hidden"]` (default `768`).

```bash
python HDFS/logbert.py train
```

Checkpoints and options are written under `output/hdfs/bert/` (e.g. `best_bert.pth`, `parameters.txt`).

### 3. `predict`

```bash
python HDFS/logbert.py predict
```

Uses `test_normal.csv`, `testAnomaly.csv`, and the same embedding / vocab conventions as `bert_pytorch/predict_log.py`.

## Data conventions

- Session integers are **1-based** `contentId` values. `LogDataset` indexes `embedding_arr` with `contentId - 1`.
- `train_vocab.csv` maps `contentId` → `vocabId` for tokens seen in training (classification head).
- `all_vocab.csv` lists `template_id` and `Template` (cleaned key). `HDFS.npy` rows follow sorted `template_id` order.

## Legacy path

`bert_pytorch/data_clean.py` → `load_HDFS()` builds **per-line dense BERT (etc.) vectors** and saves `.npz`. That pipeline is **not** the same as the discrete-ID + `.npy` template matrix used by `logbert.py`; do not mix intermediates between them.
