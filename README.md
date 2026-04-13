# LogRD

BERT-style encoder over discrete log keys with a **precomputed template embedding matrix** (`.npy`) for log anomaly detection. The product name is **LogRD**; the entry script is still named `logbert.py` for historical reasons.

## Layout

| Path | Notes |
|------|--------|
| `HDFS/` | HDFS: `logbert.py` (`process` / `train` / `predict`), `hdfs_process.py` |
| `BGL/BGL/` | BGL: `logbert.py`, `bgl_process.py` (raw→CSVs→sequences→SimCSE→whitening), `build_sequences.py`, etc. |
| `bert_pytorch/` | Datasets, train/predict, `data_clean.py` |
| `Net/` | Backbone and log-specific head |
| `output/` | Default dataset outputs (see `.gitignore`; usually not committed) |
| `WhiteningNpy/` | Embedding `.npy` files (ignored by default) |

## Requirements

```bash
pip install -r requirements.txt
```

Install **PyTorch** for your platform from [pytorch.org](https://pytorch.org) if needed.

## HDFS (short)

From the repo root:

```bash
python HDFS/logbert.py process   # raw → CSVs, vocab, SimCSE embeddings, etc.
python HDFS/logbert.py train
python HDFS/logbert.py predict
```

Common flags: `--no-embed` skips embedding; `--log` / `--label` / `--out_dir` override paths.

## BGL (short)

From the repo root or under `BGL/BGL/`:

```bash
python BGL/BGL/logbert.py process
```

Chains raw `BGL.log` → `BGL_df` → session CSVs and `train_vocab` → SimCSE raw `.npy` → whitened `whitened_embeddings_256d.npy`. Use `--skip-raw` if `BGL_df` already exists; use `--augment` for LLM-augmented whitening stats (requires `OPENAI_API_KEY`).

Or run `python BGL/BGL/bgl_process.py` with the same flags as the `process` subcommand.

Train / predict: `python BGL/BGL/logbert.py train` and `predict` (paths in `output/bgl/` and `WhiteningNpy/` per `logbert.py` options).

## Data conventions

- Template IDs in sessions are **contentId** values; `LogDataset` indexes `embedding_arr` with `contentId - 1`.
- `train_vocab.csv` maps `contentId` → `vocabId` for keys seen in training.

## Notes

- Large files and generated artifacts are listed in `.gitignore`; place raw logs locally and run `process` to build artifacts.
- Importing `data_clean.py` may pull heavy dependencies; split out `clean()` only if you need a lighter path.
- Legacy `load_HDFS()`-style dense vector pipelines differ from the current discrete-ID + template-matrix path; do not mix intermediates.
