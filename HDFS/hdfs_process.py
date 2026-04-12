"""
HDFS：clean → 模板 ID → CSV +（可选）SimCSE 模板向量。
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bert_pytorch.data_clean import clean  # noqa: E402

BLK_PATTERN = re.compile(r"(blk_-?\d+)")


def _parse_timestamp(line: str) -> float | None:
    parts = line.split()
    if len(parts) < 2:
        return None
    ts = " ".join(parts[:2])
    try:
        return datetime.strptime(ts, "%y%m%d %H%M%S").timestamp()
    except ValueError:
        return None


def _extract_message(line: str) -> str:
    """去掉日期、时间、pid、级别后的文本（从第 5 个 token 起）。"""
    parts = line.split()
    if len(parts) >= 5:
        return " ".join(parts[4:])
    return line.strip()


def _iter_hdfs_events(log_path: str):
    """每行产出 (timestamp, block_id, template_key) 或 None（跳过）。"""
    with open(log_path, mode="r", encoding="utf8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts = _parse_timestamp(line)
            if ts is None:
                continue
            blks = BLK_PATTERN.findall(line)
            blks = list(dict.fromkeys(blks))
            if len(blks) != 1:
                continue
            msg = _extract_message(line)
            key = clean(msg).lower().strip()
            if not key:
                key = "__EMPTY__"
            yield ts, blks[0], key


def _collect_unique_templates(log_path: str) -> list[str]:
    keys = set()
    for _, _, key in _iter_hdfs_events(log_path):
        keys.add(key)
    return sorted(keys)


def _assign_template_ids(sorted_keys: list[str]) -> dict[str, int]:
    return {k: i + 1 for i, k in enumerate(sorted_keys)}


def build_block_sequences(log_path: str, key_to_id: dict[str, int]) -> dict[str, list[int]]:
    blocks: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for ts, blk, key in _iter_hdfs_events(log_path):
        tid = key_to_id[key]
        blocks[blk].append((ts, tid))
    out: dict[str, list[int]] = {}
    for blk, pairs in blocks.items():
        pairs.sort(key=lambda x: x[0])
        out[blk] = [t for _, t in pairs]
    return out


def _split_normal_block_ids(
    normal_ids: list[str],
    train_ratio: float,
    seed: int,
    split: str,
) -> tuple[list[str], list[str]]:
    n = len(normal_ids)
    k = int(n * train_ratio)
    k = max(0, min(k, n))
    ids = list(normal_ids)
    if split == "random":
        rng = np.random.RandomState(seed)
        rng.shuffle(ids)
    else:
        ids.sort()
    return ids[:k], ids[k:]


def write_csvs(
    block_seqs: dict[str, list[int]],
    label_path: str,
    out_dir: str,
    train_ratio: float,
    seed: int,
    split: str,
) -> None:
    labels = pd.read_csv(label_path, engine="c", na_filter=False)
    labels = labels.set_index("BlockId")["Label"].to_dict()

    normal = [b for b in block_seqs if labels.get(b) == "Normal"]
    anomaly = [b for b in block_seqs if labels.get(b) == "Anomaly"]

    missing_label = set(block_seqs) - set(labels)
    if missing_label:
        print(f"警告：{len(missing_label)} 个块在日志中有序列但不在 label 文件中，已忽略。")

    train_ids, test_norm_ids = _split_normal_block_ids(normal, train_ratio, seed, split)

    def rows(block_list: list[str]) -> list[tuple]:
        r = []
        for bid in block_list:
            seq = block_seqs[bid]
            r.append((bid, ", ".join(map(str, seq))))
        return r

    os.makedirs(out_dir, exist_ok=True)
    train_df = pd.DataFrame(rows(train_ids), columns=["BlockId", "contentIds"])
    test_n_df = pd.DataFrame(rows(test_norm_ids), columns=["BlockId", "contentIds"])
    ano_rows = [(bid, ", ".join(map(str, block_seqs[bid])), labels[bid]) for bid in anomaly]
    ano_df = pd.DataFrame(ano_rows, columns=["BlockId", "contentIds", "Label"])

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_n_df.to_csv(os.path.join(out_dir, "test_normal.csv"), index=False)
    ano_df.to_csv(os.path.join(out_dir, "testAnomaly.csv"), index=False)

    print(
        f"train.csv: {len(train_df)}  Normal 块 | test_normal.csv: {len(test_n_df)} | "
        f"testAnomaly.csv: {len(ano_df)}"
    )


def write_all_vocab(sorted_keys: list[str], key_to_id: dict[str, int], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["template_id", "Template"])
        for k in sorted_keys:
            w.writerow([key_to_id[k], k])
    print(f"all_vocab.csv: {len(sorted_keys)} 条模板 -> {out_path}")


def write_train_vocab(train_csv: str, out_path: str) -> None:
    train = pd.read_csv(train_csv)
    ids = set()
    for cell in train["contentIds"].astype(str):
        for part in cell.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
    sorted_ids = sorted(ids)
    df = pd.DataFrame(
        {"contentId": sorted_ids, "vocabId": range(1, len(sorted_ids) + 1)}
    )
    df.to_csv(out_path, index=False)
    print(f"train_vocab.csv: {len(df)} 个 contentId -> {out_path}")


def embed_templates_simcse(
    all_vocab_csv: str,
    out_npy: str,
    model_name: str,
    batch_size: int = 64,
    device: str | None = None,
) -> None:
    """用 HuggingFace 上有监督 SimCSE 权重对 Template 列编码，行顺序与 template_id 1..N 一致。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    df = pd.read_csv(all_vocab_csv)
    if "template_id" not in df.columns or "Template" not in df.columns:
        raise ValueError("all_vocab.csv 需要列 template_id, Template")

    df = df.sort_values("template_id").reset_index(drop=True)
    texts = df["Template"].astype(str).tolist()
    n = len(texts)
    if n == 0:
        raise ValueError("all_vocab 为空")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    all_emb = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model(**enc)
            # SimCSE / BERT：用 [CLS] 或 mean last_hidden_state；与常见 SimCSE 推理一致用 CLS
            h = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_emb.append(h)
    arr = np.vstack(all_emb).astype(np.float32)
    if arr.shape[0] != n:
        raise RuntimeError("嵌入行数与模板数不一致")

    os.makedirs(os.path.dirname(out_npy) or ".", exist_ok=True)
    np.save(out_npy, arr)
    print(f"已保存 SimCSE 嵌入 {arr.shape} -> {out_npy}")


def run_process(
    log_path: str,
    label_path: str,
    out_dir: str,
    train_ratio: float = 0.8,
    seed: int = 1234,
    split: str = "sequential",
    embed: bool = True,
    embed_model: str = "princeton-nlp/sup-simcse-bert-base-uncased",
    embed_batch_size: int = 64,
    whitening_npy: str | None = None,
) -> None:
    print("Pass 1/2: 收集唯一模板键 …")
    sorted_keys = _collect_unique_templates(log_path)
    key_to_id = _assign_template_ids(sorted_keys)
    all_vocab_path = os.path.join(out_dir, "all_vocab.csv")
    write_all_vocab(sorted_keys, key_to_id, all_vocab_path)

    print("Pass 2/2: 按块聚合序列 …")
    block_seqs = build_block_sequences(log_path, key_to_id)
    write_csvs(block_seqs, label_path, out_dir, train_ratio, seed, split)

    train_csv = os.path.join(out_dir, "train.csv")
    write_train_vocab(train_csv, os.path.join(out_dir, "train_vocab.csv"))

    if embed:
        target = whitening_npy or os.path.join(str(_ROOT), "WhiteningNpy", "HDFS.npy")
        target = os.path.normpath(os.path.abspath(target))
        embed_templates_simcse(
            all_vocab_path,
            target,
            model_name=embed_model,
            batch_size=embed_batch_size,
        )
    else:
        print("已跳过 SimCSE 嵌入（--no-embed）。")
