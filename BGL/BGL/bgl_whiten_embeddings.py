"""
BGL：对 SimCSE 原始语义向量 .npy 做 BERT-whitening（与
https://github.com/autoliuweijie/BERT-whitening-pytorch 中 all_utils 一致）。

仅用「训练集出现过的模板」对应的向量估计语义均值与 kernel（协方差特征向量 + 1/sqrt(奇异值)），
可选：用 gpt-4o 为每个训练模板生成若干语义相似句，经同一 SimCSE 编码后并入协方差估计，缓解秩亏。
再作用于**全部**行（与 raw npy 行号对齐：第 e-1 行对应 EventID=e）。
全零行（词表中未占位的槽位）不做变换，输出仍为零向量。

在仓库根目录无参运行：python BGL/BGL/bgl_whiten_embeddings.py
LLM 增广默认关闭；需显式传入 --augment 开启，此时应设置 OPENAI_API_KEY 或 --api-key。

默认读 WhiteningNpy/bgl_simcse_first_last_raw.npy，写 WhiteningNpy/whitened_embeddings_256d.npy。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _norm_header(h: str) -> str:
    return (h or "").strip().lower()


def load_train_content_ids(train_vocab_path: Path) -> np.ndarray:
    """train_vocab.csv：contentId, vocabId → 返回升序唯一的 contentId（EventID）。"""
    df = pd.read_csv(train_vocab_path)
    cols = {_norm_header(c): c for c in df.columns}
    cid_col = cols.get("contentid") or cols.get("content_id")
    if not cid_col:
        raise ValueError(f"{train_vocab_path} 需含 contentId 列")
    ids = df[cid_col].astype(int).unique()
    return np.sort(ids)


def compute_kernel_bias(vecs: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    与 BERT-whitening-pytorch all_utils.compute_kernel_bias 一致：
    mu = mean(vecs); cov = cov(vecs.T); SVD(cov) -> W = U @ diag(1/sqrt(s))；bias = -mu。
    vecs: (n, d)，n>=2 为宜。
    """
    if vecs.ndim != 2:
        raise ValueError("vecs 须为二维 (n, d)")
    n = vecs.shape[0]
    if n < 2:
        raise ValueError(
            f"估计白化至少需要 2 条训练模板向量，当前 n={n}。"
            "请检查 train_vocab.csv 与 train.csv。"
        )
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, _ = np.linalg.svd(cov)
    s = np.maximum(s, eps)
    w = u @ np.diag(1.0 / np.sqrt(s))
    bias = -mu
    return w, bias


def transform_and_normalize(
    vecs: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    zero_eps: float = 1e-10,
) -> np.ndarray:
    """(vecs + bias) @ kernel，再按行 L2 归一化（与 all_utils.transform_and_normalize 一致）。"""
    x = (vecs + bias) @ kernel
    norms = (x**2).sum(axis=1, keepdims=True) ** 0.5
    return x / np.maximum(norms, zero_eps)


def whiten_full_matrix(
    raw: np.ndarray,
    train_content_ids: np.ndarray,
    n_components: int,
    eps: float = 1e-12,
    extra_train_vecs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    raw: (max_event_id, d)，行 e-1 对应 EventID e。
    用 train_content_ids 对应的行（及可选 extra_train_vecs）估计 W,bias，再变换全部行；全零行保持为零。
    返回 (whitened_all, kernel[:, :n_components], bias)。
    """
    max_eid = raw.shape[0]
    d = raw.shape[1]

    idx = train_content_ids.astype(np.int64) - 1
    if (idx < 0).any() or (idx >= max_eid).any():
        bad = train_content_ids[(idx < 0) | (idx >= max_eid)]
        raise ValueError(f"train_vocab 中 contentId 超出 raw npy 行范围: {bad[:10]} …")

    train_vecs = raw[idx]
    zero_train = (np.abs(train_vecs).sum(axis=1) < eps)
    if zero_train.any():
        n_bad = int(zero_train.sum())
        raise ValueError(
            f"训练集中有 {n_bad} 个 contentId 在 raw npy 中对应全零行，无法估计白化。"
            "请先完成 embed_simcse_first_last 且保证 EventID 连续覆盖。"
        )

    if extra_train_vecs is not None:
        if extra_train_vecs.ndim != 2 or extra_train_vecs.shape[1] != d:
            raise ValueError(
                f"extra_train_vecs 须为 (m, {d})，当前 {extra_train_vecs.shape}"
            )
        train_vecs = np.vstack([train_vecs, extra_train_vecs.astype(np.float64)])

    w_full, bias = compute_kernel_bias(train_vecs, eps=eps)
    if n_components > w_full.shape[1]:
        raise ValueError(
            f"n_components={n_components} 大于维度 {w_full.shape[1]}"
        )
    kernel = w_full[:, :n_components]

    empty_rows = np.abs(raw).sum(axis=1) < eps
    out = np.zeros((max_eid, n_components), dtype=np.float32)
    if (~empty_rows).any():
        out[~empty_rows] = transform_and_normalize(
            raw[~empty_rows], kernel, bias
        ).astype(np.float32)
    return out, kernel.astype(np.float32), bias.astype(np.float32)


def save_whiten_pkl(path: Path, kernel: np.ndarray, bias: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"kernel": kernel, "bias": bias}, f)


def _augment_prompt(log: str, n: int) -> str:
    return (
        f"You are an expert in system log analysis. Generate {n} semantically similar variations "
        f"of the given log message.\n\n"
        f"Input log: {json.dumps(log)}\n\n"
        "Requirements:\n"
        "- Preserve core meaning, technical info, severity level, and system components\n"
        "- Use synonyms or slight rewording while keeping the structure natural\n"
        "- Variations must be plausible and high-quality\n\n"
        "Return in JSON format:\n"
        "{\n"
        '  "original": "<exactly the input log string>",\n'
        f'  "similar": ["variation 1", "variation 2", "... up to {n}"]\n'
        "}"
    )


def _parse_json_from_chat(text: str) -> Dict[str, Any]:
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    return json.loads(s)


def _openai_similar_variations(
    log: str,
    n: int,
    api_key: str,
    model: str,
) -> List[str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = _augment_prompt(log, n)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    content = (resp.choices[0].message.content or "").strip()
    data = _parse_json_from_chat(content)
    sim = data.get("similar")
    if not isinstance(sim, list):
        raise ValueError("响应 JSON 缺少 similar 数组")
    out = [str(x).strip() for x in sim if str(x).strip()]
    if len(out) < n:
        raise ValueError(f"期望 {n} 条 similar，实际 {len(out)}")
    return out[:n]


def _build_extra_train_embeddings(
    train_content_ids: np.ndarray,
    id_csv: Path,
    n_var: int,
    api_key: str,
    openai_model: str,
    simcse_model: str,
    batch_size: int,
    max_length: int,
    device: Optional[str],
    augment_jsonl: Path,
) -> np.ndarray:
    from embed_simcse_first_last import encode_texts, load_id_csv, load_simcse_encoder

    id_pairs = load_id_csv(id_csv)
    id_to_text = {e: t for e, t in id_pairs}

    records: List[Dict[str, Any]] = []
    flat_texts: List[str] = []

    for cid in train_content_ids.astype(int):
        log = id_to_text.get(int(cid))
        if log is None:
            raise KeyError(f"BGL_df-ID 中缺少 EventID={cid}")
        sims = _openai_similar_variations(log, n_var, api_key, openai_model)
        records.append({"event_id": int(cid), "original": log, "similar": sims})
        flat_texts.extend(sims)

    augment_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with augment_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    model, tok, dev = load_simcse_encoder(simcse_model, device)
    return encode_texts(flat_texts, model, tok, dev, batch_size, max_length)


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    data_dir = here.parent / "output" / "bgl"
    p = argparse.ArgumentParser(
        description="BERT-whitening：train 模板估计参数，作用于全量 raw npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--raw-npy",
        type=Path,
        default=repo_root / "WhiteningNpy" / "bgl_simcse_first_last_raw.npy",
    )
    p.add_argument(
        "--train-vocab",
        type=Path,
        default=data_dir / "train_vocab.csv",
    )
    p.add_argument(
        "--out-npy",
        type=Path,
        default=repo_root / "WhiteningNpy" / "whitened_embeddings_256d.npy",
    )
    p.add_argument("--n-components", type=int, default=256)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument(
        "--params-pkl",
        type=Path,
        default=None,
        help="可选，保存 kernel 与 bias 的 pickle",
    )
    p.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LLM+SimCSE 增广训练协方差（默认关闭；--augment 开启，--no-augment 显式关闭）",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key；默认读环境变量 OPENAI_API_KEY",
    )
    p.add_argument(
        "--id-csv",
        type=Path,
        default=data_dir / "BGL_df-ID.csv",
        help="与 embed 一致的词表，用于 EventID→正文",
    )
    p.add_argument(
        "--augment-jsonl",
        type=Path,
        default=data_dir / "train_template_augmentations.jsonl",
        help="保存 LLM 原始 JSONL",
    )
    p.add_argument("--n-variations", type=int, default=3)
    p.add_argument("--openai-model", type=str, default="gpt-4o")
    p.add_argument(
        "--simcse-model",
        type=str,
        default="princeton-nlp/sup-simcse-bert-base-uncased",
    )
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if not args.raw_npy.is_file():
        raise FileNotFoundError(args.raw_npy)
    if not args.train_vocab.is_file():
        raise FileNotFoundError(args.train_vocab)

    raw = np.load(args.raw_npy)
    if raw.ndim != 2:
        raise ValueError(f"raw npy 须为二维，当前 shape={raw.shape}")

    cids = load_train_content_ids(args.train_vocab)
    print(f"训练模板数（唯一 contentId）: {len(cids)}")
    print(f"raw 形状: {raw.shape} | 输出维: {args.n_components}")

    extra: Optional[np.ndarray] = None
    if args.augment:
        key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("请设置 OPENAI_API_KEY 或使用 --api-key")
        if not args.id_csv.is_file():
            raise FileNotFoundError(args.id_csv)
        print(
            f"OpenAI 增广: model={args.openai_model} | 每模板 {args.n_variations} 条 | "
            f"SimCSE={args.simcse_model}"
        )
        extra = _build_extra_train_embeddings(
            cids,
            args.id_csv,
            args.n_variations,
            key,
            args.openai_model,
            args.simcse_model,
            args.encode_batch_size,
            args.max_length,
            args.device,
            args.augment_jsonl,
        )
        print(f"  增广向量 {extra.shape} | 已写 {args.augment_jsonl}")

    out, kernel, bias = whiten_full_matrix(
        raw.astype(np.float64),
        cids,
        n_components=args.n_components,
        eps=args.eps,
        extra_train_vecs=extra,
    )

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, out)
    print(f"已保存 {out.shape} -> {args.out_npy}")

    if args.params_pkl is not None:
        save_whiten_pkl(args.params_pkl, kernel, bias)
        print(f"已保存 kernel/bias -> {args.params_pkl}")


if __name__ == "__main__":
    main()
