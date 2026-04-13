"""
BGL：用有监督 SimCSE 编码词表（BGL_df-ID.csv），池化为 first_last_avg，保存原始语义向量 .npy。

行布局与下游一致：输出形状为 (max(EventID), hidden_dim)，第 e-1 行对应 EventID == e 的模板
（与 log_dataset 中 embedding_arr[template_id - 1] 的用法一致；未出现的 ID 行为全零）。

在仓库根目录无参运行：python BGL/BGL/embed_simcse_first_last.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _norm_header(h: str) -> str:
    return (h or "").strip().lower()


def load_id_csv(path: Path) -> List[Tuple[int, str]]:
    import csv

    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} 无表头")
        cols = {_norm_header(h): h for h in reader.fieldnames}
        c_key = cols.get("content") or cols.get("template")
        id_key = cols.get("eventid") or cols.get("contentid") or cols.get("template_id")
        if not c_key or not id_key:
            raise ValueError(f"{path} 需要 content 与 EventID（或等价列名）")
        for row in reader:
            text = (row.get(c_key) or "").strip()
            if not text:
                continue
            try:
                eid = int(str(row.get(id_key, "")).strip())
            except ValueError:
                continue
            rows.append((eid, text))
    if not rows:
        raise ValueError(f"{path} 无有效行")
    rows.sort(key=lambda x: x[0])
    return rows


def first_last_avg(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """last_hidden: [B, L, H]；对每条序列取首 token 与末有效 token 的向量平均。"""
    first = last_hidden[:, 0, :]
    seq_lens = attention_mask.sum(dim=1).long() - 1
    seq_lens = seq_lens.clamp(min=0)
    b = torch.arange(last_hidden.size(0), device=last_hidden.device, dtype=torch.long)
    last = last_hidden[b, seq_lens, :]
    return (first + last) * 0.5


@torch.inference_mode()
def encode_texts(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        hidden = out.last_hidden_state
        emb = first_last_avg(hidden, enc["attention_mask"])
        outs.append(emb.float().cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def load_simcse_encoder(
    model_name: str,
    device: str | None,
) -> Tuple[torch.nn.Module, object, torch.device]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(dev)
    return model, tok, dev


def build_embedding_table(
    pairs: List[Tuple[int, str]],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str | None,
) -> Tuple[np.ndarray, int]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(dev)

    max_eid = max(e for e, _ in pairs)
    texts = [t for _, t in pairs]
    vecs = encode_texts(texts, model, tok, dev, batch_size, max_length)

    dim = vecs.shape[1]
    table = np.zeros((max_eid, dim), dtype=np.float32)
    for (eid, _), row in zip(pairs, vecs):
        if eid <= 0:
            raise ValueError(f"EventID 须为正整数，收到 {eid}")
        table[eid - 1] = row
    return table, max_eid


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    default_id = here.parent / "output" / "bgl" / "BGL_df-ID.csv"
    default_out = repo_root / "WhiteningNpy" / "bgl_simcse_first_last_raw.npy"
    parser = argparse.ArgumentParser(
        description="SimCSE 有监督 + first_last_avg → raw .npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--id-csv", type=Path, default=default_id, help="BGL_df-ID.csv")
    parser.add_argument("--out-npy", type=Path, default=default_out, help="输出 .npy")
    parser.add_argument(
        "--model",
        type=str,
        default="princeton-nlp/sup-simcse-bert-base-uncased",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.id_csv.is_file():
        raise FileNotFoundError(args.id_csv)

    pairs = load_id_csv(args.id_csv)
    print(f"词表: {len(pairs)} 条，max(EventID)={max(e for e, _ in pairs)}")
    print(f"模型: {args.model} | 池化: first_last_avg")

    table, max_eid = build_embedding_table(
        pairs,
        args.model,
        args.batch_size,
        args.max_length,
        args.device,
    )

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, table)
    print(f"已保存 {table.shape} -> {args.out_npy}（行 e-1 对应 EventID=e）")


if __name__ == "__main__":
    main()
