"""
从 raw/BGL.log 生成 BGL_df.csv 与 BGL_df-ID.csv（与 build_sequences 解耦）。

假定行内存在高精度时间字段（形如 2005-06-03-15.42.50.363779），
时间字段后出现重复 node 字段，再之后为 RAS/APP 等开头的消息正文；
正文经 lower 后由 data_clean.clean 得到 Content。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TIME_TOKEN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+$")

_clean_impl = None


def _clean_text(raw: str) -> str:
    global _clean_impl
    if _clean_impl is None:
        from bert_pytorch.data_clean import clean

        _clean_impl = clean
    return _clean_impl(raw)


def _parse_line(line: str) -> Optional[Tuple[str, str, int]]:
    line = line.strip()
    if not line:
        return None
    label = 0 if line[0] == "-" else 1
    parts = line.split()
    time_idx = None
    for i, p in enumerate(parts):
        if _TIME_TOKEN.match(p):
            time_idx = i
            break
    if time_idx is None or time_idx + 2 >= len(parts):
        return None
    time_str = parts[time_idx]
    msg_tokens = parts[time_idx + 2 :]
    raw = " ".join(msg_tokens).lower()
    content = _clean_text(raw).strip()
    if not content:
        content = "__EMPTY__"
    return time_str, content, label


def raw_to_bgl_dfs(
    raw_log_path: Path,
    out_dir: Path,
    bgl_df_name: str = "BGL_df.csv",
    id_name: str = "BGL_df-ID.csv",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_path = out_dir / bgl_df_name
    id_path = out_dir / id_name

    content_to_id: Dict[str, int] = {}
    next_id = 1
    n_ok = 0
    n_skip = 0

    with raw_log_path.open("r", encoding="utf-8", errors="ignore") as fin, df_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        w = csv.writer(fout)
        w.writerow(["Content", "ContentID", "Time", "Label"])
        for line in fin:
            parsed = _parse_line(line)
            if parsed is None:
                n_skip += 1
                continue
            time_str, content, label = parsed
            if content not in content_to_id:
                content_to_id[content] = next_id
                next_id += 1
            cid = content_to_id[content]
            w.writerow([content, cid, time_str, label])
            n_ok += 1
            if n_ok % 500000 == 0:
                print(f"  已写 {n_ok} 行 …")

    rows_id = sorted(content_to_id.items(), key=lambda x: x[1])
    with id_path.open("w", encoding="utf-8", newline="") as f:
        iw = csv.writer(f)
        iw.writerow(["content", "EventID"])
        for c, eid in rows_id:
            iw.writerow([c, eid])

    print(f"BGL_df: {n_ok} 行 -> {df_path}（跳过 {n_skip} 行）")
    print(f"BGL_df-ID: {len(rows_id)} 条模板 -> {id_path}")


def main() -> None:
    here = Path(__file__).resolve().parent
    default_raw = here.parent / "output" / "bgl" / "raw" / "BGL.log"
    default_out = here.parent / "output" / "bgl"
    p = argparse.ArgumentParser(description="raw BGL.log -> BGL_df.csv + BGL_df-ID.csv")
    p.add_argument("--raw", type=Path, default=default_raw, help="BGL.log 路径")
    p.add_argument("--out-dir", type=Path, default=default_out, help="输出目录")
    p.add_argument("--bgl-df", type=str, default="BGL_df.csv")
    p.add_argument("--bgl-df-id", type=str, default="BGL_df-ID.csv")
    args = p.parse_args()
    if not args.raw.is_file():
        raise FileNotFoundError(args.raw)
    raw_to_bgl_dfs(args.raw, args.out_dir, args.bgl_df, args.bgl_df_id)


if __name__ == "__main__":
    main()
