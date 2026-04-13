"""
BGL：由逐行 Content 与词表构造时间窗口序列（不依赖 LLM_template）。

输入（二者必填）：
- BGL_df.csv：Content, ContentID, Time, Label（ContentID 须与 BGL_df-ID 中 EventID 一致）。
- BGL_df-ID.csv：content, EventID

输出（与现有 logbert 数据名对齐）：
先在**整段日志**上滑窗得到全部序列，再划分：异常 → df_abnormal_processed.csv；正常 → 按全局时间切分点 t_cut（由 train_ratio 与首尾时间确定）将窗口起点早于 t_cut 的划入 train.csv，其余正常划入 test_normal_processed.csv。
对 train 再按 **contentIds** 去重得到 **df_train_uni_processed.csv**（保留首次出现顺序）。
由 **train.csv** 中实际出现过的模板 ID 生成 **train_vocab.csv**（contentId → vocabId，与 HDFS 流水线一致）。

列格式：ID, contentIds, Label（contentIds 为逗号+空格分隔的 template_id，与样例一致）。
"""

from __future__ import annotations

import argparse
import bisect
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _norm_header(h: str) -> str:
    return (h or "").strip().lower()


def _parse_time_to_unix(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%d-%H.%M.%S.%f")
        return int(dt.timestamp())
    except ValueError:
        pass
    try:
        dt = datetime.strptime(s, "%Y-%m-%d-%H.%M.%S")
        return int(dt.timestamp())
    except ValueError:
        return None


def load_vocab_from_id_csv(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        cols = {_norm_header(h): h for h in reader.fieldnames}
        c_key = cols.get("content") or cols.get("template")
        id_key = cols.get("eventid") or cols.get("contentid") or cols.get("template_id")
        if not c_key or not id_key:
            raise ValueError(f"{path} 需要列 content 与 EventID（或 ContentID / template_id）")
        for row in reader:
            c = (row.get(c_key) or "").strip()
            if not c:
                continue
            try:
                tid = int(str(row.get(id_key, "")).strip())
            except ValueError:
                continue
            out[c] = tid
    return out


def build_rows_from_bgl_df(
    bgl_df_path: Path,
    vocab: Dict[str, int],
) -> List[Tuple[int, int, int]]:
    rows_out: List[Tuple[int, int, int]] = []
    bad_time = 0
    line_no = 1

    with bgl_df_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"{bgl_df_path} 无表头")
        cols = {_norm_header(h): h for h in reader.fieldnames}
        c_col = cols.get("content")
        id_col = cols.get("contentid")
        t_col = cols.get("time")
        l_col = cols.get("label")
        if not all([c_col, id_col, t_col, l_col]):
            raise ValueError(
                f"{bgl_df_path} 需要列 Content, ContentID, Time, Label（当前: {list(reader.fieldnames)})"
            )

        for row in reader:
            line_no += 1
            content = (row.get(c_col) or "").strip()
            if not content:
                raise ValueError(f"{bgl_df_path} 第 {line_no} 行: Content 为空")
            if content not in vocab:
                raise ValueError(
                    f"{bgl_df_path} 第 {line_no} 行: Content 不在 BGL_df-ID 词表中（前 120 字符）: {content[:120]!r}"
                )
            tid = vocab[content]
            try:
                row_cid = int(str(row.get(id_col, "")).strip())
            except ValueError:
                raise ValueError(f"{bgl_df_path} 第 {line_no} 行: ContentID 非法")
            if row_cid != tid:
                raise ValueError(
                    f"{bgl_df_path} 第 {line_no} 行: ContentID={row_cid} 与词表 EventID={tid} 不一致"
                )

            ts = _parse_time_to_unix(row.get(t_col) or "")
            if ts is None:
                bad_time += 1
                continue
            try:
                label_ev = int(str(row.get(l_col, "0")).strip())
            except ValueError:
                label_ev = 0

            rows_out.append((ts, tid, label_ev))

    print(f"  有效行: {len(rows_out)} | 时间解析失败: {bad_time}")
    return rows_out


def sliding_windows(
    logs: List[Tuple[int, int, int]],
    window_seconds: int,
    step_seconds: int,
) -> List[Tuple[List[int], int, int]]:
    """(template_ids, window_label, window_start_unix)"""
    logs = sorted(logs, key=lambda x: x[0])
    if not logs:
        return []
    timestamps = [ts for ts, _, _ in logs]
    t_min, t_max = timestamps[0], timestamps[-1]
    out: List[Tuple[List[int], int, int]] = []
    t_start = t_min
    window_count = 0
    while t_start + window_seconds <= t_max:
        t_end = t_start + window_seconds
        start_idx = bisect.bisect_left(timestamps, t_start)
        end_idx = bisect.bisect_left(timestamps, t_end)
        if start_idx < end_idx:
            window_logs = logs[start_idx:end_idx]
            template_ids = [tid for _, tid, _ in window_logs]
            seq_label = 1 if any(lb == 1 for _, _, lb in window_logs) else 0
            out.append((template_ids, seq_label, int(t_start)))
        t_start += step_seconds
        window_count += 1
        if window_count % 1000 == 0:
            print(f"    窗口进度 {window_count}，当前序列 {len(out)}")
    return out


def _filter_short(
    seqs: List[Tuple[List[int], int, int]],
) -> List[Tuple[List[int], int, int]]:
    return [(ids, lab, ts) for ids, lab, ts in seqs if len(ids) > 1]


def dedup_rows_by_content_ids(rows: List[Tuple[List[int], int]]) -> List[Tuple[List[int], int]]:
    seen: Set[str] = set()
    out: List[Tuple[List[int], int]] = []
    for ids, lab in rows:
        key = ", ".join(map(str, ids))
        if key in seen:
            continue
        seen.add(key)
        out.append((ids, lab))
    return out


def write_logbert_csv(path: Path, rows: List[Tuple[List[int], int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "contentIds", "Label"])
        for i, (ids, lab) in enumerate(rows, start=1):
            w.writerow([i, ", ".join(map(str, ids)), lab])
    print(f"  -> {path} ({len(rows)} 行)")


def write_train_vocab_from_train_csv(train_csv: Path, out_path: Path) -> None:
    """收集 train.csv 中所有 contentIds 里的模板 ID，写出 contentId -> vocabId（与 HDFS hdfs_process.write_train_vocab 一致）。"""
    ids: Set[int] = set()
    with train_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{train_csv} 无表头")
        cols = {_norm_header(h): h for h in reader.fieldnames}
        c_col = cols.get("contentids")
        if not c_col:
            raise ValueError(f"{train_csv} 需含 contentIds 列")
        for row in reader:
            cell = str(row.get(c_col, "") or "")
            for part in cell.split(","):
                part = part.strip()
                if part.isdigit():
                    ids.add(int(part))
    if not ids:
        raise ValueError(f"{train_csv} 中未解析到任何模板 ID")
    sorted_ids = sorted(ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["contentId", "vocabId"])
        for vid, tid in enumerate(sorted_ids, start=1):
            w.writerow([tid, vid])
    print(f"  -> {out_path} ({len(sorted_ids)} 个 contentId)")


def main() -> None:
    parser = argparse.ArgumentParser(description="BGL 时间窗口序列（Content/EventID）")
    here = Path(__file__).resolve().parent
    default_data = here.parent / "output" / "bgl"
    parser.add_argument("--data-dir", type=Path, default=default_data, help="含 BGL_df.csv 的目录")
    parser.add_argument("--bgl-df", type=str, default="BGL_df.csv")
    parser.add_argument("--bgl-df-id", type=str, default="BGL_df-ID.csv")
    parser.add_argument("--window-minutes", type=int, default=5)
    parser.add_argument("--step-minutes", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="",
        help="输出子目录，空为 data-dir",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    bgl_df_path = data_dir / args.bgl_df
    id_path = data_dir / args.bgl_df_id
    out_dir = data_dir / args.out_subdir if args.out_subdir else data_dir

    if not bgl_df_path.is_file():
        raise FileNotFoundError(bgl_df_path)
    if not id_path.is_file():
        raise FileNotFoundError(f"缺少词表文件: {id_path}")

    print("加载词表:", id_path)
    vocab = load_vocab_from_id_csv(id_path)
    if not vocab:
        raise ValueError(f"{id_path} 词表为空或无法解析")
    print(f"  词表条目: {len(vocab)}")

    print("读取:", bgl_df_path)
    logs = build_rows_from_bgl_df(bgl_df_path, vocab)
    logs.sort(key=lambda x: x[0])
    if not logs:
        raise RuntimeError("无有效日志行")
    ts_all = [x[0] for x in logs]
    t_min, t_max = ts_all[0], ts_all[-1]
    span = t_max - t_min
    if span <= 0:
        raise RuntimeError("时间跨度无效")
    t_cut = t_min + args.train_ratio * span
    print(f"  全局时间切分 t_cut(Unix): {t_cut:.0f}（正常窗按窗口起点与此比较）")

    wsec = args.window_minutes * 60
    ssec = args.step_minutes * 60

    print("全量滑窗 …")
    all_seq = _filter_short(sliding_windows(logs, wsec, ssec))

    train_rows: List[Tuple[List[int], int]] = []
    test_normal: List[Tuple[List[int], int]] = []
    abnormal_all: List[Tuple[List[int], int]] = []

    for ids, lab, wstart in all_seq:
        if lab == 1:
            abnormal_all.append((ids, lab))
        else:
            if wstart < t_cut:
                train_rows.append((ids, lab))
            else:
                test_normal.append((ids, lab))

    print(
        f"  序列: 正常训练 {len(train_rows)} | 正常测试 {len(test_normal)} | 异常 {len(abnormal_all)}"
    )

    print(f"写出至 {out_dir} …")
    write_logbert_csv(out_dir / "train.csv", train_rows)
    train_uni = dedup_rows_by_content_ids(train_rows)
    print(f"  train 按 contentIds 去重: {len(train_rows)} -> {len(train_uni)}")
    write_logbert_csv(out_dir / "df_train_uni_processed.csv", train_uni)
    write_train_vocab_from_train_csv(out_dir / "train.csv", out_dir / "train_vocab.csv")
    write_logbert_csv(out_dir / "test_normal_processed.csv", test_normal)
    write_logbert_csv(out_dir / "df_abnormal_processed.csv", abnormal_all)
    print("完成。")


if __name__ == "__main__":
    main()
