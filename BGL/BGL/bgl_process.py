"""
BGL：从 raw/BGL.log 串联到白化嵌入（BGL_df → 序列 CSV → SimCSE raw npy → whitening npy）。
供 logbert.py process 或本文件直接运行调用。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _bgl_scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _bgl_scripts_dir().parent.parent


def _run_step(name: str, extra: Optional[list[str]] = None) -> None:
    script = _bgl_scripts_dir() / name
    cmd = [sys.executable, str(script)]
    if extra:
        cmd.extend(extra)
    print("[bgl_process]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_repo_root()), check=True)


def run_bgl_process(
    raw_log: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    skip_raw: bool = False,
    augment: bool = False,
) -> None:
    """
    1) bgl_raw_to_df：raw → BGL_df.csv + BGL_df-ID.csv
    2) build_sequences：滑窗序列 + train_vocab 等
    3) embed_simcse_first_last：词表 → WhiteningNpy/bgl_simcse_first_last_raw.npy
    4) bgl_whiten_embeddings：白化 → WhiteningNpy/whitened_embeddings_256d.npy
    """
    ddir = data_dir if data_dir is not None else _bgl_scripts_dir().parent / "output" / "bgl"
    rlog = raw_log if raw_log is not None else ddir / "raw" / "BGL.log"

    if not skip_raw:
        if not rlog.is_file():
            raise FileNotFoundError(rlog)
        _run_step(
            "bgl_raw_to_df.py",
            ["--raw", str(rlog), "--out-dir", str(ddir)],
        )
    else:
        print("[bgl_process] 跳过 raw → BGL_df（--skip-raw）", flush=True)

    _run_step("build_sequences.py", ["--data-dir", str(ddir)])
    _run_step("embed_simcse_first_last.py")

    if augment:
        _run_step("bgl_whiten_embeddings.py", ["--augment"])
    else:
        _run_step("bgl_whiten_embeddings.py")

    print("[bgl_process] 完成。", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="BGL：raw → CSV → 序列 → SimCSE → 白化嵌入",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = _bgl_scripts_dir()
    default_data = here.parent / "output" / "bgl"
    default_raw = default_data / "raw" / "BGL.log"
    p.add_argument("--raw", type=Path, default=default_raw, help="BGL.log")
    p.add_argument("--data-dir", type=Path, default=default_data, help="含 BGL_df 等的目录")
    p.add_argument(
        "--skip-raw",
        action="store_true",
        help="已有 BGL_df.csv / BGL_df-ID.csv 时跳过第一步",
    )
    p.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="白化前用 LLM 增广协方差（需 OPENAI_API_KEY）",
    )
    args = p.parse_args()
    run_bgl_process(
        raw_log=args.raw,
        data_dir=args.data_dir,
        skip_raw=args.skip_raw,
        augment=args.augment,
    )


if __name__ == "__main__":
    main()
