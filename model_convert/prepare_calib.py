#!/usr/bin/env python3
"""Build Pulsar2 Numpy calibration archives for FireRedPunc."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get(
    "FIREREDASR2S_DIR", PROJECT_ROOT.parent / "FireRedASR2S" / "fireredasr2s"))
MODEL_DIR = Path(os.environ.get(
    "FIREREDPUNC_DIR", PROJECT_ROOT.parent / "FireRedPunc"))
WORK = PROJECT_ROOT / ".work"

sys.path.insert(0, str(UPSTREAM))

from fireredpunc.punc import ModelIO  # noqa: E402


def save_tar(out_dir: Path, name: str, arrays: list[np.ndarray]):
    tmp = out_dir / f"_{name}"
    tmp.mkdir(parents=True, exist_ok=True)
    for old in tmp.glob("*.npy"):
        old.unlink()
    for i, arr in enumerate(arrays):
        np.save(tmp / f"{i:05d}.npy", arr)
    tar_path = out_dir / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(tmp.glob("*.npy")):
            tar.add(f, arcname=f.name)
    return tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(WORK / "corpus" / "calib.json"))
    ap.add_argument("--out-dir", default=str(WORK / "calib"))
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[:args.limit]

    io = ModelIO(MODEL_DIR)
    cls_id = 101
    ids_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    lengths = []
    used = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        tokens, _ = io.tokenizer.tokenize(text, recover_unk=True)
        ids = [io.in_dict.get(t, io.in_dict.unk) for t in tokens]
        ids = ids[: args.seq_len - 1]
        if not ids:
            continue
        arr = np.zeros((1, args.seq_len), dtype=np.int32)
        arr[0, 0] = cls_id
        arr[0, 1:1 + len(ids)] = np.asarray(ids, dtype=np.int32)
        mask = np.zeros((1, args.seq_len), dtype=np.float32)
        mask[0, :1 + len(ids)] = 1.0
        ids_list.append(arr)
        mask_list.append(mask)
        lengths.append(len(ids))
        used.append(row["id"])

    p_ids = save_tar(out_dir, "input_ids", ids_list)
    p_mask = save_tar(out_dir, "attention_mask", mask_list)
    manifest = {
        "rows": len(ids_list),
        "seq_len": args.seq_len,
        "lengths": {"min": min(lengths) if lengths else 0,
                    "max": max(lengths) if lengths else 0,
                    "mean": round(float(np.mean(lengths)), 2) if lengths else 0},
        "ids": used,
        "files": {"input_ids": str(p_ids), "attention_mask": str(p_mask)},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "ids"},
                     ensure_ascii=False, indent=2))
    for p in (p_ids, p_mask):
        print(f"{p.name}: {p.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
