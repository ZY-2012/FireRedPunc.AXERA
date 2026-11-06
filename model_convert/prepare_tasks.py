#!/usr/bin/env python3
"""Create a Magnetar task directory for FireRedPunc COMPILE.

Magnetar's generic config is single-input; the custom config below lists both
`input_ids` and `attention_mask` and their calibration archives.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK = PROJECT_ROOT / ".work"
DEFAULT_ONNX = WORK / "onnx" / "punc.fp32.sim.onnx"
DEFAULT_CALIB = WORK / "calib"
DEFAULT_TASK = WORK / "magnetar_tasks" / "punc"


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_meta(onnx_path: Path, seq_len: int) -> dict:
    return {
        "model_name": "fireredpunc",
        "framework": "pytorch",
        "inputs": [
            {"name": "input_ids", "shape": [1, seq_len], "dtype": "int32",
             "layout": "NCHW"},
            {"name": "attention_mask", "shape": [1, seq_len], "dtype": "float32",
             "layout": "NCHW"},
        ],
        "outputs": [
            {"name": "logits", "shape": [1, seq_len - 1, 5], "dtype": "float32"},
        ],
        "opset": 17,
        "source": "FireRedTeam/FireRedPunc (chinese-lert-base)",
    }


def build_config(seq_len: int, npu_mode: str = "NPU1",
                 precision: str = "U16") -> dict:
    return {
        "model_type": "ONNX",
        "npu_mode": npu_mode,
        "target_hardware": "AX650",
        "input": "/workspace/export/model.onnx",
        "output_name": "model.axmodel",
        "output_dir": "/workspace/compile",
        "work_dir": "/workspace/compile/work",
        "input_shapes": f"input_ids:1x{seq_len},attention_mask:1x{seq_len}",
        "onnx_opt": {
            "disable_onnx_optimization": False,
            "enable_onnxsim": True,
            "model_check": True,
        },
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "input_ids",
                    "calibration_dataset": "/workspace/export/calib_data/input_ids.tar.gz",
                    "calibration_size": -1,
                    "calibration_format": "Numpy",
                    "calibration_mean": [],
                    "calibration_std": [],
                },
                {
                    "tensor_name": "attention_mask",
                    "calibration_dataset": "/workspace/export/calib_data/attention_mask.tar.gz",
                    "calibration_size": -1,
                    "calibration_format": "Numpy",
                    "calibration_mean": [],
                    "calibration_std": [],
                },
            ],
            "layer_configs": [
                {"start_tensor_names": ["DEFAULT"],
                 "end_tensor_names": ["DEFAULT"],
                 "data_type": precision},
                {"op_type": "Gather", "data_type": "FP32"},
            ],
            "calibration_method": "MinMax",
            "enable_smooth_quant": True,
            "smooth_quant_threshold": 0.5,
            "conv_bias_data_type": "FP32",
            "disable_auto_refine_scale": True,
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
            "transformer_opt_level": 0,
            "highest_mix_precision": False,
        },
        "input_processors": [
            {"tensor_name": "input_ids", "tensor_layout": "NCHW",
             "src_dtype": "S32", "src_layout": "NCHW"},
            {"tensor_name": "attention_mask", "tensor_layout": "NCHW",
             "src_dtype": "FP32", "src_layout": "NCHW"},
        ],
        "compiler": {"check": 0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--calib", default=str(DEFAULT_CALIB))
    ap.add_argument("--task-root", default=str(DEFAULT_TASK))
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--npu-mode", default="NPU1", choices=["NPU1", "NPU2", "NPU3"])
    ap.add_argument("--precision", default="U16", choices=["U16", "U8"])
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    calib_dir = Path(args.calib)
    task = Path(args.task_root)
    for d in ("origin", "export", "export/calib_data", "compile", "simulate",
              "runonboard", "package", "cache"):
        (task / d).mkdir(parents=True, exist_ok=True)

    link_or_copy(onnx_path, task / "export" / "model.onnx")
    for name in ("input_ids.tar.gz", "attention_mask.tar.gz"):
        src = calib_dir / name
        if not src.is_file():
            raise SystemExit(f"missing calibration archive: {src}")
        link_or_copy(src, task / "export" / "calib_data" / name)

    meta = build_meta(onnx_path, args.seq_len)
    (task / "export" / "model_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = build_config(args.seq_len, args.npu_mode, args.precision)
    (task / "compile" / "custom_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (task / "config.json").write_text(json.dumps({
        "MODEL_NAME": "fireredpunc",
        "TARGET_HARDWARE": "AX650",
        "SOURCE": str(onnx_path),
        "TASK_DIR": str(task),
        "SDK_LANG": "python",
    }, indent=2) + "\n", encoding="utf-8")
    (task / "task.md").write_text(
        "# FireRedPunc AX650\n\n"
        f"- onnx: {onnx_path}\n"
        f"- seq_len: {args.seq_len}\n"
        f"- npu_mode: {args.npu_mode}\n"
        f"- precision: {args.precision}\n"
        "- calib: 351 zh/en/mixed unpunctuated rows\n",
        encoding="utf-8")
    print("task:", task)
    print("config:", task / "compile" / "custom_config.json")


if __name__ == "__main__":
    main()
