#!/usr/bin/env python3
"""Run Magnetar's COMPILE stage for the FireRedPunc task."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK = PROJECT_ROOT / ".work"
MAGNETAR_ROOT = Path(os.environ.get(
    "MAGNETAR_ROOT", PROJECT_ROOT.parents[2] / "Magnetar"))
if not (MAGNETAR_ROOT / "magnetar").is_dir():
    raise SystemExit(f"Magnetar not found: {MAGNETAR_ROOT}")

sys.path.insert(0, str(MAGNETAR_ROOT))

from magnetar.stages import compile as mcompile  # noqa: E402
import magnetar.docker_util as _du  # noqa: E402

_ORIG_DOCKER = _du.docker_pulsar2
_TIMEOUT = 7200


def _patched_docker(image, workspace, command, timeout=3600, **kwargs):
    return _ORIG_DOCKER(image, workspace, command, timeout=_TIMEOUT, **kwargs)


_du.docker_pulsar2 = _patched_docker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=str(WORK / "magnetar_tasks" / "punc"))
    ap.add_argument("--image", default=os.environ.get(
        "PULSAR2_IMAGE", "docker-registry.aitsw.axera-tech.com/pulsar2:7.0"))
    ap.add_argument("--precision", default="", choices=["", "U16", "U8"])
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    global _TIMEOUT
    _TIMEOUT = args.timeout
    task = Path(args.task)
    cfg = json.loads((task / "compile" / "custom_config.json").read_text(
        encoding="utf-8"))
    if args.precision:
        cfg["quant"]["layer_configs"][0]["data_type"] = args.precision
        (task / "compile" / "custom_config.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    print(f"===== COMPILE FireRedPunc ({cfg['quant']['layer_configs'][0]['data_type']}) =====",
          flush=True)
    mcompile.run(task, "AX650", args.image, input_dtype="FP32",
                 custom_config=cfg, skip_validation=True)
    src = task / "compile" / "model.axmodel"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    dst = PROJECT_ROOT / "models" / "punc.axmodel"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    summary = mcompile.summarize_compile_log(task)
    real_errors = [e for e in summary.get("errors", [])
                   if "it/s" not in e and "Analysing" not in e
                   and "Calibration Progress" not in e]
    summary["errors"] = real_errors
    summary["axmodel_bytes"] = src.stat().st_size
    summary["copied_to"] = str(dst)
    out = task / "compile" / "compile_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[punc] axmodel={src.stat().st_size / 1024 / 1024:.1f} MiB "
          f"macs={summary.get('macs')} errors={real_errors}", flush=True)
    print("copied:", dst)


if __name__ == "__main__":
    main()
