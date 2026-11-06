#!/usr/bin/env python3
"""Board-side evaluation: AXMODEL vs fp32 ONNX + gold punctuation.

Run on the AX650 board with the system Python 3.10 (has axengine,
onnxruntime, transformers).  No Pulsar2 simulation is used.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from punc_ax import FireRedPuncAX, RuleBaedTxtFix  # noqa: E402

PUNC_ALL = set("，。？！、,.?!．｡､")
SENT_END = set("。.？！?!")
CJK = set("，。？！、,.?!．｡､")


def skel_and_marks(text: str, marks_set: set[str]):
    skel: list[str] = []
    marks: set[int] = set()
    for ch in text:
        if ch in marks_set:
            if skel:
                marks.add(len(skel) - 1)
        else:
            skel.append(ch.lower())
    return "".join(skel), marks


def prf(pred: set[int], ref: set[int]):
    tp = len(pred & ref)
    fp = len(pred - ref)
    fn = len(ref - pred)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 1.0
    return {"p": p, "r": r, "f": f, "tp": tp, "fp": fp, "fn": fn}


def agg(items):
    if not items:
        return {}
    return {
        "p": round(float(np.mean([x["p"] for x in items])), 4),
        "r": round(float(np.mean([x["r"] for x in items])), 4),
        "f": round(float(np.mean([x["f"] for x in items])), 4),
        "tp": int(sum(x["tp"] for x in items)),
        "fp": int(sum(x["fp"] for x in items)),
        "fn": int(sum(x["fn"] for x in items)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models"))
    ap.add_argument("--corpus", default=str(ROOT / ".work" / "corpus" / "eval.json"))
    ap.add_argument("--onnx", default=str(ROOT / ".work" / "onnx" / "punc.fp32.sim.onnx"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / ".work" / "reports" / "board_report.json"))
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    p = FireRedPuncAX(args.model_dir)
    rows = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    onnx_sess = None
    if not args.no_onnx:
        import onnxruntime as ort
        onnx_sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    tok_total = tok_agree = 0
    text_exact = 0
    mism = 0
    stats = {"all": [], "excl_final": [], "sent_all": [], "sent_excl_final": []}
    diffs = []

    for row in rows:
        text = row["input"]
        tokens, ids, preds = p.classify(text)
        n = len(ids)
        if n == 0:
            continue
        ax_text = RuleBaedTxtFix.fix(p._add_punc(tokens, preds))

        if onnx_sess is not None:
            x = np.zeros((1, p.seq_len), dtype=np.int32)
            m = np.zeros((1, p.seq_len), dtype=np.float32)
            x[0, 0] = p.cls_id
            x[0, 1:1 + n] = np.asarray(ids, dtype=np.int32)
            m[0, :1 + n] = 1.0
            logits = onnx_sess.run(None, {"input_ids": x,
                                          "attention_mask": m})[0]
            preds_ref = logits[0, :n].argmax(-1).tolist()
            tok_total += n
            tok_agree += sum(a == b for a, b in zip(preds, preds_ref))
            ref_text = RuleBaedTxtFix.fix(p._add_punc(tokens, preds_ref))
            if ax_text == ref_text:
                text_exact += 1
            elif len(diffs) < 8:
                diffs.append({"id": row["id"], "ref": ref_text, "ax": ax_text})

        ref_skel, ref_all = skel_and_marks(row["ref"], PUNC_ALL)
        ax_skel, ax_all = skel_and_marks(ax_text, PUNC_ALL)
        ref_s, ref_sm = skel_and_marks(row["ref"], SENT_END)
        ax_s, ax_sm = skel_and_marks(ax_text, SENT_END)
        if ref_skel != ax_skel or ref_s != ax_s:
            mism += 1
            continue
        last = max(0, len(ref_skel) - 1)
        stats["all"].append(prf(ax_all, ref_all))
        stats["excl_final"].append(prf(
            {x for x in ax_all if x < last}, {x for x in ref_all if x < last}))
        stats["sent_all"].append(prf(ax_sm, ref_sm))
        stats["sent_excl_final"].append(prf(
            {x for x in ax_sm if x < last}, {x for x in ref_sm if x < last}))

    lat = p.latency("会议将在下午三点开始请大家准时参加", 50)
    report = {
        "rows": len(rows),
        "tokens": tok_total,
        "token_agree_vs_fp32": round(tok_agree / tok_total, 6) if tok_total else None,
        "text_exact_vs_fp32": round(text_exact / len(rows), 6) if onnx_sess else None,
        "skel_mismatch_or_skipped": mism,
        "punctuation": {k: agg(v) for k, v in stats.items()},
        "latency": lat,
        "axmodel_bytes": (Path(args.model_dir) / "punc.axmodel").stat().st_size,
        "diffs": diffs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "diffs"},
                     ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
