#!/usr/bin/env python3
"""Compare the exported ONNX with the upstream PyTorch FireRedPunc.

Reports token-class agreement, exact text agreement, and punctuation F1
against the gold reference on the eval corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import transformers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get(
    "FIREREDASR2S_DIR", PROJECT_ROOT.parent / "FireRedASR2S" / "fireredasr2s"))
MODEL_DIR = Path(os.environ.get(
    "FIREREDPUNC_DIR", PROJECT_ROOT.parent / "FireRedPunc"))
WORK = PROJECT_ROOT / ".work"
DEFAULT_ONNX = WORK / "onnx" / "punc.fp32.sim.onnx"
DEFAULT_CORPUS = WORK / "corpus" / "eval.json"

sys.path.insert(0, str(UPSTREAM))

from fireredpunc.models.fireredpunc_bert import FireRedPuncBert  # noqa: E402
from fireredpunc.punc import ModelIO, RuleBaedTxtFix  # noqa: E402

PUNC_ALL = set("，。？！、,.?!;；:：")
SENT_END = set("。.？！?!")


def load_model():
    import argparse as _argparse
    torch.serialization.add_safe_globals([_argparse.Namespace])
    package = torch.load(MODEL_DIR / "model.pth.tar", map_location="cpu",
                         weights_only=True)
    args = package["args"]
    args.pretrained_bert = str(MODEL_DIR / "chinese-lert-base")
    args.bert = transformers.BertModel.from_pretrained(
        args.pretrained_bert, attn_implementation="eager")
    args.bert.pooler = None
    args.hidden_size = args.bert.config.hidden_size
    model = FireRedPuncBert(args)
    model.load_state_dict(package["model_state_dict"], strict=False)
    model.eval()
    return model, ModelIO(MODEL_DIR), int(args.cls_id)


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
    return {"p": round(p, 4), "r": round(r, 4), "f": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(WORK / "reports" / "parity_onnx.json"))
    args = ap.parse_args()

    model, model_io, cls_id = load_model()
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    rows = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[:args.limit]

    total_tok = agree_tok = 0
    exact_text = 0
    skel_mismatch = 0
    mark_stats = {"all": [], "excl_final": [], "sent_all": [], "sent_excl_final": []}
    diffs = []

    for i, row in enumerate(rows):
        text = row["input"]
        padded, lengths, tokens = model_io.text2tensor([text])
        n = int(lengths[0])
        if n > args.seq_len - 1:
            skel_mismatch += 1
            continue
        with torch.no_grad():
            logits_t = model.forward_model(padded, lengths)
        preds_t = logits_t.argmax(-1)[0].tolist()
        ref_text = RuleBaedTxtFix.fix(
            model_io.add_punc_to_txt([tokens[0]], [preds_t])[0])

        ids = np.zeros((1, args.seq_len), dtype=np.int32)
        mask = np.zeros((1, args.seq_len), dtype=np.float32)
        ids[0, 0] = cls_id
        ids[0, 1:1 + n] = padded[0, :n].numpy()
        mask[0, :1 + n] = 1.0
        logits_o = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        preds_o = logits_o[0, :n].argmax(-1).tolist()
        onnx_text = RuleBaedTxtFix.fix(
            model_io.add_punc_to_txt([tokens[0]], [preds_o])[0])

        total_tok += n
        agree_tok += sum(a == b for a, b in zip(preds_t, preds_o))
        if onnx_text == ref_text:
            exact_text += 1
        elif len(diffs) < 8:
            diffs.append({"id": row["id"], "torch": ref_text, "onnx": onnx_text})

        ref_skel, ref_all = skel_and_marks(row["ref"], PUNC_ALL)
        onnx_skel, onnx_all = skel_and_marks(onnx_text, PUNC_ALL)
        ref_sent, ref_sent_marks = skel_and_marks(row["ref"], SENT_END)
        onnx_sent, onnx_sent_marks = skel_and_marks(onnx_text, SENT_END)
        if ref_skel != onnx_skel or ref_sent != onnx_sent:
            skel_mismatch += 1
            continue
        last = max(0, len(ref_skel) - 1)
        mark_stats["all"].append(prf(onnx_all, ref_all))
        mark_stats["excl_final"].append(prf(
            {p for p in onnx_all if p < last}, {p for p in ref_all if p < last}))
        mark_stats["sent_all"].append(prf(onnx_sent_marks, ref_sent_marks))
        mark_stats["sent_excl_final"].append(prf(
            {p for p in onnx_sent_marks if p < last},
            {p for p in ref_sent_marks if p < last}))

    def agg(items):
        if not items:
            return {}
        out = {}
        for key in ("p", "r", "f"):
            out[key] = round(float(np.mean([x[key] for x in items])), 4)
        out["tp"] = int(sum(x["tp"] for x in items))
        out["fp"] = int(sum(x["fp"] for x in items))
        out["fn"] = int(sum(x["fn"] for x in items))
        return out

    report = {
        "onnx": args.onnx,
        "rows": len(rows),
        "tokens": total_tok,
        "token_agree_vs_torch": round(agree_tok / total_tok, 6) if total_tok else 0,
        "text_exact_vs_torch": round(exact_text / len(rows), 6) if rows else 0,
        "skel_mismatch_or_skipped": skel_mismatch,
        "punctuation": {k: agg(v) for k, v in mark_stats.items()},
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
