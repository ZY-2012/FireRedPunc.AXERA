#!/usr/bin/env python3
"""Export FireRedPunc to a static-shape ONNX for AX650.

Layout follows upstream FireRedPuncBert:
  input_ids      [1, seq_len] int32   ([CLS] + tokens, no [SEP])
  attention_mask [1, seq_len] float32
  logits         [1, seq_len-1, 5]    (the [CLS] row is dropped)

The graph is exported with HuggingFace BERT eager attention so Pulsar2 sees
explicit attention ops instead of an SDPA node.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import transformers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get(
    "FIREREDASR2S_DIR", PROJECT_ROOT.parent / "FireRedASR2S" / "fireredasr2s"))
MODEL_DIR = Path(os.environ.get(
    "FIREREDPUNC_DIR", PROJECT_ROOT.parent / "FireRedPunc"))
WORK = PROJECT_ROOT / ".work"
DEFAULT_OUT = WORK / "onnx" / "punc.fp32.onnx"
DEFAULT_META = WORK / "onnx" / "punc.fp32.json"

sys.path.insert(0, str(UPSTREAM))

from fireredpunc.models.fireredpunc_bert import FireRedPuncBert  # noqa: E402
from fireredpunc.punc import ModelIO, RuleBaedTxtFix  # noqa: E402


class Exportable(nn.Module):
    """FireRedPuncBert._forward for inputs that already start with [CLS]."""

    def __init__(self, model: FireRedPuncBert):
        super().__init__()
        self.bert = model.bert
        self.classifier = model.classifier

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)[0][:, 1:]
        return self.classifier(hidden)


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
    missing, unexpected = model.load_state_dict(package["model_state_dict"],
                                                strict=False)
    model.eval()
    return model, args, missing, unexpected


def make_inputs(seq_len: int):
    # [CLS] + a valid Chinese token, the rest [PAD].
    ids = np.zeros((1, seq_len), dtype=np.int32)
    ids[0, 0] = 101
    ids[0, 1] = 872  # 你
    mask = np.zeros((1, seq_len), dtype=np.float32)
    mask[0, :2] = 1.0
    return torch.from_numpy(ids), torch.from_numpy(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_path = Path(args.out)
    meta_path = Path(args.meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    model, model_args, missing, unexpected = load_model()
    print("state_dict missing:", missing, "unexpected:", unexpected)
    print(f"cls_id={model_args.cls_id} odim={model_args.odim} "
          f"seq_len={args.seq_len}")

    wrap = Exportable(model).eval()
    ids, mask = make_inputs(args.seq_len)
    with torch.no_grad():
        ref = wrap(ids, mask).numpy()

    torch.onnx.export(
        wrap, (ids, mask), str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    graph = onnx.load(str(out_path))
    onnx.checker.check_model(graph)
    onnxsim_path = out_path.with_suffix(".sim.onnx")
    try:
        from onnxsim import simplify
        graph_sim, ok = simplify(graph)
        if ok:
            onnx.save(graph_sim, str(onnxsim_path))
            out_path = onnxsim_path
            onnx.checker.check_model(onnx.load(str(out_path)))
    except Exception as exc:
        print(f"onnxsim skipped: {exc}")

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input_ids": ids.numpy(),
                          "attention_mask": mask.numpy()})[0]
    max_abs = float(np.abs(got - ref).max())
    agree = float((got.argmax(-1) == ref.argmax(-1)).mean())
    print(f"onnx={out_path} size={out_path.stat().st_size/1e6:.1f} MB "
          f"max_abs={max_abs:.3e} argmax_agree={agree:.6f}")

    ops = sorted({node.op_type for node in onnx.load(str(out_path)).graph.node})
    meta = {
        "model_name": "fireredpunc",
        "framework": "pytorch",
        "source": str(MODEL_DIR),
        "onnx": str(out_path),
        "seq_len": args.seq_len,
        "cls_id": int(model_args.cls_id),
        "odim": int(model_args.odim),
        "opset": args.opset,
        "onnx_size_bytes": out_path.stat().st_size,
        "torch_vs_onnx_max_abs": max_abs,
        "torch_vs_onnx_argmax_agree": agree,
        "ops": ops,
        "inputs": [
            {"name": "input_ids", "shape": [1, args.seq_len], "dtype": "int32"},
            {"name": "attention_mask", "shape": [1, args.seq_len], "dtype": "float32"},
        ],
        "outputs": [
            {"name": "logits", "shape": [1, args.seq_len - 1, 5], "dtype": "float32"},
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print("wrote", meta_path)
    print("ops:", ",".join(ops))


if __name__ == "__main__":
    main()
