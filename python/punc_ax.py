#!/usr/bin/env python3
"""FireRedPunc AX650 runtime.

Standalone (no torch): transformers slow BertTokenizer + AXEngine.

  p = FireRedPuncAX("models")
  p.punctuate("今天天气很好我们去公园散步吧")
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

try:
    from transformers import BertTokenizer
except Exception as exc:  # pragma: no cover
    raise SystemExit("transformers is required (BertTokenizer): %s" % exc)

PUNC_ALL = set("，。？！、,.?!．｡､")
DIGIT_JOINERS = set(",.．")
SENT_PUNCT = re.compile(r"[。！？!?]")
ALNUM = re.compile(r"[a-zA-Z0-9#]+")


class TokenDict:
    def __init__(self, path: str, unk: str = ""):
        self.id2word, self.word2id = self.read_dict(path)
        self.unk = unk
        if unk:
            assert unk in self.word2id, unk
        self.unkid = self.word2id[unk] if unk else -1

    def get(self, key, default):
        if isinstance(default, str):
            default = self.word2id[default]
        return self.word2id.get(key, default)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.word2id.get(key, self.word2id[self.unk]) if self.unk else self.word2id[key]
        if isinstance(key, int):
            return self.id2word[key]
        raise TypeError(key)

    def read_dict(self, path: str):
        id2word, word2id = [], {}
        with open(path, encoding="utf8") as f:
            for i, line in enumerate(f):
                toks = line.strip().split()
                if len(toks) >= 2:
                    word, index = toks[0], int(toks[1])
                elif len(toks) == 1:
                    word, index = toks[0], i
                else:
                    word, index = " ", i
                if word == "<space>":
                    word = " "
                assert len(id2word) == index
                word2id[word] = index
                id2word.append(word)
        return id2word, word2id


class HfBertTokenizer:
    """Upstream FireRedPunc tokenizer, including the _recover_unk pass."""

    def __init__(self, tokenizer_dir: str):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

    def tokenize(self, text: str, recover_unk: bool = False):
        tokens = self.tokenizer.tokenize(text)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        if recover_unk:
            try:
                tokens = self._recover_unk(text.lower(), tokens)
            except Exception:
                traceback.print_exc()
        return tokens, tokens_id

    def _recover_unk(self, text: str, tokens: list[str]):
        if "[UNK]" not in tokens:
            return tokens
        new_tokens = []
        text_no_space = text.replace(" ", "")
        if re.match(r"^[^a-zA-Z0-9']+$", text):
            tmp_text = text_no_space
            if len(tmp_text) == len(tokens):
                success = True
                for t, tok in zip(tmp_text, tokens):
                    if tok != "[UNK]" and t != tok:
                        success = False
                        break
                    new_tokens.append(t)
                if success:
                    return new_tokens
        new_tokens = []
        text_pos = 0
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "[UNK]":
                unk_count = 0
                j = i
                while j < len(tokens) and tokens[j] == "[UNK]":
                    unk_count += 1
                    j += 1
                post_token = ""
                if j < len(tokens):
                    post_token = tokens[j].replace("##", "")
                if post_token:
                    remaining = text_no_space[text_pos:]
                    anchor_pos = remaining.find(post_token)
                    unk_chars = remaining[:anchor_pos] if anchor_pos != -1 else remaining[:unk_count]
                else:
                    unk_chars = text_no_space[text_pos:text_pos + unk_count]
                for k in range(unk_count):
                    new_tokens.append(unk_chars[k] if k < len(unk_chars) else "")
                text_pos += len(unk_chars)
                i = j
            else:
                new_tokens.append(token)
                text_pos += len(token.replace("##", ""))
                i += 1
        return [t for t in new_tokens if t and t != "[UNK]"]


class RuleBaedTxtFix:
    @classmethod
    def fix(cls, txt_ori: str, capitalize_first: bool = True):
        txt = txt_ori.lower()
        txt = re.sub(r"([a-z])，([a-z])", r"\1, \2", txt)
        txt = re.sub(r"([a-z])。([a-z])", r"\1. \2", txt)
        txt = re.sub(r"([a-z])？([a-z])", r"\1? \2", txt)
        txt = re.sub(r"([a-z])！([a-z])", r"\1! \2", txt)
        txt = re.sub(r"^([a-z]+)，", r"\1,", txt)
        txt = re.sub(r"^([a-z]+)。", r"\1.", txt)
        txt = re.sub(r"^([a-z]+)？", r"\1?", txt)
        txt = re.sub(r"^([a-z]+)！", r"\1!", txt)
        txt = re.sub(r"( [a-zA-Z']+)，$", r"\1,", txt)
        txt = re.sub(r"( [a-zA-Z']+)。$", r"\1.", txt)
        txt = re.sub(r"( [a-zA-Z']+)？$", r"\1?", txt)
        txt = re.sub(r"( [a-zA-Z']+)！$", r"\1!", txt)
        txt = re.sub("^i ", "I ", txt)
        txt = re.sub("^i'm ", "I'm ", txt)
        txt = re.sub("^i'd ", "I'd ", txt)
        txt = re.sub("^i've ", "I've ", txt)
        txt = re.sub("^i'll ", "I'll ", txt)
        txt = re.sub(" i ", " I ", txt)
        txt = re.sub(" i'm ", " I'm ", txt)
        txt = re.sub(" i'd ", " I'd ", txt)
        txt = re.sub(" i've ", " I've ", txt)
        txt = re.sub(" i'll ", " I'll ", txt)
        if capitalize_first and txt and re.match("[a-z]", txt[0]):
            txt = txt[0].upper() + txt[1:]
        txt = re.sub(r"([.!?。？！])\s+([a-z])",
                     lambda m: f"{m.group(1)} {m.group(2).upper()}", txt)
        return txt


def strip_marks(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        joins = ch in DIGIT_JOINERS and prev.isdigit() and nxt.isdigit()
        out.append(" " if ch in PUNC_ALL and not joins else ch)
    return "".join(out)


class FireRedPuncAX:
    def __init__(self, model_dir: str | Path, seq_len: int = 128):
        self.dir = Path(model_dir)
        self.seq_len = int(seq_len)
        self.max_tokens = self.seq_len - 1
        self.cls_id = 101
        self.tokenizer = HfBertTokenizer(str(self.dir / "tokenizer"))
        self.in_dict = TokenDict(str(self.dir / "vocab.txt"), unk="[UNK]")
        self.out_dict = TokenDict(str(self.dir / "out_dict"))
        from axengine import InferenceSession
        # Force the on-board NPU provider.  When AXCL libs are on
        # LD_LIBRARY_PATH, axengine lists AXCLRT first and tries to open a
        # host PCIe device that does not exist on the board.
        self.sess = InferenceSession(
            str(self.dir / "punc.axmodel"),
            providers=["AxEngineExecutionProvider"])
        self._warmup()

    def _warmup(self):
        try:
            self.classify("你好")
        except Exception:
            traceback.print_exc()

    def _encode(self, text: str):
        tokens, _ = self.tokenizer.tokenize(text, recover_unk=True)
        ids = [self.in_dict.get(t, self.in_dict.unk) for t in tokens]
        return tokens, ids

    def classify(self, text: str):
        text = strip_marks(text or "")
        tokens, ids = self._encode(text)
        ids = ids[: self.max_tokens]
        n = len(ids)
        if n == 0:
            return [], [], []
        x = np.zeros((1, self.seq_len), dtype=np.int32)
        x[0, 0] = self.cls_id
        x[0, 1:1 + n] = np.asarray(ids, dtype=np.int32)
        mask = np.zeros((1, self.seq_len), dtype=np.float32)
        mask[0, :1 + n] = 1.0
        logits = self.sess.run(None, {"input_ids": x, "attention_mask": mask})[0]
        preds = logits[0, :n].argmax(-1).tolist()
        return tokens[:n], ids, preds

    def _add_punc(self, tokens: list[str], preds: list[int]) -> str:
        txt = ""
        for i, token in enumerate(tokens):
            tag = self.out_dict[preds[i]]
            if token.startswith("##"):
                token = token.replace("##", "")
            elif ALNUM.search(token) and i > 0 and ALNUM.search(tokens[i - 1]):
                if self.out_dict[preds[i - 1]] == " ":
                    token = " " + token
            txt += token if tag == " " else token + tag
        return txt.replace("  ", " ")

    def punctuate(self, text: str, force: bool = False) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if not force and SENT_PUNCT.search(text):
            return text
        tokens, _, preds = self.classify(text)
        if not tokens:
            return text
        return RuleBaedTxtFix.fix(self._add_punc(tokens, preds))

    def latency(self, text: str, n: int = 30) -> dict:
        self.punctuate(text, force=True)
        vals = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.punctuate(text, force=True)
            vals.append((time.perf_counter() - t0) * 1000)
        vals.sort()
        return {"median_ms": round(vals[len(vals) // 2], 2),
                "p95_ms": round(vals[int(len(vals) * 0.95) - 1], 2),
                "min_ms": round(vals[0], 2), "max_ms": round(vals[-1], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--text-file", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    p = FireRedPuncAX(args.model_dir)
    texts = list(args.text or [])
    if args.text_file:
        texts += [ln.strip() for ln in Path(args.text_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not texts:
        texts = ["今天天气很好我们去公园散步吧",
                 "会议将在下午三点开始请大家准时参加",
                 "hello how are you i am fine thanks"]
    for t in texts:
        out = p.punctuate(t, force=args.force)
        print(f"IN : {t}")
        print(f"OUT: {out}")
        if args.stats:
            print("    ", p.latency(t))


if __name__ == "__main__":
    main()
