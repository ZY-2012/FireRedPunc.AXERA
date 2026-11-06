#!/usr/bin/env python3
"""Build calibration/eval text for FireRedPunc.

Sources:
  - Sokuji punctuation benchmark corpus (zh/en passages)
  - Opus-MT Tatoeba zh-en / en-zh test files (source side)

The model is trained on unpunctuated ASR text, so calibration inputs have all
sentence/clause marks removed.  Eval items keep the reference text alongside
the stripped input.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from transformers import BertTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK = PROJECT_ROOT / ".work"
PUNC_ROOT = PROJECT_ROOT.parent
SOKUJI = Path(os.environ.get(
    "SOKUJI_PUNC_DIR",
    PUNC_ROOT / "sokuji" / "benchmark" / "punctuation-restoration" / "corpus"))
OPUS = Path(os.environ.get(
    "OPUS_MT_DATA_DIR",
    PUNC_ROOT.parents[1] / "opus-mt" / "Opus-MT.AXERA" / "model_convert" / "data"))
TOKENIZER_DIR = Path(os.environ.get(
    "FIREREDPUNC_TOKENIZER_DIR", PUNC_ROOT / "FireRedPunc" / "chinese-lert-base"))

STRIP_MARKS = set("，。？！、,.?!．｡､")
DIGIT_JOINERS = set(",.．")
SENT_SPLIT = re.compile(r"(?<=[。！？!?])")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def is_digit(ch: str) -> bool:
    return bool(ch) and ch.isdigit()


def strip_marks(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        joins = ch in DIGIT_JOINERS and is_digit(prev) and is_digit(nxt)
        out.append(" " if ch in STRIP_MARKS and not joins else ch)
    return "".join(out)


def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def sentence_chunks(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    return [clean_text(p) for p in SENT_SPLIT.split(text) if clean_text(p)]


def load_sokuji() -> list[dict]:
    rows = []
    for name in ("synthetic.gold.json", "gpt-live.gold.json"):
        data = json.loads((SOKUJI / name).read_text(encoding="utf-8"))
        for item in data.get("items", []):
            if item.get("lang") not in ("zh", "en"):
                continue
            ref = clean_text(item.get("ref", ""))
            if not ref:
                continue
            rows.append({
                "id": item.get("id", name),
                "lang": item["lang"],
                "ref": ref,
                "source": name,
            })
    return rows


def load_tatoeba(path: Path, direction: str) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for i in range(0, len(lines) - 3, 4):
        src = lines[i].strip()
        if not src or src.startswith(">>"):
            continue
        src = re.sub(r"^>>[^<]+<<\s*", "", src).strip()
        if direction == "en-zh":
            if not re.search(r"[A-Za-z]", src):
                continue
            if len(CJK.findall(src)) > 0.2 * max(1, len(src)):
                continue
            lang = "en"
        else:
            if len(CJK.findall(src)) < 0.25 * max(1, len(src)):
                continue
            lang = "zh"
        if any(ch in src for ch in "{}[]\\"):
            continue
        rows.append({"id": f"tatoeba-{direction}-{i//4}", "lang": lang,
                     "ref": clean_text(src), "source": "tatoeba"})
    return rows


def fit_tokens(text: str, tokenizer: BertTokenizer, limit: int = 127) -> list[str]:
    """Return text if it fits, otherwise skip it.

    Corpus passages are split into sentences before this call; a single
    sentence over 127 tokens is rare and is dropped rather than truncated
    mid-sentence (bad reference punctuation is worse than one fewer row).
    """
    text = clean_text(text)
    if not text:
        return []
    return [text] if len(tokenizer.tokenize(text)) <= limit else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(WORK / "corpus"))
    ap.add_argument("--max-tokens", type=int, default=127)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = BertTokenizer.from_pretrained(str(TOKENIZER_DIR))
    rows = load_sokuji()
    rows += load_tatoeba(OPUS / "zh-en.test.txt", "zh-en")
    rows += load_tatoeba(OPUS / "en-zh.test.txt", "en-zh")

    evals = []
    calib = []
    seen_eval = set()
    seen_calib = set()
    for row in rows:
        ref = row["ref"]
        add_eval = True
        # Full passage when it fits, otherwise its sentences.
        chunks = [ref] if len(tok.tokenize(ref)) <= args.max_tokens else sentence_chunks(ref)
        for j, chunk in enumerate(chunks):
            for k, part in enumerate(fit_tokens(chunk, tok, args.max_tokens)):
                ref_part = clean_text(part)
                inp = clean_text(strip_marks(ref_part))
                if not inp:
                    continue
                key = (row["lang"], inp)
                item = {
                    "id": f"{row['id']}#{j}.{k}",
                    "lang": row["lang"],
                    "ref": ref_part,
                    "input": inp,
                    "source": row["source"],
                }
                if add_eval and key not in seen_eval:
                    seen_eval.add(key)
                    evals.append(item)
                # Calibration: include the stripped text, with sentence-level
                # items preferred over duplicate full passages.
                if key not in seen_calib:
                    seen_calib.add(key)
                    calib.append({"id": item["id"], "lang": row["lang"],
                                  "text": inp, "source": row["source"]})
                # Also add shorter sentence-level calibration rows.
                if len(chunks) == 1:
                    for s in sentence_chunks(ref_part):
                        s_inp = clean_text(strip_marks(s))
                        if not s_inp:
                            continue
                        skey = (row["lang"], s_inp)
                        if skey in seen_calib:
                            continue
                        if len(tok.tokenize(s_inp)) > args.max_tokens:
                            continue
                        seen_calib.add(skey)
                        calib.append({"id": f"{item['id']}.s", "lang": row["lang"],
                                      "text": s_inp, "source": row["source"]})

    # Keep calibration bounded but representative: cap per language if huge.
    def cap(rows_, n):
        zh = [r for r in rows_ if r["lang"] == "zh"][:n]
        en = [r for r in rows_ if r["lang"] == "en"][:n]
        return zh + en

    calib = cap(calib, 400)
    evals = cap(evals, 300)
    for name, data in (("calib.json", calib), ("eval.json", evals)):
        path = out_dir / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        print(f"{name}: {len(data)} rows -> {path}")
    print("calib by source:",
          {k: sum(1 for r in calib if r['source'] == k) for k in
           sorted({r['source'] for r in calib})})
    print("eval by source:",
          {k: sum(1 for r in evals if r['source'] == k) for k in
           sorted({r['source'] for r in evals})})


if __name__ == "__main__":
    main()
