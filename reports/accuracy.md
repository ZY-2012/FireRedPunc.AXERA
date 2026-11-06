# FireRedPunc AX650 accuracy

Source model: FireRedTeam/FireRedPunc (Apache-2.0), chinese-lert-base BERT-base,
101.68M parameters, 5 labels `{space, ，, 。, ？, ！}`.

## Export parity (host, x86)

Static ONNX, seq_len=128, `[CLS] + tokens` (no `[SEP]`), output `[1,127,5]`.

| comparison | rows | tokens | token agree | text exact |
|---|---:|---:|---:|---:|
| ONNX fp32 vs PyTorch | 272 | 4523 | **100%** | **100%** |

- ONNX size: 406.1 MB (fp32)
- max abs logit diff: 2.53e-05
- Ops: Add, And, Cast, Div, Erf, Flatten, Gather, LayerNormalization, MatMul, Mul, Reshape, Slice, Softmax, Transpose, Where

## Board parity (AX650, U16)

Pulsar2 7.0 via Magnetar COMPILE; NPU1; U16 activations + S8 weights;
word embedding `Gather` kept FP32; smooth quant enabled.

| comparison | rows | tokens | token agree | text exact |
|---|---:|---:|---:|---:|
| AXMODEL vs fp32 ONNX | 272 | 4523 | **100%** | **100%** |
| C++ vs Python runtime | 272 | — | — | **100%** |

## C++ runtime parity (board)

C++ tokenizer + AXEngine output was compared with `python/punc_ax.py` on the
same 272 sentences: **272/272 text exact**, including English casing and
punctuation.

## Punctuation quality vs gold (board)

Evaluation corpus: Sokuji punctuation benchmark zh/en passages + Tatoeba
source sentences, 272 rows. Metrics are character-position P/R/F1 on a
lowercased punctuation-free skeleton. 62 rows are excluded from the gold
metric because tokenizer lowercasing/normalization made the skeleton differ
from the reference; they are still included in the token/text parity above.

| metric | P | R | F1 |
|---|---:|---:|---:|
| breakpoint, all marks | 0.9810 | 0.9960 | 0.9784 |
| breakpoint, excluding final position | 1.0000 | 0.9929 | **0.9937** |
| sentence-end, all marks | 0.9810 | 0.9960 | 0.9784 |
| sentence-end, excluding final position | 1.0000 | 0.9929 | **0.9937** |

The all-marks numbers include the model's final `。`/`.` where the reference
sometimes has no final mark; the VAD endpoint already supplies that boundary,
so the excluding-final number is the useful one for the pipeline.

## Notes

- Upstream `RuleBaedTxtFix` lowercases English (`iPhone` -> `iphone`). This is
  the upstream behavior; the translation pipeline should restore casing from
  the original text if en→zh quality requires it.
- The model expects unpunctuated input. `punc_ax.py` strips `，。？！、,.?!`
  first (keeping `,`/`.` between digits).
- Inputs over 127 tokens are truncated by the current runtime; the existing
  VAD path caps utterances at 10 s, which is well inside 127 tokens.
