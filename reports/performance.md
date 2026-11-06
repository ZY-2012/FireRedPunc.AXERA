# FireRedPunc AX650 performance

Board: AX650N, Ubuntu 22.04, axengine 0.1.3,
provider `AxEngineExecutionProvider`, chip type MC50, engine 2.12.0s,
Pulsar2 compiler 7.0.

## Model

| item | value |
|---|---|
| AXMODEL | `models/punc.axmodel` |
| size | 151.5 MiB (158,837,626 bytes) |
| NPU mode | NPU1 (single core) |
| precision | U16 activations + S8 weights, Gather FP32 |
| seq_len | 128 tokens |
| output | `[1,127,5]` logits |

## Latency

Measured in `scripts/board_eval.py` with 50 runs after warmup, text
"会议将在下午三点开始请大家准时参加":

| metric | ms |
|---|---:|
| median | 23.2 - 23.5 |
| p95 | 23.5 - 23.9 |
| min | 22.9 |
| max | 30.9 |

C++ runtime (including tokenizer and post-processing): median **25.7 ms**,
p95 **25.8 ms**; output 272/272 exact vs Python.

等效 RTF（按每个 10 s VAD 语音段折算）：Python **0.0023**，C++ **0.0026**。

The spread across repeated CLI runs is 23.2–23.5 ms median. The Punc stage is
negligible compared with MT (50–300 ms) and ZipVoice first-frame (~1.0–1.5 s).

## Board validation

```bash
ssh root@<board>
cd <project-root>
export LD_LIBRARY_PATH=/soc/lib:/opt/lib:${LD_LIBRARY_PATH:-}
python3 scripts/board_eval.py
```

Output: `.work/reports/board_report.json`.

No Pulsar2 simulation is used; the board runs the compiled AXMODEL directly.
The fp32 ONNX is used only as a CPU reference for token/text parity.
