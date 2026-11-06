# 模型转换

FireRedPunc PyTorch → 静态 ONNX → Pulsar2 U16 → AXMODEL。

```bash
python3 export_onnx.py --seq-len 128
python3 build_corpus.py
python3 prepare_calib.py
python3 validate_onnx.py
python3 prepare_tasks.py
MAGNETAR_ROOT=/path/to/Magnetar python3 compile_magnetar.py
```

- 静态 shape：`input_ids [1,128] int32`，`attention_mask [1,128] float32`
- `[CLS] + tokens`，无 `[SEP]`；logits 第 i 行对应第 i 个 token
- 量化：U16 激活 + S8 权重，词嵌入 `Gather` 保持 FP32，MinMax + SmoothQuant
- Pulsar2 7.0，AX650，NPU1

详细配置见 `compile_magnetar.py` 和 `reports/`。
