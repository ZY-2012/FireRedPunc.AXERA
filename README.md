# FireRedPunc.AXERA

FireRedPunc 中英标点恢复模型在 AXERA 平台上的部署工程（当前 AX650N）。

- [x] Python 推理（axengine）
- [x] C++ 推理（AXEngine，纯 C++ tokenizer）
- [x] 模型转换（ONNX 导出 → 对分验证 → Pulsar2 U16 量化）

量化模型和预编译 C++ 可执行文件发布在 HuggingFace：
[AXERA-TECH/FireRedPunc.AXERA](https://huggingface.co/AXERA-TECH/FireRedPunc.AXERA)。
本仓库仅含代码。

## 模型

- 上游：[FireRedTeam/FireRedPunc](https://huggingface.co/FireRedTeam/FireRedPunc)
- 结构：`chinese-lert-base`（BERT-base，12 层，768 hidden，vocab 21,128）
- 任务：中英标点恢复，输出 5 类：`space / ， / 。 / ？ / ！`
- 输入：`[CLS] + tokens`，`input_ids [1,128] int32`，`attention_mask [1,128] float32`
- 输出：`logits [1,127,5]`
- 量化：Pulsar2 7.0，AX650，NPU1，U16 激活 + S8 权重，Gather FP32
- AXMODEL 大小：151.5 MiB

## 结果（AX650N）

| 项 | 结果 |
|---|---|
| ONNX fp32 vs PyTorch | 272 行 / 4523 token，token 100%，text 100% |
| AXMODEL vs fp32 ONNX | 272 行 / 4523 token，token 100%，text 100% |
| 标点 F1（排除句末） | 0.9937 |
| Python 延迟 | median 23.2–23.5 ms，p95 23.5–23.9 ms |
| C++ 延迟 | median 25.7 ms，p95 25.8 ms |

详细报告见 [`reports/`](reports/)。

## 目录结构

```
FireRedPunc.AXERA/
├── model_convert/        # ONNX 导出、校准、Pulsar2 编译
├── python/               # Python 板端推理
├── cpp/                  # C++ 板端推理（交叉编译 + AXEngine）
├── scripts/              # 板端评测
├── reports/              # 精度 / 性能报告
├── requirements.txt
└── .work/                # 临时产物（gitignore）
```

## 板端使用

模型和 C++ 二进制从 HuggingFace 下载：

```bash
hf download AXERA-TECH/FireRedPunc.AXERA models bin --local-dir .
```

### Python

```bash
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
python3 python/punc_ax.py --model-dir models \
  --text "今天天气很好我们去公园散步吧"
```

### C++

```bash
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
./bin/fireredpunc_ax650 --model-dir models \
  --text "今天天气很好我们去公园散步吧"
# 源码编译产物：./cpp/bin/fireredpunc_ax650
```

C++ 交叉编译：

```bash
export TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
export BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out
bash cpp/build_ax650.sh
```

## 转换复现

```bash
python3 model_convert/export_onnx.py --seq-len 128
python3 model_convert/build_corpus.py
python3 model_convert/prepare_calib.py
python3 model_convert/validate_onnx.py
python3 model_convert/prepare_tasks.py
MAGNETAR_ROOT=/path/to/Magnetar python3 model_convert/compile_magnetar.py
```

Pulsar2 镜像：`docker-registry.aitsw.axera-tech.com/pulsar2:7.0`。

## 参考

- [FireRedPunc](https://huggingface.co/FireRedTeam/FireRedPunc)
- [FireRedPunc.AXERA](https://huggingface.co/AXERA-TECH/FireRedPunc.AXERA)（模型 + C++ 二进制）
- [Magnetar](https://github.com/AXERA-TECH/Magnetar)

## License

模型与上游代码：Apache-2.0；本工程转换/运行时代码：Apache-2.0。
