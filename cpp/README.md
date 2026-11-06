# FireRedPunc C++ 推理

AX650N 板端 C++ 标点恢复程序，纯 C++ 代码，不依赖 Python/transformers。

## 编译

```bash
export TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
export BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out
bash cpp/build_ax650.sh
```

产物：`cpp/bin/fireredpunc_ax650`。

## 运行

```bash
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
./cpp/bin/fireredpunc_ax650 --model-dir models \
  --text "今天天气很好我们去公园散步吧"
```

输出：

```
IN : 今天天气很好我们去公园散步吧
OUT: 今天天气很好，我们去公园散步吧。
```

支持 `--text` 重复传入、`--text-file`（一行一句）、`--force`、`--stats`。
