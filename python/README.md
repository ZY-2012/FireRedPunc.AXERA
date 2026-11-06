# Python 推理

```bash
pip3 install numpy transformers
pip3 install axengine-<version>-py3-none-any.whl
```

```python
from punc_ax import FireRedPuncAX

p = FireRedPuncAX("models")
print(p.punctuate("今天天气很好我们去公园散步吧"))
```

或：

```bash
python3 python/punc_ax.py --model-dir models \
  --text "今天天气很好我们去公园散步吧"
```
