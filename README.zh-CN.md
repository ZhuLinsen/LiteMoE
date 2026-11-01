<p align="center"><img src="assets/logo.svg" width="360" alt="LiteMoE"></p>

<p align="center"><b>一个下午就能读完的 MoE 推理引擎。</b></p>

<p align="center">约 1600 行 Python，固定批次下吞吐最高比 vLLM 多 16%。</p>

```bash
git clone https://github.com/ZhuLinsen/LiteMoE && cd LiteMoE
pip install -e '.[vllm]'
litemoe "用一句话解释混合专家模型。"
```

| 同场 12 点几何平均（对比 vLLM 0.32.0） | Granite 1B | Granite 3B | 总计 |
|---|---:|---:|---:|
| NVIDIA A100 | 109.8% | 106.6% | **108.2%** |
| NVIDIA L4 | 106.8% | 104.9% | **105.9%** |

用 `bash tools/reproduce.sh` 可以在自己的 GPU 上复现。

**如何理解这个结果：** LiteMoE 调用 vLLM 的专家、Attention 和路由算子，自己负责 forward、KV 和生成循环。测试条件是固定批次、贪心解码、单 GPU。vLLM 还带调度器和完整服务，LiteMoE 没有。详见[测试方法和已知限制](docs/benchmark.md)。

**目前支持：** Granite 3.1 1B 和 3B，单张 NVIDIA GPU，贪心解码。

**暂不支持：** 其他模型、多卡、采样、服务化、动态批处理。

生产环境请用 vLLM。LiteMoE 适合学习 MoE 推理，或作为实验的小型基础。

[Apache-2.0](LICENSE)
