<p align="center"><img src="assets/logo.svg" width="360" alt="LiteMoE"></p>

<p align="center"><b>一个下午就能读完的 MoE 推理引擎。</b></p>

<p align="center">约 1600 行 Python，固定批次下吞吐比 vLLM 高 5～9%。</p>

```bash
git clone https://github.com/ZhuLinsen/LiteMoE && cd LiteMoE
pip install -e '.[vllm]'
litemoe "用一句话解释混合专家模型。"
```

| 同场 12 点几何平均（对比 vLLM 0.32.0） | Granite 1B | Granite 3B | 总计 |
|---|---:|---:|---:|
| NVIDIA A100 | 110.6% | 107.1% | **108.8%** |
| NVIDIA L4 | 105.5% | 105.0% | **105.3%** |

用 `bash tools/reproduce.sh` 可以在自己的 GPU 上复现。

**如何理解这个结果：** LiteMoE 调用 vLLM 的专家、Attention 和路由算子，自己负责 forward、KV 和生成循环。测试条件是固定批次、贪心解码、单 GPU。vLLM 还带调度器和完整服务，LiteMoE 没有。

**准确率：** 在 GSM8K 全部 1319 道测试题上（Granite 3B，A100），LiteMoE 为 46.2%，vLLM 为 47.2%，差值 -0.9 个百分点（95% 置信区间 -2.9 到 +1.1）。未检测到损失，但输出并不完全相同，而且这只是一种任务、一个模型的结果。详见[质量结果和已知限制](docs/benchmark.md)。

**目前支持：** Granite 3.1 1B（显存约 3.5 GB）和 3B（约 7.5 GB），单张支持 BF16 的 NVIDIA GPU，贪心解码，上下文最长 4096 token。已在 Ubuntu 24.04、Python 3.12、CUDA 13.0、A100 和 L4 上测试。

**暂不支持：** 其他模型、多卡、采样、服务化、动态批处理。

生产环境请用 vLLM。LiteMoE 适合学习 MoE 推理，或作为实验的小型基础。

[Apache-2.0](LICENSE)
