<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" width="360" alt="LiteMoE">
  </picture>
</p>

<p align="center"><b>A MoE inference engine you can read in an afternoon.</b></p>

<p align="center">About 1,600 lines of Python. 5-9% more throughput than vLLM on fixed batches.</p>

<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center"><a href="docs/tutorial.md"><b>Tutorial: how LLM inference works</b></a> · <a href="https://zhulinsen.github.io/LiteMoE/visualizer/?lang=en"><b>Interactive demo</b></a></p>

```bash
git clone https://github.com/ZhuLinsen/LiteMoE && cd LiteMoE
pip install -e '.[vllm]'
litemoe "Explain mixture-of-experts in one sentence."
```

## Benchmark

| 12-point geomean vs vLLM 0.32.0, same session | Granite 1B | Granite 3B | Overall |
|---|---:|---:|---:|
| NVIDIA A100 | 110.6% | 107.1% | **108.8%** |
| NVIDIA L4 | 105.5% | 105.0% | **105.3%** |

Reproduce it on your GPU with `bash tools/reproduce.sh`.

<a href="https://zhulinsen.github.io/LiteMoE/visualizer/?lang=en"><img src="docs/visualizer/demo.en.gif" width="800" alt="One recorded request, step by step: the token going in and out, the top 5 candidates, KV cache growth and expert routing"></a>

**What this does and doesn't mean:** LiteMoE calls vLLM's expert, attention and router kernels, and owns forward, KV and the generation loop. These runs use fixed batches, greedy decoding and one GPU. vLLM also runs a scheduler and a full serving stack, which LiteMoE doesn't have.

**Accuracy:** on all 1,319 GSM8K test problems (Granite 3B, A100), LiteMoE scored 46.2% and vLLM 47.2%, a difference of -0.9 points (95% CI -2.9 to +1.1). No loss was detected, but outputs aren't identical and this is one task on one model. [Quality results and known limits](benchmarks/README.md)

## Where the speed comes from

Most of the gain comes from removing work that can't affect live requests: per-layer masks, index lookups and output copies in Decode. Whole Decode steps run as a CUDA Graph.

Start reading at [`litemoe/model.py`](litemoe/model.py).

## Status

| | |
|---|---|
| **Works** | Granite 3.1 1B (3.5 GB GPU memory) and 3B (7.5 GB), one NVIDIA GPU with BF16, greedy decoding, up to 4,096 tokens of context |
| **Tested on** | Ubuntu 24.04, Python 3.12, CUDA 13.0, A100 and L4 |
| **Not yet** | Other models, multi-GPU, sampling, serving, dynamic batching |

Use vLLM for production. Use LiteMoE to learn how MoE inference works, or as a small base for experiments.

## License

[Apache-2.0](LICENSE). Not affiliated with vLLM, IBM or DeepSeek. [Third-party notices](THIRD_PARTY_NOTICES.md)
