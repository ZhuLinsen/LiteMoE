<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" width="360" alt="LiteMoE">
  </picture>
</p>

<p align="center"><b>A MoE inference engine you can read in an afternoon.</b></p>

<p align="center">About 1,600 lines of Python. Up to 16% more throughput than vLLM on fixed batches.</p>

```bash
git clone https://github.com/ZhuLinsen/LiteMoE && cd LiteMoE
pip install -e '.[vllm]'
litemoe "Explain mixture-of-experts in one sentence."
```

## Benchmark

| 12-point geomean vs vLLM 0.32.0, same session | Granite 1B | Granite 3B | Overall |
|---|---:|---:|---:|
| NVIDIA A100 | 109.8% | 106.6% | **108.2%** |
| NVIDIA L4 | 106.8% | 104.9% | **105.9%** |

Reproduce it on your GPU with `bash tools/reproduce.sh`.

**What this does and doesn't mean:** LiteMoE calls vLLM's expert, attention and router kernels, and owns forward, KV and the generation loop. These runs use fixed batches, greedy decoding and one GPU. vLLM also runs a scheduler and a full serving stack, which LiteMoE doesn't have. [Method and known limits](docs/benchmark.md)

## Where the speed comes from

Most of the gain comes from removing work that can't affect live requests: per-layer masks, index lookups and output copies in Decode. Whole Decode steps run as a CUDA Graph.

Start reading at [`litemoe/model.py`](litemoe/model.py).

## Status

| | |
|---|---|
| **Works** | Granite 3.1 1B and 3B, one NVIDIA GPU, greedy decoding |
| **Not yet** | Other models, multi-GPU, sampling, serving, dynamic batching |

Use vLLM for production. Use LiteMoE to learn how MoE inference works, or as a small base for experiments.

## License

[Apache-2.0](LICENSE). Not affiliated with vLLM, IBM or DeepSeek. [Third-party notices](THIRD_PARTY_NOTICES.md)

[中文说明](README.zh-CN.md)
