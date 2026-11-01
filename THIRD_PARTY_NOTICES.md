# Third-party notices

LiteMoE is licensed under Apache-2.0. It depends on these separately installed packages, which keep their own licenses:

- [vLLM](https://github.com/vllm-project/vllm) 0.32.0, Apache-2.0: expert, rotary-embedding, router and FlashAttention operators, called through the installed package.
- PyTorch, Transformers, safetensors, huggingface-hub, NumPy and Triton.
- DeepSpeed, optional: an adapter in `litemoe/ops.py` calls its installed grouped GEMM operator.

Model weights are downloaded separately. The Granite 3.1 model pages list Apache-2.0.

LiteMoE is not affiliated with or endorsed by vLLM, IBM, DeepSeek or DeepSpeed.
