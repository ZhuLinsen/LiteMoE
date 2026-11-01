# Benchmark

Run `bash tools/reproduce.sh`. It creates an environment, runs LiteMoE and vLLM 0.32.0 one after another on the same GPU allocation, and prints the ratios. Raw summaries are in [`benchmarks/published`](../benchmarks/published).

## Setup

- Inputs: 12 frozen points in `benchmarks/`, six each for Granite 3.1 1B and 3B.
- Profiles: batch 1, 8 and 32 with 128-token prompts; batch 1 with 2,048; batch 8 with 1,024; and a mixed batch of 32 to 2,048.
- Output: 128 tokens per request, greedy, ignoring EOS, BF16.
- Timing: 3 warmups and 5 measured repeats per point, median end-to-end throughput.

## Results

| Profile | A100 1B | A100 3B | L4 1B | L4 3B |
|---|---:|---:|---:|---:|
| b1-p128 | 116.4% | 110.0% | 104.7% | 102.2% |
| b8-p128 | 116.3% | 112.2% | 113.4% | 110.7% |
| b32-p128 | 110.1% | 108.7% | 109.2% | 111.2% |
| b1-p2048 | 109.5% | 104.5% | 102.8% | 99.0% |
| b8-p1024 | 107.0% | 105.1% | 109.9% | 106.6% |
| mixed-b8 | 100.5% | 99.4% | 101.7% | 100.4% |
| **Geomean** | **109.8%** | **106.6%** | **106.8%** | **104.9%** |

Each card was measured once on these numbers. Ratios moved by up to about 2 points between earlier runs on the same card.

## Known limits

- **Not a like-for-like product comparison.** vLLM's time includes a scheduler and serving machinery that LiteMoE doesn't have. These numbers don't predict behavior under dynamic traffic, which hasn't been measured.
- **Shared kernels.** Expert, attention, rotary and router kernels come from vLLM. LiteMoE's gain comes from the execution layer around them.
- **vLLM uses its default settings.** Its kernel tile configuration wasn't tuned. LiteMoE uses tuned tiles on A100 for Decode, worth 1-2%.
- **Numerics.** The fast path changes BF16 rounding, so some outputs differ from HF and vLLM. Against HF FP32 on 636 points, the error ratio to HF BF16 was 0.90 (1B) and 0.99 (3B), an after-the-fact summary. Worst-case logit error on 3B is higher than HF BF16 (3.35 vs 2.37 on L4). Pre-set stricter checks did not all pass, including 3B upper bounds and one 3B argmax mismatch in 108 points on L4.
- **Quality coverage.** WikiText NLL changed by at most 0.0036 on an earlier core, and 16 mixed prompts showed no clear difference. There is no task-level evaluation.
