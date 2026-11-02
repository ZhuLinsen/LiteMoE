# Benchmark

Run `bash tools/reproduce.sh`. It creates an environment, runs LiteMoE and vLLM 0.32.0 one after another on the same GPU allocation, and prints the ratios. Summaries are in [`published`](published).

## Setup

- Inputs: 12 frozen points in this directory, six each for Granite 3.1 1B and 3B.
- Profiles: batch 1, 8 and 32 with 128-token prompts; batch 1 with 2,048; batch 8 with 1,024; and a mixed batch of 32 to 2,048.
- Output: 128 tokens per request, greedy, ignoring EOS, BF16.
- Timing: 3 warmups and 5 measured repeats per point, median end-to-end throughput.

## Results

| Profile | A100 1B | A100 3B | L4 1B | L4 3B |
|---|---:|---:|---:|---:|
| b1-p128 | 119.4% | 110.2% | 103.8% | 101.7% |
| b8-p128 | 116.7% | 113.1% | 112.2% | 110.2% |
| b32-p128 | 111.1% | 109.4% | 107.6% | 111.7% |
| b1-p2048 | 109.7% | 105.4% | 101.2% | 99.6% |
| b8-p1024 | 107.0% | 105.2% | 108.2% | 107.1% |
| mixed-b8 | 100.7% | 99.6% | 100.8% | 100.2% |
| **Geomean** | **110.6%** | **107.1%** | **105.5%** | **105.0%** |

Measured at commit `78c9092`, source digest `cb2ec5310c46c101`, from a fresh clone and install. Benchmark tools were later trimmed without changing `litemoe/`. Per-repeat timings are in [`published/raw`](published/raw), with output tokens replaced by a hash. Each card was measured once on these numbers. Ratios moved by up to about 5 points between runs on the same card, and repeat runs put A100 at 108.2-108.8% and L4 at 105.2-105.9%.

Tested on Ubuntu 24.04, Python 3.12.3, NVIDIA driver 580.82.07, CUDA 13.0, torch 2.13.0. `litemoe` peaked near 3.5 GB of GPU memory on 1B and 7.5 GB on 3B, with a context limit of 4,096 tokens.

## Task-level quality

`tools/quality.py` runs GSM8K test problems, greedy, 256 tokens, batch 16. The bar was set before the first run: the lower 95% bound of the paired accuracy difference must stay above -3 points.

The decisive run used Granite 3B on all 1,319 problems on A100, chosen after the smaller runs below were inconclusive. **It passed:**

| 3B, 1,319 problems, A100 | Accuracy | LiteMoE minus this (95% CI) |
|---|---:|---:|
| LiteMoE | 46.2% | |
| vLLM | 47.2% | -0.9 (-2.9 to +1.1) |
| HF | 46.9% | -0.7 (-2.6 to +1.3) |

On the 819 problems not used in the earlier runs, the differences were -0.9 (-3.4 to +1.7) against vLLM and +0.1 (-2.4 to +2.6) against HF. No loss was detected, but a loss of up to about 3 points is not ruled out.

The earlier 500-problem runs were too small to clear the bar, missing it in 6 of 7 comparisons. No interval excluded zero, and LiteMoE was never ahead:

| 500 problems | LiteMoE | vLLM | HF | LiteMoE - vLLM (95% CI) | LiteMoE - HF (95% CI) |
|---|---:|---:|---:|---:|---:|
| A100 1B | 34.6% | 35.8% | 37.0% | -1.2 (-4.2 to +1.8) | -2.4 (-5.4 to +0.6) |
| A100 3B | 47.6% | 50.0% | 50.4% | -2.4 (-5.6 to +0.8) | -2.8 (-6.0 to +0.4) |
| L4 1B | 36.2% | 37.2% | 36.2% | -1.0 (-4.0 to +2.0) | 0.0 (-2.9 to +2.9) |
| L4 3B | 49.2% | 51.2% | timed out | -2.0 (-5.3 to +1.3) | not measured |

Granite 1B and L4 have not had a full-set run. Scores on the same problems also moved between runs on one card: the first 500 on A100 3B scored 47.6% and then 48.0% for LiteMoE, 50.0% and then 49.0% for vLLM. GSM8K measures one kind of task.

Across 500 problems per run there were no empty outputs, tokens after EOS or budget overruns, and repeating one batch gave identical LiteMoE output. Per-problem results are in [`quality-gsm8k.json`](published/quality-gsm8k.json).

## Known limits

- **Not a like-for-like product comparison.** vLLM's time includes a scheduler and serving machinery that LiteMoE doesn't have. These numbers don't predict behavior under dynamic traffic, which hasn't been measured.
- **Shared kernels.** Expert, attention, rotary and router kernels come from vLLM. LiteMoE's gain comes from the execution layer around them.
- **vLLM uses its default settings.** Its kernel tile configuration wasn't tuned. LiteMoE uses tuned tiles on A100 for Decode, worth 1-2%.
- **Numerics.** The fast path changes BF16 rounding, so some outputs differ from HF and vLLM. Against HF FP32 on 636 points, the error ratio to HF BF16 was 0.90 (1B) and 0.99 (3B), an after-the-fact summary. Worst-case logit error on 3B is higher than HF BF16 (3.35 vs 2.37 on L4). Pre-set stricter checks did not all pass, including 3B upper bounds and one 3B argmax mismatch in 108 points on L4.
- **Earlier evidence.** The numerics figures above and a WikiText NLL change of at most 0.0036 come from earlier cores and were not rerun on this commit.
