# linear-attention-benchmark

Measured comparison of softmax, PyTorch SDPA, local-window ("sparse") and kernelized linear attention
(feature map `relu(x)+1`), run on a Google Colab **T4**, plus a key-value retrieval test and a real
Qwen2.5-0.5B prefill with its attention swapped.

**Rule for this repo: every number is produced by code in this repo on the recorded run.**
Nothing is estimated, extrapolated or hardcoded. If something cannot be measured (for example an
out-of-memory case) it is reported as `oom` / blank, not filled in. The scripts refuse to run on CPU or on
a GPU that is not the one requested (`--require-gpu T4`).

## What is measured

| Script | What | How |
|---|---|---|
| `run_efficiency.py` | latency + peak memory, non-causal, B=1, H=8, D=64, N = 512 ... 65,536, fp32 and fp16 | `torch.cuda.Event` timing, 3 warmup, 20 timed reps (7 if a call takes >2 s; actual count is in the CSV), median and IQR; memory = `torch.cuda.max_memory_allocated` during one call minus what was allocated before it (inputs); output finiteness checked; max abs error vs fp32 SDPA recorded for N <= 4096 |
| `run_retrieval.py` | needle-in-distractors key-value retrieval, N = 64 ... 65,536 | 300 trials per setting, each with its own seed derived from `(seed, N, trial)`; success = attention output is closest (cosine) to the needle's value among all N values; 95% Wilson and bootstrap CIs; per-trial 0/1 results are in the CSV |
| `run_qwen.py` | real `Qwen/Qwen2.5-0.5B` (pretrained weights, fp16), transformer body prefill at 1k ... 32k tokens of WikiText-2 | attention function swapped through transformers' attention interface (transformers default SDPA path "sdpa", the same SDPA kernel on explicitly head-expanded K/V "sdpa_mha", causal local-window / causal chunked linear); latency + peak memory; WikiText-2 test perplexity at 2048 context for each variant |
| `make_figures.py`, `make_report.py` | plots and tables | read only the CSVs; the results block below is generated |

Methods: `softmax_naive` (materialises N x N scores), `sdpa` (PyTorch picks the kernel), `local_window`
(blocked: a query sees its own block and the neighbouring blocks, block size 256, vectorised, not a Python loop),
`linear` (computed in the input dtype, so fp16 can overflow and that is reported), `linear_acc32` (fp16 inputs,
sums accumulated in fp32; fp16 only because in fp32 it is identical to `linear`).

Correctness of the implementations is tested on CPU against reference formulas in `tests/` (these tests check math, they
are not benchmarks). `local_window` and `linear` are different functions from softmax attention, so their outputs are not
expected to match it; the error column quantifies the difference.

## Adaptive-rank linear attention (`lab/adaptive.py`, `run_adaptive.py`)
"Rank" is the number of positive random features m (FAVOR+-style approximation of the softmax kernel) with m in {64, 256, 1024}.
The router works per input (per batch element and head): it computes exact softmax attention for 16 probe queries, tries the ranks
in ascending order, and accepts the first rank whose output on the probes has mean relative L2 error <= tol (fixed at 0.1 before
any run; a tol sweep is also reported). Inputs that fail every rank are either answered with the largest rank anyway and flagged
`tol_not_met` (`adaptive_rank`), or with exact attention (`adaptive_rank_fb`). Probe cost and discarded lower-rank attempts count
in the measured latency and memory. It is compared with fixed-rank `rf64/rf256/rf1024` and exact SDPA on efficiency (two input
regimes: uniform scale and per-head mixed scales), the retrieval task, and real Qwen2.5-0.5B prefill + perplexity.
Random-feature attention is computed in fp32 internally for both input dtypes. In retrieval the probe queries are
`beta * random keys` (same distribution as the real query, never the query itself).

## Not included
* No causal efficiency sweep; causal attention appears only in the Qwen step.
* Qwen: the linear and local-window variants are inference-time swaps on weights trained for softmax, with no fine-tuning. The perplexity table shows what that costs. This says nothing about linear attention models trained from scratch.

## Reproduce
Open `notebooks/run_on_colab.ipynb` in Colab with a T4 runtime and run all cells
(<https://colab.research.google.com/github/Jaiadithya71/linear-attention-benchmark/blob/main/notebooks/run_on_colab.ipynb>).
The run configuration (GPU name, torch/CUDA/transformers versions, git commit, start/end UTC) is saved next to the CSVs in
`results/run_config_*.json`, and raw logs in `results/log_*.txt`.

<!-- RESULTS:START -->
_Results not generated yet._
<!-- RESULTS:END -->
