"""Real Qwen2.5-0.5B prefill with its attention swapped. Weights are the pretrained ones (no training / fine-tuning),
so linear/local variants are an *inference-time substitution*: speed and memory are real, and quality is measured
(perplexity on WikiText-2 test) rather than assumed. Transformer body only (no lm_head) for the latency/memory runs."""
import argparse, csv, math, os
import torch
from lab.env import require_cuda, utcnow, gather, write_config
from lab.timing import time_cuda, measure_memory, is_oom
from lab.qwen_patch import patched_attention

MB = 1024 ** 2
PF = ["variant", "N", "status", "median_ms", "q25_ms", "q75_ms", "min_ms", "reps", "warmup", "peak_extra_mb", "peak_total_mb", "finite_output"]
QF = ["variant", "ctx", "windows", "tokens_scored", "nll_per_token", "perplexity", "nonfinite_windows"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--require-gpu", default="T4")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--Ns", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--ppl-ctx", type=int, default=2048)
    p.add_argument("--ppl-windows", type=int, default=16)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()
    gpu = require_cuda(args.require_gpu)
    os.makedirs(args.out, exist_ok=True)
    started = utcnow()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    cfg = model.config
    print("GPU:", gpu, "| model", args.model, "layers", cfg.num_hidden_layers, "heads", cfg.num_attention_heads,
          "kv_heads", cfg.num_key_value_heads, flush=True)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    need = max(max(args.Ns), args.ppl_ctx * args.ppl_windows)
    if ids_all.numel() < need:
        raise RuntimeError(f"corpus has {ids_all.numel()} tokens, need {need}")
    variants = ["sdpa", "sdpa_mha", "local_window", "linear"]
    pf_rows, q_rows = [], []
    with torch.no_grad():
        for N in args.Ns:
            ids = ids_all[:N][None].cuda()
            for var in variants:
                row = dict(variant=var, N=N)
                with patched_attention(var, args.window):
                    call = lambda: model.model(input_ids=ids, use_cache=False).last_hidden_state
                    try:
                        out = call(); torch.cuda.synchronize()
                        row["finite_output"] = bool(torch.isfinite(out).all().item()); del out
                        t = time_cuda(call, warmup=args.warmup, reps=args.reps)
                        extra, peak = measure_memory(call)
                        row.update(status="ok", peak_extra_mb=extra / MB, peak_total_mb=peak / MB, **t)
                    except Exception as e:
                        if not is_oom(e):
                            raise
                        row["status"] = "oom"
                torch.cuda.empty_cache()
                pf_rows.append(row)
                print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in row.items() if k in ("variant", "N", "status", "median_ms", "peak_extra_mb", "finite_output")}, flush=True)
            with open(os.path.join(args.out, "qwen_prefill.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, PF); w.writeheader(); w.writerows(pf_rows)
        C = args.ppl_ctx
        for var in variants:
            nll, cnt, bad = 0.0, 0, 0
            with patched_attention(var, args.window):
                for i in range(args.ppl_windows):
                    ids = ids_all[i * C:(i + 1) * C][None].cuda()
                    logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
                    l = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
                    if not math.isfinite(l):
                        bad += 1; continue
                    nll += l; cnt += C - 1
            nll_t = nll / cnt if cnt else float("nan")
            q_rows.append(dict(variant=var, ctx=C, windows=args.ppl_windows, tokens_scored=cnt, nll_per_token=nll_t,
                               perplexity=math.exp(nll_t) if cnt and nll_t < 700 else float("nan"), nonfinite_windows=bad))
            print(q_rows[-1], flush=True)
    with open(os.path.join(args.out, "qwen_quality.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, QF); w.writeheader(); w.writerows(q_rows)
    write_config(os.path.join(args.out, "run_config_qwen.json"), gather("qwen", started, utcnow(), args))


if __name__ == "__main__":
    main()
