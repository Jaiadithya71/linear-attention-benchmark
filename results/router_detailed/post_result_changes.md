# Changes after version 1 measurements

Version 1 completed sensitivity and scaling, then failed at the first Qwen capture. Its runner wrongly unpacked `_expand_kv(q,k,v)` into three tensors. The existing helper returns only expanded key and value.

For the Qwen-only resume:
- Changed `q,k,v=_expand_kv(q,k,v)` to `k,v=_expand_kv(q,k,v)`.
- Added a regression test for the helper's return contract.
- Ran only the existing Qwen stage, into a separate output directory.
- Kept the same model, layers, corpus, context, calibration/evaluation windows, tolerance grid and router implementations.

No synthetic regime, tolerance, probe count, seed, or timing method was changed after observing results. The original source used for synthetic measurements is preserved under `source_v1/`, matching the version 1 manifest hashes. `finished_utc` remains null because that full run failed; `completed_parts` records the two completed synthetic stages.

The report script normalizes measured repeated-call total times by the actual number of calls when comparing per-call latency. Raw timing CSVs are unchanged. Synthetic charts and analysis are derived from the saved measured CSVs, not new benchmark runs. Original-router OOM rows remain in the raw CSVs and are excluded, not counted as successes, in speedup medians.

The Qwen-only resume completed successfully. Its manifest, raw results and visible log excerpt are in `../router_detailed_qwen/`. Model-level results are in the root SUMMARY.md.
