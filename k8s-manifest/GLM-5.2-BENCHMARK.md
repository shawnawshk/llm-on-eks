# GLM-5.2-FP8 on EKS p5en.48xlarge — Benchmark Report

**Date**: 2026-07-07
**Model**: `zai-org/GLM-5.2-FP8` — 753B MoE (DSA sparse attention, 256 experts/top-8, 1 MTP layer), FP8 weights ~756 GB
**Hardware**: p5en.48xlarge (8× H200 141 GB, 16× EFA), Karpenter `reserved-capacity-pool` (us-west-2d capacity block)
**Load generator**: NVIDIA genai-perf 0.0.16.post1 (Triton 26.06 SDK pod, in-cluster); `sglang.bench_serving` for the final PD run (genai-perf SSE parser fails against the router at concurrency 40)

## Workload

All runs use the same profile unless noted:

| Parameter | Value |
|---|---|
| Input length (ISL) | 8000 tokens, stddev 0 |
| Output length (OSL) | 1024 tokens, pinned via `max_tokens` + `ignore_eos` |
| Concurrency | 20 (and 40 for 2-node shapes) |
| Thinking mode | **off** — `{"chat_template_kwargs":{"enable_thinking":false}}` (nested form required; flat `enable_thinking` is ignored) |
| Endpoint | streaming `/v1/chat/completions`, in-cluster ClusterIP |

genai-perf reports ~40 requests counted per run — it measures a steady-state window inside `--num-prompts 100`; the numbers are valid.

## Executive Summary

Six configurations were benchmarked across four deployment shapes. Key numbers (concurrency as noted):

| # | Shape | Config | Conc. | TTFT p50 | TTFT p90/p99 | ITL avg | tok/s/user | **Total tok/s** | Stable |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 1-node TP8 SGLang | chunk 2048 (default), mem 0.85, MTP 5-1-6 | 20 | 2,040 ms | 14,656 / 17,160 ms | 27.9 ms | 40.1 | 476 | ✅ |
| R2 | 1-node TP8 SGLang | chunk 32K (16K eff.), mem 0.85 | 20 | — | — | — | — | — | ❌ OOM |
| R4 | 1-node TP8 SGLang | chunk 32K (effective), mem **0.80**, MTP 1-1-2 | 20 | 3,202 ms | 17,145 / 17,790 ms | 27.5 ms | 39.3 | 456 | ✅ |
| V1 | 1-node TP8 vLLM 0.24 | defaults, mem 0.85, MTP 5 drafts | 20 | 1,953 ms | 18,314 / 25,420 ms | 28.7 ms | 38.8 | 454 | ✅ |
| L2 | 2-node TP16 LWS+EFA | mem **0.80**, MTP 1-1-2 | 20 | 2,458 ms | 14,813 / 17,365 ms | 32.4 ms | 32.9 | 396 | ✅ |
| L3 | 2-node TP16 LWS+EFA | (same) | **40** | **1,230 ms** | **1,542 / 1,936 ms** | 41.1 ms | 26.2 | **730** | ✅ |
| P1 | 2-node PD 1P+1D NIXL | both TP8, mem 0.85 | 20 | 11,560 ms | 35,258 / 38,412 ms | **25.0 ms** | 40.5 | 356 | ✅ |
| P2 | 2-node PD 1P+1D NIXL | (same, bench_serving) | 40 | 17,246 ms (median) | — / 53,122 ms | 31.7 ms | ~31 | 712 | ✅ |

### Recommendation matrix (for this 8K-in / 1K-out, prefill-heavy workload)

| Priority | Recommended shape | Why |
|---|---|---|
| Max throughput per dollar | **2× independent TP8 replicas** (extrapolated ~912 tok/s) | No cross-node allreduce tax; each node at full efficiency. *Not yet measured — see Open Items.* |
| Hard TTFT SLO at high concurrency | **2-node TP16 (LWS + EFA)** | Only shape with TTFT p99 < 2.5 s at concurrency 40 |
| Low concurrency / single-node budget | 1-node TP8, SGLang or vLLM (they tie) | ~455 tok/s wall; fine for ≤10 concurrent 8K requests |
| Strict ITL SLO or decode-heavy workloads | PD disaggregation (needs ≥2P:1D here) | ITL p99 27.5 ms is untouchable, but 1P:1D inverts this workload's needs |

## Findings

### 1. The single-node wall is prefill compute, not scheduling

Chunked-prefill size was swept 2048 → 16384 → 32768 on TP8: total throughput stayed at 456–476 tok/s and TTFT tails did not move. vLLM on all-default settings landed at 454 tok/s — two engines, three schedulers, same number. At 20 concurrent 8K prompts the prefill arrival rate saturates 8× H200 compute in a co-located setup; scheduling parameters only redistribute the queue, they cannot shrink it. The SGLang cookbook's chunked-prefill gains (+34–78 %) did not reproduce on this workload/config.

### 2. `--mem-fraction-static` safe line is 0.80, single- and multi-node

0.85 OOM-crashed twice under load, with different triggers:

- **TP8 + 32K chunk** (R2): large-chunk prefill activations spiked ~3 GiB/GPU against ~2 GiB free → all 8 ranks OOM mid-benchmark.
- **TP16 + default 8K chunk** (L1, initial deploy): the static pool is a *percentage of total VRAM*, so TP16's per-GPU weight savings (~94 GiB → ~47 GiB) were silently absorbed by the KV pool, leaving only ~600 MiB activation headroom → OOM at concurrency 20.

Dropping to 0.80 frees ~7 GiB/GPU for activations; both shapes then survived every run with zero restarts. Multi-node does **not** automatically gain memory headroom — the fraction semantics guarantee it doesn't.

### 3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is incompatible with TP serving

Attempted as a fragmentation mitigation; crashed at startup — custom all-reduce cannot IPC-register VMM-backed allocations (`custom_all_reduce.cuh:37: CUDA error: invalid argument` during CUDA graph capture). Do not combine with multi-GPU TP.

### 4. `--chunked-prefill-size` is silently capped by `--max-prefill-tokens`

SGLang's default `max_prefill_tokens=16384` caps the effective chunk. Setting `--chunked-prefill-size=32768` alone gives you 16K chunks; both flags must be raised together. (Made no difference here — see Finding 1 — but matters on workloads that are genuinely chunk-bound.)

### 5. TP16 needs concurrency ≥ ~40 to justify itself

At concurrency 20, TP16 was *worse* than TP8 (396 vs 456 tok/s): 16 GPUs under-fed, and every decode step pays two cross-node EFA allreduces (ITL 27.5 → 32.4 ms). At concurrency 40 it transforms: prefill bursts are absorbed whole — TTFT p90 collapsed from 14.8 s to **1.5 s** (10×) and throughput hit 730 tok/s. The cost is per-user decode speed (26 vs 39 tok/s) — the allreduce tax again.

### 6. PD 1P:1D is the wrong ratio for an 8:1 ISL:OSL workload

PD delivered its core promise — decode purity (ITL p99 27.5 ms, max 108 ms; per-user floor 36 tok/s) — but at concurrency 40 the single prefill node became the whole bottleneck: TTFT median 17 s, p99 53 s, while the decode node idled (peak output 1013 tok/s vs 712 sustained shows the spare capacity). Input throughput matched TP16 (~5.6K tok/s), confirming equal total compute — PD just partitions it statically, and this workload needs most of it on prefill. PD suits decode-heavy traffic or extreme ITL SLOs, and would need ≥2P:1D (3 nodes) here.

### 7. Engine scheduling styles differ where the wall doesn't

At the same 455 tok/s wall, vLLM favors the median user (TTFT p50 1.9 s, TTST 640 ms) while SGLang bounds the tail better (TTFT max 17.8 s vs 25.7 s). Pick by SLO shape; throughput won't change.

## Deployment shapes tested

| Shape | Manifest | Image |
|---|---|---|
| 1-node TP8 SGLang | `sglang/glm-5.2-fp8-p5en.yaml` (R4 config committed) | `lmsysorg/sglang:v0.5.13.post1` |
| 1-node TP8 vLLM | deployed ad-hoc (`glm-5-2-vllm`), not committed | `vllm/vllm-openai:v0.24.0` |
| 2-node TP16 LWS | `lws/lws-glm-5.2-fp8-2node.yaml` (0.80 + MTP 1-1-2 committed) | ECR `sglang-efa-p5:v0.5.13.post1-nixl` (EFA 1.49 + aws-ofi-nccl + NIXL) |
| 2-node PD 1P+1D | deployed ad-hoc (`lws-glm-5-2-prefill`/`-decode` + `glm-5-2-router`), not committed | same ECR image |

GLM-5.2 requires SGLang ≥ v0.5.13.post1 (`glm_moe_dsa` arch) and transformers ≥ 5.x for its tokenizer — old Triton SDK images (≤24.12) cannot tokenize it; use 26.06+.

## Crash post-mortems

| Event | Trigger | Root cause | Fix |
|---|---|---|---|
| TP8 OOM under load | c20 benchmark, ~4 min in | 0.85 static pool + 32K-chunk activation spikes | mem 0.80 |
| TP8 startup crash | CUDA graph capture | `expandable_segments` vs custom all-reduce IPC | remove env var |
| TP16 OOM under load | c20 benchmark, ~4 min in | 0.85 static pool absorbs TP16 weight savings into KV | mem 0.80 |
| genai-perf c40 failures ×2 | router streaming at c40 | genai-perf 0.0.16 SSE parser (`splintered SSE`, `orjson` error); backends healthy | use `sglang.bench_serving` |

## Open items

1. **2× TP8 replicas at c40** — the throughput-per-dollar favorite is extrapolated (456 × 2 ≈ 912 tok/s), not measured. Requires freeing the PD nodes and scaling `glm-5-2` to `replicas: 2`.
2. PD at 2P:1D ratio (3 nodes) if a decode-pure + low-TTFT profile is ever required.
3. MTP acceptance-length telemetry was not collected; draft-token tuning was done on cookbook guidance, not measured accept rates.

## Reproduce

```bash
# client pod: Triton 26.06 SDK (genai-perf + transformers 5.x)
kubectl exec deploy/triton-26-06 -- bash -lc "
genai-perf profile -m zai-org/GLM-5.2-FP8 \
  --url <service>.default.svc.cluster.local:80 \
  --endpoint-type chat --streaming \
  --num-prompts 100 \
  --synthetic-input-tokens-mean 8000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 1024 --output-tokens-stddev 0 \
  --concurrency 20 \
  --tokenizer zai-org/GLM-5.2-FP8 \
  --extra-inputs max_tokens:1024 \
  --extra-inputs ignore_eos:true \
  --extra-inputs '{\"chat_template_kwargs\":{\"enable_thinking\":false}}'"

# alternative when genai-perf's SSE parser chokes (PD router, high concurrency):
python3 -m sglang.bench_serving --backend sglang-oai-chat \
  --base-url http://<service>:80 --model zai-org/GLM-5.2-FP8 \
  --dataset-name random --random-input-len 8000 --random-output-len 1024 \
  --random-range-ratio 1.0 --num-prompts 150 --max-concurrency 40 \
  --extra-request-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```
