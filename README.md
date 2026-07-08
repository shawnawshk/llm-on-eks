# LLM on EKS

Working examples for serving open-weight LLMs on Amazon EKS — from single-GPU
deployments to multi-node tensor parallelism over EFA and prefill/decode
disaggregation. Manifests are organized by serving stack; models covered include
GLM-5.2 (753B MoE), DeepSeek-R1/V3.2, the R1 distills, Qwen, gpt-oss, and Llama 4.

**Looking for a specific model?** See [docs/MODEL-INDEX.md](docs/MODEL-INDEX.md)
— the by-model view of every manifest, with instance types and maintenance status.

## Repository Layout

```
.
├── nodepool.yaml               # Karpenter NodePools (GPU + Neuron)
├── open-webui.yaml             # Optional chat UI (point OPENAI_API_BASE_URLS at any model Service)
├── docs/
│   ├── MODEL-INDEX.md          # By-model index of all manifests
│   ├── GLM-5.2-BENCHMARK.md    # GLM-5.2 benchmark: TP8 vs TP16 vs PD-disagg on p5en
│   ├── PD_DISAGGREGATION.md    # Prefill/decode disaggregation architecture deep-dive
│   └── benchmark-commands.md   # genai-perf / bench_serving command reference
└── k8s-manifest/
    ├── priority-class.yaml     # PriorityClass used by the serving pods
    ├── sglang/                 # Single-node SGLang deployments
    ├── vllm/                   # Single-node vLLM deployments
    ├── lws/                    # Multi-node: LeaderWorkerSet + EFA (TP across nodes, PD-disagg)
    │   └── Dockerfile.efa-*    # EFA-enabled SGLang image builds (suffix = sglang version)
    └── genai-perf/             # Load-test client pod (no GPU)
```

## Deployment Shapes

| Shape | When to use | Example |
|---|---|---|
| **Single-node** (`sglang/`, `vllm/`) | Model fits one node's GPUs; simplest ops, no cross-node traffic | `sglang/glm-5.2-fp8-p5en.yaml` — 753B FP8 MoE on 8× H200, TP8 |
| **Multi-node TP** (`lws/lws-*-tp16-*.yaml`) | Model too big for one node, or hard TTFT SLO at high concurrency | `lws/lws-glm-5.2-tp16-p5en.yaml` — TP16 across 2 nodes via EFA |
| **PD disaggregation** (`lws/lws-*-pd-*.yaml`) | Strict ITL SLO or decode-heavy traffic; prefill and decode scale independently | `lws/lws-glm-5.2-pd-p5en.yaml` — NIXL KV transfer over EFA RDMA + sglang-router |

Head-to-head numbers for all three shapes on the same hardware:
[docs/GLM-5.2-BENCHMARK.md](docs/GLM-5.2-BENCHMARK.md).

## Prerequisites

- EKS cluster with [Karpenter](https://karpenter.sh/); GPU capacity for the
  instance types named in each manifest (`nodeSelector`)
- `kubectl apply -f nodepool.yaml -f k8s-manifest/priority-class.yaml`
- Multi-node (`lws/`) additionally needs:
  - [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) controller
  - EFA device plugin (`aws-efa-k8s-device-plugin`) on the GPU nodes
  - An EFA-enabled image — build from `k8s-manifest/lws/Dockerfile.efa-*`
    and push to your registry (see [k8s-manifest/lws/README.md](k8s-manifest/lws/README.md))
- Models are pulled from HuggingFace on first start and cached on node-local
  NVMe (`hostPath`); large models (750 GB+) take a while on the first pod start

## Quick Start (single-node example)

```bash
# infra
kubectl apply -f nodepool.yaml -f k8s-manifest/priority-class.yaml

# deploy GLM-5.2-FP8 on 1x p5en.48xlarge (Karpenter provisions the node)
kubectl apply -f k8s-manifest/sglang/glm-5.2-fp8-p5en.yaml

# wait for weights download + engine start, then test
kubectl get pods -l app=glm-5-2 -w
kubectl run curl --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://glm-5-2.default.svc.cluster.local/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-5.2-FP8","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

Optional chat UI: edit `OPENAI_API_BASE_URLS` in `open-webui.yaml` to point at
your model's Service, then `kubectl apply -f open-webui.yaml`.

## Load Testing

Deploy the client pod and run genai-perf against any model Service in-cluster:

```bash
kubectl apply -f k8s-manifest/genai-perf/genai-perf-triton-2606.yaml
```

Full command reference (including thinking-model pitfalls and version-specific
flag changes): [docs/benchmark-commands.md](docs/benchmark-commands.md).

## Hard-Won Notes

Condensed from the benchmark writeups — details in [docs/](docs/):

- **`--mem-fraction-static` 0.80 is the safe line** for SGLang under heavy
  8K-input load on H200, single- and multi-node; 0.85 OOM-crashed both shapes.
- **Multi-node TP needs high concurrency to pay off** — at low concurrency the
  cross-node allreduce tax makes TP16 slower than TP8.
- **PD disaggregation ratio must match the traffic** — 1P:1D inverts on
  prefill-heavy (long-input) workloads; the prefill node bottlenecks while
  decode idles.
- **Reasoning models need thinking disabled for clean benchmarks** — nested
  `chat_template_kwargs.enable_thinking:false`, not a flat key.

## Conventions

- Manifest naming: `<model>-<engine>[-<instance>].yaml`; LWS:
  `lws-<model>-<topology>-<instance>.yaml`
- New manifest → add a row to [docs/MODEL-INDEX.md](docs/MODEL-INDEX.md)
  (a repo hook reminds about this)
- Old manifests are kept as reference (📦 in the index); they may pin stale
  images or lack probes — review before applying
