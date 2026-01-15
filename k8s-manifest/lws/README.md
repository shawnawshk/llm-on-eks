# SGLang with EFA on AWS P5 Instances

This directory contains configurations for running SGLang with AWS Elastic Fabric Adapter (EFA) support on P5 instances using Kubernetes LeaderWorkerSet.

## Prerequisites

- EKS cluster with P5 instances (p5.48xlarge recommended)
- EFA device plugin installed on nodes
- LeaderWorkerSet controller installed
- ECR repository access

## Deployment Options

| Option | Files | Description |
|--------|-------|-------------|
| [Standard Deployment](#option-1-standard-deployment) | `Dockerfile`, `lws-p5.yaml` | Basic SGLang with EFA for multi-node tensor parallelism |
| [PD Disaggregation](#option-2-pd-disaggregation-deployment) | `Dockerfile.pd-disagg`, `lws-p5-pd-disagg.yaml` | Prefill-Decode separation with NIXL for higher throughput |

---

## Option 1: Standard Deployment

Demonstrates SGLang running on P5 instances with EFA-enabled multi-node tensor parallelism.

### Docker Image

The `Dockerfile` builds on the official SGLang image and adds:
- Libfabric v1.22.0 with EFA provider support
- AWS OFI NCCL plugin v1.11.0-aws for GPU-to-GPU communication over EFA

### Quick Start

1. Build the image:

```bash
docker build -t sglang-efa:v0.5.7 .
```

2. Push to your container registry and update the image in `lws-p5.yaml`

3. Deploy:

```bash
kubectl apply -f lws-p5.yaml
```

### Verify EFA

```bash
# Check EFA devices
kubectl exec -it <pod-name> -- ls /dev/infiniband/

# Verify NCCL uses EFA
kubectl exec -it <pod-name> -- bash -c "NCCL_DEBUG=INFO python -c 'import torch; torch.cuda.is_available()'"
```

### Configuration Reference

#### Key Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `FI_PROVIDER` | `efa` | Use EFA provider for libfabric |
| `FI_EFA_USE_DEVICE_RDMA` | `1` | Enable RDMA for low-latency communication |
| `NCCL_NET_PLUGIN` | `ofi` | Use OFI (libfabric) plugin for NCCL |
| `NCCL_P2P_LEVEL` | `NVL` | Enable NVLink for GPU-to-GPU communication |

#### Key SGLang Arguments

| Argument | Description |
|----------|-------------|
| `--tensor-parallel-size=16` | Distribute model across 16 GPUs (2 nodes × 8 GPUs) |
| `--disable-custom-all-reduce` | Use NCCL for collective operations |
| `--enable-nccl-nvls` | Enable NVLS (NVLink Switch) for faster communication |

#### Resource Requirements

| Resource | Value | Notes |
|----------|-------|-------|
| `vpc.amazonaws.com/efa` | 16 | EFA devices per pod (16 for p5.48xlarge) |
| `nvidia.com/gpu` | 8 | GPUs per pod |
| Shared memory | 500Gi | For inter-process communication |

---

## Option 2: PD Disaggregation Deployment

Prefill-Decode disaggregation separates compute-intensive prefill and memory-intensive decode phases for better throughput.

For detailed architecture and configuration, see [PD_DISAGGREGATION.md](PD_DISAGGREGATION.md).

### Quick Start

1. Build the Docker image:

```bash
docker build -f Dockerfile.pd-disagg -t sglang-efa-pd:v0.5.7 .
```

2. Push to your container registry (e.g., ECR):

```bash
docker tag sglang-efa-pd:v0.5.7 <account-id>.dkr.ecr.<region>.amazonaws.com/sglang-efa-pd:v0.5.7
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/sglang-efa-pd:v0.5.7
```

3. Update the image in `lws-p5-pd-disagg.yaml` to match your registry:

```yaml
image: <account-id>.dkr.ecr.<region>.amazonaws.com/sglang-efa-pd:v0.5.7
```

4. Deploy:

```bash
kubectl apply -f lws-p5-pd-disagg.yaml
```

---

## Troubleshooting

### EFA Devices Not Found

```bash
# Ensure EFA device plugin is running
kubectl get ds -n kube-system aws-efa-k8s-device-plugin
```

### NCCL Not Using EFA

```bash
# Check libfabric installation
kubectl exec -it <pod-name> -- fi_info -p efa
```

### NIXL Backend Errors (PD Disaggregation)

```bash
# Check libfabric EFA provider
kubectl exec -it <pod-name> -- fi_info -p efa

# Check NIXL logs
kubectl logs <pod-name> | grep -i "nixl\|backend"
```

## References

- [AWS EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [SGLang PD Disaggregation](https://docs.sglang.io/advanced_features/pd_disaggregation.html)
- [NIXL GitHub](https://github.com/ai-dynamo/nixl)
- [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws)
