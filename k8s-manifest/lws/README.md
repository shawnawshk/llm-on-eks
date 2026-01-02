# SGLang with EFA on AWS P5 Instances

This directory contains configurations for running SGLang with AWS Elastic Fabric Adapter (EFA) support on P5 instances using Kubernetes LeaderWorkerSet.

## Overview

- **lws-p5.yaml**: Kubernetes LeaderWorkerSet manifest for deploying SGLang with EFA on P5 instances
- **Dockerfile**: Custom image based on SGLang with EFA libraries installed

## Prerequisites

- EKS cluster with P5 instances (p5.48xlarge recommended)
- EFA device plugin installed on nodes
- LeaderWorkerSet controller installed
- ECR repository access

## Custom Docker Image

The Dockerfile builds on top of the official SGLang image and adds:

- **Libfabric v1.22.0** with EFA provider support
- **AWS OFI NCCL plugin v1.11.0-aws** for GPU-to-GPU communication over EFA

### Building the Image

```bash
docker build -t sglang-efa:v0.5.7 .
```

### Pushing to ECR

```bash
# Tag for ECR
docker tag sglang-efa:v0.5.7 <account-id>.dkr.ecr.<region>.amazonaws.com/sglang-efa-p5:<tag>

# Authenticate
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# Push
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/sglang-efa-p5:<tag>
```

## Deployment

### Deploy LeaderWorkerSet

```bash
kubectl apply -f lws-p5.yaml
```

### Verify EFA Devices

Check that EFA devices are available in the pods:

```bash
kubectl exec -it <pod-name> -- ls /dev/infiniband/
```

You should see EFA devices like `uverbs0`, `uverbs1`, etc.

### Check NCCL with EFA

Verify NCCL can use EFA:

```bash
kubectl exec -it <pod-name> -- bash -c "NCCL_DEBUG=INFO python -c 'import torch; torch.cuda.is_available()'"
```

Look for messages indicating EFA/libfabric is being used.

## Configuration

### Key Settings in lws-p5.yaml

#### Security Context
- **privileged: true**: Required for EFA device access and RDMA operations
- **capabilities**: Added IPC_LOCK, SYS_RESOURCE, SYS_ADMIN for memory locking and resource management

#### Resource Limits
- **vpc.amazonaws.com/efa: 16**: Number of EFA devices per pod (16 for p5.48xlarge with 8 GPUs)
- **nvidia.com/gpu: 8**: Number of GPUs per pod

#### Volume Mounts
- **efa-devices**: Host path mount for `/dev/infiniband` to access EFA devices
- **shm**: Shared memory volume (500Gi) for inter-process communication
- **cache-volume**: Persistent storage for model weights

#### NCCL Configuration
- `NCCL_P2P_LEVEL=NVL`: Enable NVLink for GPU-to-GPU communication
- `NCCL_P2P_DISABLE=0`: Enable peer-to-peer communication
- `NCCL_SOCKET_IFNAME=eth0`: Network interface for NCCL

#### EFA Configuration
- `FI_PROVIDER=efa`: Use EFA provider for libfabric
- `FI_EFA_USE_DEVICE_RDMA=1`: Enable RDMA for low-latency communication
- `NCCL_NET_PLUGIN=ofi`: Use OFI (libfabric) plugin for NCCL
- `NCCL_TUNER_PLUGIN=ofi`: Use OFI tuner for optimal performance

#### SGLang Arguments
- `--tensor-parallel-size=16`: Distribute model across 16 GPUs (2 nodes × 8 GPUs)
- `--disable-custom-all-reduce`: Use NCCL for collective operations
- `--enable-nccl-nvls`: Enable NVLS (NVLink Switch) for faster communication
- `--dist-init-addr=$(LWS_LEADER_ADDRESS):20000`: Distributed training coordination
- `--nnodes=$(LWS_GROUP_SIZE)`: Number of nodes in the group
- `--node-rank=$(LWS_WORKER_INDEX)`: Rank of current node

### Environment Variables

```yaml
env:
  # EFA Configuration
  - name: FI_PROVIDER
    value: "efa"
  - name: FI_EFA_USE_DEVICE_RDMA
    value: "1"
  - name: NCCL_NET_PLUGIN
    value: "ofi"
  - name: NCCL_TUNER_PLUGIN
    value: "ofi"
  # NCCL Configuration
  - name: NCCL_P2P_LEVEL
    value: "NVL"
  - name: NCCL_P2P_DISABLE
    value: "0"
  - name: NCCL_SOCKET_IFNAME
    value: "eth0"
```

## Troubleshooting

### EFA Devices Not Found

Ensure the EFA device plugin is running:

```bash
kubectl get ds -n kube-system aws-efa-k8s-device-plugin
```

### NCCL Not Using EFA

Check libfabric installation:

```bash
kubectl exec -it <pod-name> -- fi_info -p efa
```

Should show EFA provider information.

### Performance Issues

- Verify all GPUs can see EFA devices
- Check NCCL_DEBUG logs for warnings
- Ensure proper network topology configuration

## References

- [AWS EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws)
