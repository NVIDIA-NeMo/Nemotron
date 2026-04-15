# Llama-3.1-Nemotron-Nano-8B-v1 on OCI OKE (Private Phoenix Deployment)

This cookbook documents a validated private deployment of
`nvidia/Llama-3.1-Nemotron-Nano-8B-v1` on **Oracle Cloud Infrastructure (OCI)** using:

- `us-phoenix-1`
- a **private** Oracle Kubernetes Engine (OKE) cluster
- a single `VM.GPU.A10.1` worker
- `vLLM` with an OpenAI-compatible `/v1` endpoint

This guide is intentionally **private-only**:

- no public Kubernetes API endpoint
- no public worker-node IPs
- no public inference endpoint

Access is handled through **OCI Bastion** and local port forwarding.

Note: the Terraform sample in this cookbook provisions the **OCI Bastion
service** for reproducible private access. It does **not** create a public
bastion host VM.

This gives Nemotron users a reproducible Oracle Cloud deployment path that
leans into OCI's strengths for enterprise workloads: private OKE control
planes, managed Bastion access, and a clean separation between infrastructure
provisioning and model serving.

## Why this configuration

This setup gives Nemotron users a reproducible OCI deployment path with a small
single-GPU footprint while preserving tool calling, structured output, and
streaming support.

For teams evaluating cloud options for Nemotron, this sample shows that OCI can
offer a practical and well-contained production shape: private networking,
managed access, and a validated GPU-backed serving path in Phoenix.

Validated capabilities on this deployment:

- chat completion
- structured output
- tool calling
- streaming
- async/concurrent requests
- OpenAI-compatible model discovery via `/v1/models`

## Tested environment

- Region: `us-phoenix-1`
- Kubernetes: OKE v1.31.10, enhanced cluster
- GPU shape: `VM.GPU.A10.1`
- CPU shape: `VM.Standard.E5.Flex`
- Model: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
- Serving stack: `vLLM v0.19.0`
- Helm chart: `vllm/vllm-stack` 0.1.10
- Inference API: OpenAI-compatible `/v1`

## Architecture

1. Create a **private** OKE cluster in Phoenix.
2. Create a CPU node pool (for router, CoreDNS) and a GPU node pool.
3. Use **OCI Bastion** to reach the cluster API locally.
4. Expand boot volume filesystems on all nodes.
5. Create StorageClasses and prerequisite PVCs.
6. Deploy Nemotron with the checked-in `vLLM` values file.
7. Patch CoreDNS for GPU tolerations.
8. Validate inference through a local port-forward.

## Prerequisites

- OCI tenancy with Phoenix capacity for `VM.GPU.A10.1`
- OKE and VCN permissions
- OCI Bastion permissions
- Boot volume update permissions (for filesystem expansion)
- `oci` CLI configured with a valid profile
- `kubectl`
- `helm`
- `ssh` (for Bastion tunneling)
- access to pull the Nemotron model from Hugging Face or an equivalent model
  artifact source accepted by your environment

## Step 1: Provision infrastructure

Terraform for the private Phoenix OKE infrastructure is available in
[`terraform/`](./terraform/).

```bash
cd terraform/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your tenancy, compartment, and SSH key path
terraform init
terraform plan
terraform apply
```

That Terraform path was validated end to end in Phoenix through:

- VCN and private subnets
- private OKE control plane
- OCI Bastion service
- CPU node pool (`VM.Standard.E5.Flex`, boot volume >= 100 GB)
- GPU node pool on `VM.GPU.A10.1` (boot volume 200 GB)

**Important:** Set the CPU node pool boot volume to at least **100 GB**. The
default OCI boot volume (47 GB) is too small for the vLLM router image
(~10.5 GB). The GPU node pool should use **200 GB**.

## Step 2: Connect to the private cluster

Download the kubeconfig using the private endpoint:

```bash
oci ce cluster create-kubeconfig \
  --cluster-id "$(terraform output -raw cluster_id)" \
  --file ~/.kube/config-nemotron \
  --region us-phoenix-1 \
  --token-version 2.0.0 \
  --kube-endpoint PRIVATE_ENDPOINT \
  --profile "$OCI_CLI_PROFILE"
```

Because the cluster is private, you must tunnel through OCI Bastion. Update the
kubeconfig to point to localhost and remove the CA certificate (since TLS
terminates at the tunnel):

```bash
# Remove the certificate-authority-data line and add:
kubectl config set-cluster <cluster-name> \
  --server=https://127.0.0.1:6443 \
  --insecure-skip-tls-verify=true \
  --kubeconfig=~/.kube/config-nemotron
```

If your OCI CLI profile is not `DEFAULT`, add it to the kubeconfig exec block:

```yaml
env:
  - name: OCI_CLI_PROFILE
    value: YOUR_PROFILE
```

Create a Bastion port-forwarding session:

```bash
export BASTION_ID="$(terraform output -raw oci_bastion_id)"
export PRIVATE_API_HOST="$(terraform output -raw apiserver_private_host)"

oci bastion session create-port-forwarding \
  --bastion-id "$BASTION_ID" \
  --ssh-public-key-file ~/.ssh/id_ed25519.pub \
  --key-type PUB \
  --target-port 6443 \
  --target-private-ip "$PRIVATE_API_HOST" \
  --display-name nemotron-oke-api \
  --session-ttl 10800 \
  --region us-phoenix-1 \
  --profile "$OCI_CLI_PROFILE"
```

Wait for the session to become ACTIVE, then start the SSH tunnel:

```bash
ssh -i ~/.ssh/id_ed25519 -N -L 6443:$PRIVATE_API_HOST:6443 \
  -p 22 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
  <session_ocid>@host.bastion.us-phoenix-1.oci.oraclecloud.com
```

Verify connectivity:

```bash
export KUBECONFIG=~/.kube/config-nemotron
kubectl get nodes
```

**Note:** Bastion sessions expire after the TTL (default 3 hours). If
`kubectl` stops responding, create a new session and restart the tunnel.

## Step 3: Expand boot volume filesystems

OCI boot volumes provision only ~47 GB of usable root filesystem regardless of
the requested volume size. Both the GPU and CPU nodes must be expanded before
deploying vLLM.

**Why this matters:** The vLLM engine image is ~10 GB, the router image is
~10.5 GB, and the model weights are ~16 GB. Without expansion, nodes run out
of ephemeral storage and pods get evicted.

For each node, create a privileged pod and expand the filesystem:

```bash
# Replace NODE_IP with the internal IP of each node
kubectl run expand-disk --restart=Never \
  --image=busybox:latest \
  --overrides="{
    \"spec\":{
      \"nodeName\":\"NODE_IP\",
      \"tolerations\":[{\"operator\":\"Exists\"}],
      \"containers\":[{
        \"name\":\"expand\",
        \"image\":\"busybox:latest\",
        \"command\":[\"sleep\",\"600\"],
        \"securityContext\":{\"privileged\":true},
        \"volumeMounts\":[{\"name\":\"host\",\"mountPath\":\"/host\"}]
      }],
      \"volumes\":[{\"name\":\"host\",\"hostPath\":{\"path\":\"/\"}}]
    }
  }"

kubectl wait --for=condition=Ready pod/expand-disk --timeout=60s
```

If the OCI boot volume was resized online (e.g., from 47 GB to 100 GB), the
OS must first rescan the block device:

```bash
kubectl exec expand-disk -- chroot /host bash -c \
  'echo 1 > /sys/class/block/sda/device/rescan'
```

Then expand the partition, LVM, and filesystem:

```bash
kubectl exec expand-disk -- chroot /host bash -c '
  growpart /dev/sda 3
  sleep 3
  pvresize /dev/sda3
  lvextend -l +100%FREE /dev/ocivolume/root
  xfs_growfs /
  df -h /
'
```

Restart kubelet so Kubernetes reports the updated storage capacity:

```bash
kubectl exec expand-disk -- nsenter -t 1 -m -p -- systemctl restart kubelet
kubectl delete pod expand-disk --force
```

Repeat for each node. Validated results:

- GPU node (200 GB boot volume): 36 GB → 189 GB usable
- CPU node (100 GB boot volume): 36 GB → 89 GB usable

## Step 4: Create StorageClasses

```bash
kubectl apply -f - <<'EOF'
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: oci-block-storage-enc
provisioner: blockvolume.csi.oraclecloud.com
parameters:
  vpusPerGB: "10"
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: oci-block-storage-hp
provisioner: blockvolume.csi.oraclecloud.com
parameters:
  vpusPerGB: "20"
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
EOF
```

## Step 5: Patch CoreDNS for GPU tolerations

If GPU nodes carry a `nvidia.com/gpu` taint, CoreDNS and the DNS autoscaler
must tolerate it or DNS resolution fails cluster-wide:

```bash
kubectl patch deployment coredns -n kube-system --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/tolerations/-",
        "value":{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}}]'

kubectl patch deployment kube-dns-autoscaler -n kube-system --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/tolerations/-",
        "value":{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}}]'
```

## Step 6: Create the templates PVC

The `vllm-stack` Helm chart (0.1.10) mounts a `vllm-templates-pvc` volume in
every engine pod for custom chat templates. This PVC must exist before
deploying:

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-templates-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: oci-block-storage-enc
  resources:
    requests:
      storage: 1Gi
EOF
```

Since this deployment uses the chat template bundled inside the vLLM image
(passed via `--chat-template` in extraArgs), this PVC only needs to exist — its
contents are not used.

## Step 7: Deploy vLLM

```bash
helm repo add vllm https://vllm-project.github.io/production-stack
helm repo update

helm upgrade --install vllm vllm/vllm-stack \
  -n default \
  -f vllm_oke_phoenix_private_values.yaml
```

**Do not** pass `--wait` to Helm. The engine pod takes several minutes to pull
the image (~10 GB) and download the model from Hugging Face.

Monitor progress:

```bash
kubectl get pods -n default -w
```

Wait for both pods to show `1/1 Running`:

- `vllm-deployment-router-*` — the request router (runs on the CPU node)
- `vllm-llama31-nemotron-nano-8b-deployment-vllm-*` — the model engine (runs
  on the GPU node)

### Key values explained

| Setting | Value | Why |
|---------|-------|-----|
| `tag` | `v0.19.0` | Pinned to the validated vLLM version |
| `maxModelLen` | `4096` | Conservative context to fit a single A10 (24 GB) |
| `gpuMemoryUtilization` | `0.95` | Maximize GPU memory for KV cache |
| `enableTool` | `true` | Enable tool/function calling |
| `toolCallParser` | `llama3_json` | Parser matching Nemotron's tool format |
| `extraArgs` | `--chat-template=...` | Pass template as a vLLM CLI arg (not via the chart's `chatTemplate` field, which prepends `/templates/`) |
| `storageClass` | `oci-block-storage-enc` | OCI Block Volume with balanced performance |

## Step 8: Validate

Port-forward the router service:

```bash
kubectl -n default port-forward svc/vllm-router-service 8080:80
```

Health check:

```bash
curl -s http://127.0.0.1:8080/health
# {"status":"healthy"}
```

Model discovery:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

Chat completion:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "messages": [{"role": "user", "content": "Reply with NEMOTRON_OK"}]
  }'
```

Tool-calling smoke test:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "messages": [{"role": "user", "content": "What time is it in UTC?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_utc_time",
        "description": "Return the current UTC time",
        "parameters": {
          "type": "object",
          "properties": {},
          "required": []
        }
      }
    }]
  }'
```

Expected behavior: the model returns a tool call with `finish_reason` set to
`tool_calls`.

## Operational notes

- Phoenix provided a workable path for this deployment when Chicago GPU capacity
  was not available.
- A single A10 is enough for the validated setup, but it requires conservative
  context sizing (`maxModelLen: 4096`).
- Private access plus local forwarding keeps the control plane and inference
  path off the public internet.
- Bastion sessions expire after the configured TTL (default 3 hours). Create a
  new session and restart the SSH tunnel when access drops.
- The NVIDIA device plugin is pre-installed on OKE enhanced clusters. No manual
  installation is required.

## Troubleshooting

### Boot volume filesystem is full / pods evicted for ephemeral storage

OCI boot volumes provision only ~47 GB of usable filesystem by default, even
when a larger volume is requested. Follow Step 3 to expand the filesystem. If
the boot volume itself is too small (e.g., 47 GB default), resize it first:

```bash
oci bv boot-volume update \
  --boot-volume-id <boot_volume_ocid> \
  --size-in-gbs 100 \
  --profile "$OCI_CLI_PROFILE" \
  --region us-phoenix-1 --force
```

After the volume reaches AVAILABLE, rescan the block device inside the node
before running `growpart`:

```bash
echo 1 > /sys/class/block/sda/device/rescan
```

### The engine pod stays Pending

Check for `persistentvolumeclaim "vllm-templates-pvc" not found` in the pod
events. The `vllm-stack` chart (0.1.10) requires this PVC to exist before the
engine pod can schedule. See Step 6.

### The engine pod crashes with a chat template error

The chart's `chatTemplate` field prepends `/templates/` to the path, which
breaks absolute paths like `/vllm-workspace/examples/...`. Pass the template
via `vllmConfig.extraArgs` instead:

```yaml
vllmConfig:
  extraArgs:
    - "--chat-template=/vllm-workspace/examples/tool_chat_template_llama3.1_json.jinja"
```

### The model pod starts but never becomes ready

Reduce context pressure and ensure the `vLLM` values include:

- `maxModelLen: 4096`
- `gpuMemoryUtilization: 0.95`

### Tool calling does not work

Make sure all of these are set:

- `enableTool: true`
- `toolCallParser: llama3_json`
- `--chat-template=/vllm-workspace/examples/tool_chat_template_llama3.1_json.jinja` in `vllmConfig.extraArgs`

### `kubectl` cannot reach the cluster

This guide assumes a **private** OKE cluster. Re-establish the Bastion tunnel
before using `kubectl`. Sessions expire after the configured TTL.

### Helm upgrade fails with field manager conflict

If you patched the router deployment manually (e.g., memory limits) and then
run `helm upgrade`, Helm may fail with a conflict. Uninstall and reinstall:

```bash
helm uninstall vllm -n default
helm install vllm vllm/vllm-stack -n default -f vllm_oke_phoenix_private_values.yaml
```

### The endpoint is reachable but `/v1/models` is empty or wrong

Confirm the deployment is serving `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` and
that the router service is forwarding to the Nemotron backend pods.
