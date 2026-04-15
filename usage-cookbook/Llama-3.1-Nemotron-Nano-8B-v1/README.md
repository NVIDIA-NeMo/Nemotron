# Llama-3.1-Nemotron-Nano-8B-v1 on OCI OKE (Private Deployment)

This cookbook documents a validated private deployment of
`nvidia/Llama-3.1-Nemotron-Nano-8B-v1` on **Oracle Cloud Infrastructure (OCI)**
using a private OKE cluster, a single `VM.GPU.A10.1` worker, and `vLLM` with an
OpenAI-compatible `/v1` endpoint.

Based on the [Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/index.html)
guide, customized for the Nemotron model with tool calling support.

## Tested environment

- Region: `us-phoenix-1`
- Kubernetes: OKE v1.31.10, enhanced cluster
- GPU shape: `VM.GPU.A10.1` (NVIDIA A10, 24 GB)
- CPU shape: `VM.Standard.E5.Flex`
- Model: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
- Serving stack: `vLLM v0.19.0`
- Helm chart: `vllm/vllm-stack` 0.1.10
- Inference API: OpenAI-compatible `/v1`

## Validated capabilities

- Chat completion
- Tool / function calling
- Streaming
- Async / concurrent requests
- OpenAI-compatible model discovery via `/v1/models`

## Prerequisites

- OCI tenancy with GPU capacity (`VM.GPU.A10.1`)
- `oci` CLI configured with a valid profile
- `kubectl`, `helm`, `ssh`, `jq`
- An SSH key pair (e.g., `~/.ssh/id_ed25519`)

**Note:** The NVIDIA device plugin is pre-installed on OKE enhanced clusters.
No manual installation is required.

## Step 1: Set environment variables

```bash
export OCI_COMPARTMENT_ID="<your-compartment-ocid>"
export OCI_REGION="us-phoenix-1"
export OCI_PROFILE="API_KEY_AUTH"
export CLUSTER_NAME="nemotron-phx"
export KUBERNETES_VERSION="v1.31.10"
```

## Step 2: Create VCN and networking

```bash
VCN_ID=$(oci network vcn create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --display-name "${CLUSTER_NAME}-vcn" \
    --cidr-blocks '["10.0.0.0/16"]' \
    --dns-label "nemotron" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

IGW_ID=$(oci network internet-gateway create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-igw" \
    --is-enabled true \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

NAT_ID=$(oci network nat-gateway create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-nat" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

SGW_SERVICE_ID=$(oci network service list \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data[?contains(name, 'All') && contains(name, 'Services')].id | [0]" \
    --raw-output)

SGW_SERVICE_NAME=$(oci network service list \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data[?contains(name, 'All') && contains(name, 'Services')].\"cidr-block\" | [0]" \
    --raw-output)

SGW_ID=$(oci network service-gateway create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-sgw" \
    --services "[{\"serviceId\": \"${SGW_SERVICE_ID}\"}]" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

PRIVATE_RT_ID=$(oci network route-table create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-private-rt" \
    --route-rules "[
        {\"cidrBlock\": \"0.0.0.0/0\", \"networkEntityId\": \"${NAT_ID}\"},
        {\"destination\": \"${SGW_SERVICE_NAME}\", \"destinationType\": \"SERVICE_CIDR_BLOCK\", \"networkEntityId\": \"${SGW_ID}\"}
    ]" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

PUBLIC_RT_ID=$(oci network route-table create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-public-rt" \
    --route-rules "[{\"cidrBlock\": \"0.0.0.0/0\", \"networkEntityId\": \"${IGW_ID}\"}]" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

SL_ID=$(oci network security-list create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-sl" \
    --egress-security-rules '[{"destination": "0.0.0.0/0", "protocol": "all", "isStateless": false}]' \
    --ingress-security-rules '[
        {"source": "0.0.0.0/0", "protocol": "6", "isStateless": false, "tcpOptions": {"destinationPortRange": {"min": 22, "max": 22}}},
        {"source": "10.0.0.0/16", "protocol": "all", "isStateless": false},
        {"source": "10.244.0.0/16", "protocol": "all", "isStateless": false},
        {"source": "10.96.0.0/16", "protocol": "all", "isStateless": false},
        {"source": "0.0.0.0/0", "protocol": "1", "isStateless": false, "icmpOptions": {"type": 3, "code": 4}}
    ]' \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)
```

Create four subnets:

```bash
API_SUBNET_ID=$(oci network subnet create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-api-subnet" \
    --cidr-block "10.0.0.0/28" \
    --route-table-id "${PRIVATE_RT_ID}" \
    --security-list-ids "[\"${SL_ID}\"]" \
    --dns-label "kubeapi" \
    --prohibit-public-ip-on-vnic true \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

WORKER_SUBNET_ID=$(oci network subnet create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-worker-subnet" \
    --cidr-block "10.0.10.0/24" \
    --route-table-id "${PRIVATE_RT_ID}" \
    --security-list-ids "[\"${SL_ID}\"]" \
    --dns-label "workers" \
    --prohibit-public-ip-on-vnic true \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

LB_SUBNET_ID=$(oci network subnet create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-lb-subnet" \
    --cidr-block "10.0.20.0/24" \
    --route-table-id "${PUBLIC_RT_ID}" \
    --security-list-ids "[\"${SL_ID}\"]" \
    --dns-label "loadbalancers" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

BASTION_SUBNET_ID=$(oci network subnet create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --vcn-id "${VCN_ID}" \
    --display-name "${CLUSTER_NAME}-bastion-subnet" \
    --cidr-block "10.0.30.0/24" \
    --route-table-id "${PUBLIC_RT_ID}" \
    --security-list-ids "[\"${SL_ID}\"]" \
    --dns-label "bastion" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)
```

## Step 3: Create private OKE cluster

```bash
oci ce cluster create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --name "${CLUSTER_NAME}" \
    --vcn-id "${VCN_ID}" \
    --kubernetes-version "${KUBERNETES_VERSION}" \
    --endpoint-subnet-id "${API_SUBNET_ID}" \
    --service-lb-subnet-ids "[\"${LB_SUBNET_ID}\"]" \
    --endpoint-public-ip-enabled false \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}"
```

Wait for the cluster to become ACTIVE (~10 minutes):

```bash
# Poll until ACTIVE
CLUSTER_ID=$(oci ce cluster list \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --name "${CLUSTER_NAME}" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query 'data[0].id' --raw-output)

watch -n 30 "oci ce cluster get --cluster-id ${CLUSTER_ID} \
    --profile ${OCI_PROFILE} --region ${OCI_REGION} \
    --query 'data.\"lifecycle-state\"' --raw-output"
```

**Do not proceed to Step 5 until the cluster is ACTIVE.**

## Step 4: Create OCI Bastion

```bash
BASTION_ID=$(oci bastion bastion create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --bastion-type STANDARD \
    --target-subnet-id "${BASTION_SUBNET_ID}" \
    --name "${CLUSTER_NAME}-bastion" \
    --client-cidr-list "[\"$(curl -s https://ifconfig.me)/32\"]" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)
```

## Step 5: Create node pools

Find the GPU-compatible node image and create both pools:

```bash
GPU_IMAGE_ID=$(oci ce node-pool-options get \
    --node-pool-option-id all \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.sources[?contains(\"source-name\", 'GPU') && \
             contains(\"source-name\", 'OKE-${KUBERNETES_VERSION#v}')].\"image-id\" | [0]" \
    --raw-output)

CPU_IMAGE_ID=$(oci ce node-pool-options get \
    --node-pool-option-id all \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.sources[?contains(\"source-name\", 'OKE-${KUBERNETES_VERSION#v}') && \
             !contains(\"source-name\", 'GPU') && \
             contains(\"source-name\", 'aarch64')==\`false\`].\"image-id\" | [0]" \
    --raw-output)

# Pick an availability domain
# Phoenix: AD-2 (index 1) had GPU capacity at validation time.
# Adjust the index if your region has different GPU availability.
AD=$(oci iam availability-domain list \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data[1].name" --raw-output)

# CPU node pool (boot volume >= 100 GB for the router image)
oci ce node-pool create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --cluster-id "${CLUSTER_ID}" \
    --name "cpu-pool" \
    --kubernetes-version "${KUBERNETES_VERSION}" \
    --node-shape "VM.Standard.E5.Flex" \
    --node-shape-config '{"ocpus": 2, "memoryInGBs": 16}' \
    --node-image-id "${CPU_IMAGE_ID}" \
    --node-boot-volume-size-in-gbs 100 \
    --size 1 \
    --placement-configs "[{\"availabilityDomain\": \"${AD}\", \"subnetId\": \"${WORKER_SUBNET_ID}\"}]" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}"

# GPU node pool (boot volume 200 GB)
oci ce node-pool create \
    --compartment-id "${OCI_COMPARTMENT_ID}" \
    --cluster-id "${CLUSTER_ID}" \
    --name "gpu-pool" \
    --kubernetes-version "${KUBERNETES_VERSION}" \
    --node-shape "VM.GPU.A10.1" \
    --node-image-id "${GPU_IMAGE_ID}" \
    --node-boot-volume-size-in-gbs 200 \
    --size 1 \
    --placement-configs "[{\"availabilityDomain\": \"${AD}\", \"subnetId\": \"${WORKER_SUBNET_ID}\"}]" \
    --initial-node-labels '[{"key": "app", "value": "gpu"}, {"key": "nvidia.com/gpu", "value": "true"}]' \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}"
```

Wait for both node pools to show nodes as ACTIVE (~10 minutes):

```bash
watch -n 30 "oci ce node-pool list \
    --compartment-id ${OCI_COMPARTMENT_ID} \
    --cluster-id ${CLUSTER_ID} \
    --profile ${OCI_PROFILE} --region ${OCI_REGION} \
    --query 'data[].{name:name,nodes:nodes[].{ip:\"private-ip\",state:\"lifecycle-state\"}}'"
```

**Do not proceed to Step 6 until both node pools show nodes as ACTIVE.**

**Important:** The CPU boot volume must be at least **100 GB**. The vLLM router
image is ~10.5 GB and the default 47 GB boot volume causes pod eviction.

## Step 6: Connect to the private cluster

Download kubeconfig and configure for tunnel access:

```bash
oci ce cluster create-kubeconfig \
    --cluster-id "${CLUSTER_ID}" \
    --file ~/.kube/config-nemotron \
    --region "${OCI_REGION}" \
    --token-version 2.0.0 \
    --kube-endpoint PRIVATE_ENDPOINT \
    --profile "${OCI_PROFILE}" --overwrite

export KUBECONFIG=~/.kube/config-nemotron

# Get the private endpoint IP
PRIVATE_IP=$(oci ce cluster get --cluster-id "${CLUSTER_ID}" \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query 'data.endpoints."private-endpoint"' --raw-output | cut -d: -f1)

# Update kubeconfig to use localhost tunnel
CLUSTER_CTX=$(kubectl config view --minify -o jsonpath='{.clusters[0].name}')
kubectl config set-cluster "${CLUSTER_CTX}" \
    --server=https://127.0.0.1:6443 \
    --insecure-skip-tls-verify=true
```

If your OCI CLI profile is not `DEFAULT`, add it to the kubeconfig:

```yaml
# In the users[].user.exec section, replace env: [] with:
env:
  - name: OCI_CLI_PROFILE
    value: YOUR_PROFILE
```

Create a Bastion session and start the SSH tunnel:

```bash
SESSION_ID=$(oci bastion session create-port-forwarding \
    --bastion-id "${BASTION_ID}" \
    --target-private-ip "${PRIVATE_IP}" \
    --target-port 6443 \
    --session-ttl 10800 \
    --display-name "nemotron-kubectl" \
    --ssh-public-key-file ~/.ssh/id_ed25519.pub \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}" \
    --query "data.id" --raw-output)

# Wait for session to become ACTIVE, then start tunnel
ssh -i ~/.ssh/id_ed25519 -N -L 6443:${PRIVATE_IP}:6443 \
    -p 22 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    ${SESSION_ID}@host.bastion.${OCI_REGION}.oci.oraclecloud.com &

# Verify
kubectl get nodes
```

**Note:** Bastion sessions expire after the TTL (default 3 hours). Create a
new session and restart the tunnel when access drops.

## Step 7: Expand boot volume filesystems

OCI boot volumes provision only ~47 GB of usable root filesystem regardless
of the requested size. Both nodes must be expanded.

**Why this matters:** The vLLM engine image is ~10 GB, the router image is
~10.5 GB, and the model weights are ~16 GB. Without expansion, pods get
evicted for low ephemeral storage.

For each node, run the following (use a unique pod name per node):

```bash
NODE_IP=<node-internal-ip>
POD_NAME=expand-$(echo $NODE_IP | tr '.' '-')

kubectl run ${POD_NAME} --restart=Never \
  --image=busybox:latest \
  --overrides="{
    \"spec\":{
      \"nodeName\":\"${NODE_IP}\",
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

kubectl wait --for=condition=Ready pod/${POD_NAME} --timeout=60s

kubectl exec ${POD_NAME} -- chroot /host bash -c '
  growpart /dev/sda 3
  sleep 3
  pvresize /dev/sda3
  lvextend -l +100%FREE /dev/ocivolume/root
  xfs_growfs /
  df -h /
'

kubectl exec ${POD_NAME} -- nsenter -t 1 -m -p -- systemctl restart kubelet
kubectl delete pod ${POD_NAME} --force
```

Repeat for each node. Expected results:

- GPU node (200 GB boot volume): 36 GB → ~189 GB usable
- CPU node (100 GB boot volume): 36 GB → ~89 GB usable

## Step 8: Create StorageClasses

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
EOF
```

## Step 9: Patch CoreDNS for GPU tolerations

```bash
kubectl patch deployment coredns -n kube-system --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/tolerations/-",
        "value":{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}}]'

kubectl patch deployment kube-dns-autoscaler -n kube-system --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/tolerations/-",
        "value":{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}}]'
```

## Step 10: Create the templates PVC

The `vllm-stack` chart (0.1.10) mounts a `vllm-templates-pvc` volume in every
engine pod. This PVC must exist before deploying:

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

## Step 11: Deploy vLLM

The checked-in values file
[`vllm_oke_phoenix_private_values.yaml`](./vllm_oke_phoenix_private_values.yaml)
contains the validated configuration for this deployment.

```bash
helm repo add vllm https://vllm-project.github.io/production-stack
helm repo update

helm upgrade --install vllm vllm/vllm-stack \
  -n default \
  -f vllm_oke_phoenix_private_values.yaml
```

**Do not** pass `--wait` to Helm. The engine pod takes several minutes to pull
the image (~10 GB) and download the model.

Monitor progress:

```bash
kubectl get pods -n default -w
```

Wait for both pods to show `1/1 Running`:

- `vllm-deployment-router-*` — request router (CPU node)
- `vllm-llama31-nemotron-nano-8b-deployment-vllm-*` — model engine (GPU node)

## Step 12: Validate

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
curl -s http://127.0.0.1:8080/v1/models | jq .
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
        "parameters": {"type": "object", "properties": {}, "required": []}
      }
    }]
  }'
```

Expected: `finish_reason` set to `tool_calls`.

## Key vLLM settings

| Setting | Value | Why |
|---------|-------|-----|
| `tag` | `v0.19.0` | Pinned to validated vLLM version |
| `maxModelLen` | `4096` | Conservative context to fit single A10 (24 GB) |
| `gpuMemoryUtilization` | `0.95` | Maximize GPU memory for KV cache |
| `enableTool` | `true` | Enable tool / function calling |
| `toolCallParser` | `llama3_json` | Parser matching Nemotron's tool format |
| `extraArgs` | `--chat-template=...` | Template passed as CLI arg (chart's `chatTemplate` field prepends `/templates/`) |
| `storageClass` | `oci-block-storage-enc` | OCI Block Volume with balanced performance |

## Troubleshooting

### Pods evicted for ephemeral storage

OCI boot volumes provision only ~47 GB of usable filesystem by default.
Follow Step 7 to expand. If the boot volume itself is too small (default
47 GB), resize it first via the OCI CLI, then rescan the block device before
running `growpart`:

```bash
echo 1 > /sys/class/block/sda/device/rescan
```

### Engine pod stays Pending with PVC not found

The `vllm-stack` chart (0.1.10) requires `vllm-templates-pvc` to exist
before the engine pod can schedule. See Step 10.

### Engine pod crashes with chat template error

The chart's `chatTemplate` field prepends `/templates/` to the path. Pass
the template via `vllmConfig.extraArgs` instead:

```yaml
vllmConfig:
  extraArgs:
    - "--chat-template=/vllm-workspace/examples/tool_chat_template_llama3.1_json.jinja"
```

### Tool calling does not work

Ensure all of these are set in the values file:

- `enableTool: true`
- `toolCallParser: llama3_json`
- `--chat-template=...` in `vllmConfig.extraArgs`

### `kubectl` cannot reach the cluster

Re-establish the Bastion tunnel. Sessions expire after the configured TTL.

### Helm upgrade fails with field manager conflict

Uninstall and reinstall:

```bash
helm uninstall vllm -n default
helm install vllm vllm/vllm-stack -n default -f vllm_oke_phoenix_private_values.yaml
```

## Cleanup

To tear down all resources:

```bash
# 1. Uninstall Helm release and PVCs
helm uninstall vllm -n default
kubectl delete pvc --all -n default

# 2. Delete node pools
oci ce node-pool delete --node-pool-id <cpu-pool-id> --force \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}"
oci ce node-pool delete --node-pool-id <gpu-pool-id> --force \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}"

# 3. Delete cluster (wait for node pools to finish deleting first)
oci ce cluster delete --cluster-id "${CLUSTER_ID}" --force \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}"

# 4. Delete bastion
oci bastion bastion delete --bastion-id "${BASTION_ID}" --force \
    --profile "${OCI_PROFILE}" --region "${OCI_REGION}"

# 5. Delete subnets, gateways, and VCN (wait for cluster deletion first)
```

## Alternative: Terraform

A Terraform sample using the `oracle-terraform-modules/oke/oci` module is
available in [`terraform/`](./terraform/) for reference. Note that the
module's NSG configuration requires its built-in bastion compute host
(`create_bastion = true`) for OCI Bastion port-forwarding to work. The
manual CLI approach above is recommended for initial deployments.
