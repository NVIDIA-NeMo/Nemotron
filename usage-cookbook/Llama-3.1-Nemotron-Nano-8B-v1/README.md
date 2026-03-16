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

This gives Nemotron users a reproducible OCI deployment path comparable to the
AWS-style GPU plus Kubernetes patterns many teams already use, while keeping
the control plane and inference path private.

## Why this configuration

This setup gives Nemotron users a reproducible OCI deployment path with a small
single-GPU footprint while preserving tool calling, structured output, and
streaming support.

Validated capabilities on this deployment:

- chat completion
- structured output
- tool calling
- streaming
- async/concurrent requests
- OpenAI-compatible model discovery via `/v1/models`

## Tested environment

- Region: `us-phoenix-1`
- Kubernetes: OKE private cluster
- GPU shape: `VM.GPU.A10.1`
- Model: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
- Serving stack: `vLLM`
- Inference API: OpenAI-compatible `/v1`

## Architecture

1. Create a **private** OKE cluster in Phoenix.
2. Create a CPU node pool and a GPU node pool.
3. Use **OCI Bastion** to reach the cluster API locally.
4. Deploy Nemotron with the checked-in `vLLM` values file.
5. Keep the inference service internal and validate it through a local
   port-forward.

## Prerequisites

- OCI tenancy with Phoenix capacity for `VM.GPU.A10.1`
- OKE permissions
- OCI Bastion permissions
- `kubectl`
- `helm`
- access to pull the Nemotron model from Hugging Face or an equivalent model
  artifact source accepted by your environment

## Deployment notes

This cookbook assumes a private OKE cluster. Keep these constraints:

- disable the public Kubernetes control-plane endpoint
- do not attach public IPs to worker nodes
- do not expose the model through a public load balancer

The known-good serving values are in
[`vllm_oke_phoenix_private_values.yaml`](./vllm_oke_phoenix_private_values.yaml).

Terraform for the private Phoenix OKE infrastructure is available in
[`terraform/`](./terraform/).

That Terraform path was validated end to end in Phoenix through:

- VCN and private subnets
- private OKE control plane
- OCI Bastion service
- CPU node pool
- GPU node pool on `VM.GPU.A10.1`

Important settings for this single-A10 deployment:

- `maxModelLen: 4096`
- `gpuMemoryUtilization: 0.95`
- `enableTool: true`
- `toolCallParser: llama3_json`
- `chatTemplate: /vllm-workspace/examples/tool_chat_template_llama3.1_json.jinja`

These settings were required to make the model stable on a single A10 while
preserving tool-calling behavior.

## Example install flow

Deploy the serving stack with the `vLLM Production Stack` Helm chart using the
checked-in values file:

```bash
helm upgrade --install vllm <path-to-vllm-production-stack-helm-chart> \
  -n default \
  -f usage-cookbook/Llama-3.1-Nemotron-Nano-8B-v1/vllm_oke_phoenix_private_values.yaml
```

Use Bastion plus the private cluster endpoint for cluster access. Then
port-forward the router service locally:

```bash
kubectl -n default port-forward svc/vllm-router-service 8080:80
```

At that point, the local validation endpoint is:

```text
http://127.0.0.1:8080/v1
```

## Validation

Health check:

```bash
curl -s http://127.0.0.1:8080/health
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

## Tool-calling smoke test

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

## Query via OCI Bastion

For this private deployment, query the cluster and model through the **OCI
Bastion service** plus local forwarding.

Export the Terraform outputs:

```bash
export BASTION_ID="<terraform output oci_bastion_id>"
export PRIVATE_API_HOST="<terraform output apiserver_private_host>"
export REGION="us-phoenix-1"
export OCI_CLI_PROFILE="API_KEY_AUTH"
```

Create a Bastion port-forwarding session to the private OKE API:

```bash
oci bastion session create-port-forwarding \
  --bastion-id "$BASTION_ID" \
  --ssh-public-key-file ~/.ssh/id_ed25519.pub \
  --key-type PUB \
  --target-port 6443 \
  --target-private-ip "$PRIVATE_API_HOST" \
  --display-name nemotron-oke-api \
  --session-ttl 10800 \
  --region "$REGION" \
  --profile "$OCI_CLI_PROFILE"
```

Inspect the created session and copy the SSH command OCI returns:

```bash
oci bastion session get \
  --session-id "<session_ocid>" \
  --region "$REGION" \
  --profile "$OCI_CLI_PROFILE"
```

Run the returned SSH command so that the private Kubernetes API is reachable on
local port `6443`, then query the cluster:

```bash
kubectl get nodes
kubectl -n default get pods
```

Port-forward the Nemotron router service:

```bash
kubectl -n default port-forward svc/vllm-router-service 8080:80
```

At that point, the private model is queryable locally without exposing a public
inference endpoint:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "messages": [{"role": "user", "content": "Reply with NEMOTRON_OK"}]
  }'
```

## Operational notes

- Phoenix provided a workable path for this deployment when Chicago GPU capacity
  was not available.
- A single A10 is enough for the validated setup, but it requires conservative
  context sizing.
- Private access plus local forwarding keeps the control plane and inference
  path off the public internet.

## Troubleshooting

### The model pod starts but never becomes ready

Reduce context pressure and ensure the `vLLM` values include:

- `maxModelLen: 4096`
- `gpuMemoryUtilization: 0.95`

### Tool calling does not work

Make sure all of these are set:

- `enableTool: true`
- `toolCallParser: llama3_json`
- `chatTemplate: /vllm-workspace/examples/tool_chat_template_llama3.1_json.jinja`

### `kubectl` cannot reach the cluster

This guide assumes a **private** OKE cluster. Re-establish the Bastion tunnel
before using `kubectl`.

### The endpoint is reachable but `/v1/models` is empty or wrong

Confirm the deployment is serving:

- `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`

and that the router service is forwarding to the Nemotron backend pods.
