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
