# Terraform: Private OCI OKE for Llama-3.1-Nemotron-Nano-8B-v1

This Terraform example provisions the **private-only** OCI infrastructure for
the validated Phoenix deployment described in the parent cookbook.

It is intended to give Nemotron users a reproducible OCI path comparable to the
AWS-style GPU plus Kubernetes deployment patterns often used for NVIDIA model
serving, while preserving a private-only topology.

It creates:

- a VCN
- a **private** OKE cluster
- a private CPU node pool
- a private GPU node pool targeting `VM.GPU.A10.1`
- an **OCI Bastion service** resource for private access

It does **not** create:

- a public Kubernetes API endpoint
- public worker-node IPs
- a public bastion host
- a public inference endpoint

## Bastion note

This sample provisions the **OCI Bastion service** so that private-cluster
access is reproducible from Terraform.

That is intentionally different from creating a public bastion VM:

- no public bastion compute instance is created
- no worker node receives a public IP
- the Kubernetes API remains private

If your environment already manages private-cluster access through a separate
operator workflow, you can remove the `oci_bastion_bastion` resource and keep
the rest of the sample unchanged.

## Module choice

This wrapper intentionally uses Oracle's official OKE Terraform module:

- `oracle-terraform-modules/oke/oci`

The Nemotron-specific layer in this directory adds:

- the Phoenix defaults
- the no-public-IP constraints
- the A10-focused worker pool defaults
- the OCI Bastion service resource required for private access

## Files

- [`main.tf`](./main.tf) - private OKE cluster, worker pools, OCI Bastion
- [`variables.tf`](./variables.tf) - deployment inputs
- [`outputs.tf`](./outputs.tf) - useful IDs and private endpoint information
- [`terraform.tfvars.example`](./terraform.tfvars.example) - starting point

## Usage

```bash
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

The validated live run completed successfully in `us-phoenix-1`, including:

- private OKE cluster creation
- OCI Bastion service creation
- CPU node pool creation
- GPU node pool creation on `VM.GPU.A10.1` in `PHX-AD-2`

After the infrastructure is ready:

1. create an OCI Bastion session to reach the private cluster
2. deploy the model with:
   - [`../vllm_oke_phoenix_private_values.yaml`](../vllm_oke_phoenix_private_values.yaml)
3. validate:
   - `/health`
   - `/v1/models`
   - chat completion
   - tool calling
   - streaming

## Notes

- The validated live deployment used `us-phoenix-1`.
- The validated GPU pool used Phoenix `AD-2`, exposed as `gpu_placement_ads`.
- The Bastion resource here is the OCI managed Bastion service, not a public
  bastion VM.
- `ssh_public_key_path` must point to an actual OpenSSH public key file; the
  wrapper reads the file contents with Terraform's `file()` function before
  passing it to OKE.
