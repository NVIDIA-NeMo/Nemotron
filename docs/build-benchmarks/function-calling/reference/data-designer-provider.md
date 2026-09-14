<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Data Designer Provider Configuration

NeMo Data Designer routes model calls made during assisted authoring and optional
publication paraphrasing. Configure a provider before using:

- model-assisted source scaffolding;
- model-drafted probe plans;
- `bfcl_author draft`;
- a publication profile with model paraphrasing enabled.

Manual Oracle Pack authoring and template-only publication do not require Data
Designer or this provider configuration. Candidate evaluation has a separate endpoint
configuration.

The registry supports OpenAI-compatible providers generally. The configuration below
uses NVIDIA Inference API as a concrete example; substitute the endpoint, provider
name, served model, and credential environment-variable name for another compatible
provider.

## Create The Provider Registry

Choose a local configuration directory and point `DATA_DESIGNER_HOME` to it:

```bash
export DATA_DESIGNER_HOME="$HOME/.config/nemotron/data-designer"
mkdir -p "$DATA_DESIGNER_HOME"
```

Create `$DATA_DESIGNER_HOME/model_providers.yaml`. This NVIDIA Inference API example
uses the OpenAI-compatible transport:

```yaml
providers:
  - name: nvidia_inference_api
    endpoint: https://inference-api.nvidia.com/v1
    provider_type: openai
    api_key: NGC_API_KEY
```

`api_key` is the name of an environment variable, not a credential value. Export the
credential in the process that runs the authoring or publication command:

```bash
export NGC_API_KEY="<provider credential>"
test -f "$DATA_DESIGNER_HOME/model_providers.yaml"
```

Do not commit a literal credential to `model_providers.yaml`, a BFCL configuration,
an authoring workspace, or a shell script. Use the secret-management mechanism
approved for the execution environment.

The provider endpoint is the OpenAI-compatible API root. For NVIDIA Inference API,
use `https://inference-api.nvidia.com/v1`; the client adds `/chat/completions`. An
endpoint ending in `/v1/chat/completions` would duplicate the route.

## Select A Model

Authoring commands receive the provider route and reviewed model identity explicitly:

```text
--model-alias author
--model-provider nvidia_inference_api
--model nvidia/<publisher>/<served-model>
--model-canonical-id <immutable-or-provider-managed-identity>
```

The provider `name` passed through `--model-provider` must exactly match an entry in
`model_providers.yaml`. The served `--model` is the identifier accepted by that
endpoint. `--model-canonical-id` is the identity recorded in authoring provenance; use
an immutable revision when one is available, or an explicitly reviewed
provider-managed identity when the provider does not expose a revision.

`model_configs.yaml` is optional. It can hold reusable model aliases and inference
defaults for Data Designer applications, but BFCL authoring does not require it when
the command supplies provider, model, temperature, seed, and timeout explicitly.

## Verify The Route

There is no BFCL command that approves a provider independently. The first
model-assisted command runs Data Designer's provider health check before generation.
For example:

```bash
python -m nemotron.steps.byob.scripts.draft_probe_plan \
  --source /srv/sources/my-domain \
  --domain-brief /srv/sources/my-domain-brief.txt \
  --output /srv/sources/my-domain-probe-plan.json \
  --clock 2026-03-02T02:00:00Z \
  --model-alias author \
  --model-provider nvidia_inference_api \
  --model nvidia/<publisher>/<served-model> \
  --model-canonical-id <immutable-or-provider-managed-identity>
```

A successful health check proves that the route and credential can make a request. It
does not authorize the model to see source evidence. Assisted authoring still requires
the exposure authorization and evidence approval described in
{doc}`../how-to/assisted-authoring`.

## Common Failures

- **Unknown provider:** make `--model-provider` match the provider registry's `name`
  and confirm `DATA_DESIGNER_HOME` points to the directory containing the registry.
- **Authentication failure:** export the environment variable named by `api_key` and
  verify that the credential is authorized for the served model.
- **Route not found:** configure the API root, not the final
  `/chat/completions` route.
- **Model not found:** use the endpoint's served model identifier; do not substitute
  the canonical provenance identity as the request model unless the provider accepts
  it.
- **Unsupported inference field:** remove or adjust only fields the provider rejects,
  then preserve the resolved inference settings in run provenance.

## Related Information

- {doc}`../how-to/assisted-authoring` for source certification and bounded model
  drafting.
- {doc}`../how-to/publish-a-release` for optional model paraphrasing during
  publication.
