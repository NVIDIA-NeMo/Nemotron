# Get started with Nemotron 3.5 Super VL in agent harnesses

Get an [NVIDIA API key](https://build.nvidia.com/settings/api-keys), then choose your harness below. `MODEL_ID` means the exact Super VL model ID enabled for your account.

| Harness | Start with Nemotron | Official instructions |
|---|---|---|
| Hermes | Set `NVIDIA_API_KEY`; use the Super VL command in the guide with your enabled model ID. | [NVIDIA setup](https://hermes-agent.nousresearch.com/docs/integrations/providers#nvidia-nim) |
| OpenCode | `/connect` → NVIDIA and enter your key; `/models` → Super VL when listed. | [NVIDIA setup](https://opencode.ai/docs/providers/#nvidia) |
| OpenClaw | With `NVIDIA_API_KEY` set, run `openclaw onboard --auth-choice nvidia-api-key`; select `nvidia/<MODEL_ID>` with `openclaw models set`. | [NVIDIA setup](https://docs.openclaw.ai/providers/nvidia) |
| Pi | `/login` → NVIDIA NIM; `/model` → Super VL when listed. | [Model selection and custom endpoints](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md) |
| LangChain Deep Agents | Set `NVIDIA_API_KEY`; pass `ChatNVIDIA(model=MODEL_ID)` to `create_deep_agent(model=...)`. | [NVIDIA provider](https://docs.langchain.com/oss/python/integrations/providers/nvidia#chat-chatnvidia) · [Choose the model](https://docs.langchain.com/oss/python/deepagents/customization#model) |
| Codex CLI | Use a Responses-compatible gateway exposing Super VL; configure Codex's provider and gateway model name. A Chat Completions-only endpoint is insufficient. | [Provider setup](https://learn.chatgpt.com/docs/config-file/config-advanced#custom-model-providers) · [Gateway requirements](https://learn.chatgpt.com/docs/enterprise/gateway-compatibility#requests-and-endpoints) |
| NOOA | Set `NVIDIA_NIM_API_KEY`; use `get_llm_client("nvidia_nim/" + MODEL_ID)` in the quickstart. | [Quickstart](https://github.com/NVIDIA-NeMo/labs-OO-Agents#quick-start) · [NVIDIA provider](https://docs.litellm.ai/docs/providers/nvidia_nim) |
