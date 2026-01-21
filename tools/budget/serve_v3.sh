#!/usr/bin/env sh

# Set some default thinking budget settings.
# 'thinking_budget' is the number of tokens the model can use in thinking/reasoning stage.
# 'thinking_budget_grace_period' extra number of tokens after the budget to find a newline to gracefully stop thinking
# 'end_token_ids' the sequence of tokens to artificially insert into the token stream to end thinking
# 'end_think_ids' the id for </think> (always 13 for this model)
# 'prompt_think_ids' the sequence to ids to allow the logit processor to recognize that the model is in thinking stage (always [12, 1010] for this model)

export THINKING_BUDGET_LOGITS_PROCESSOR_ARGS='{"thinking_budget": 150, "thinking_budget_grace_period": 30, "end_token_ids": [1338, 13], "end_think_ids": [[13]], "prompt_think_ids": [12, 1010]}'

# Start a VLLM server with the `--logits-processor` argument 
python3 -m vllm.entrypoints.openai.api_server \
	--served-model-name "model"  \
	--model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
	--logits-processors "custom_logit_processors.v1.nano_v3_logit_processors:ThinkingBudgetLogitsProcessor" \
	--port 8881 \
	--trust-remote-code 
