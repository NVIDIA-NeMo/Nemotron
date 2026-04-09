from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8881/v1", # Your vLLM server URL
    api_key="EMPTY"
)

result = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Consider all the ways you can interpret the question 'What is 5.9 plus 6.1' and give the best answer possible."}
    ],
    temperature=1.0,
    max_tokens=12200, # uses the default thinking budget set during starting of the vllm server.
)

print("*" * 100)
print("default thinking budget behavior:")
thinking_part, delim, answer_part = result.choices[0].message.content.partition("</think>")
print(thinking_part + delim)

custom_think_truncation = [1871, 5565, 11483, 6139, 2016, 1536, 6934, 1338, 13] # "Reached thining limit set by client\n\n</think>
custom_think_budget = 10
custom_think_budget_grace_period = 10

result = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Consider all the ways you can interpret the question 'What is 5.9 plus 6.1' and give the best answer possible."}
    ],
    temperature=1.0,
    max_tokens=12200,
    logprobs=False,
    extra_body={
        "vllm_xargs": {
            "thinking_budget": custom_think_budget,
            "thinking_budget_grace_period": custom_think_budget_grace_period,
            "end_token_ids": json.dumps(custom_think_truncation),
        }
    }
)

print("*" * 100)
print("thinking budget behavior customized in client:")
thinking_part, delim, answer_part = result.choices[0].message.content.partition("</think>")
print(thinking_part + delim)
