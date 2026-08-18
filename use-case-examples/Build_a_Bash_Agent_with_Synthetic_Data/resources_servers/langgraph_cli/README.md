# LangGraph CLI verifier

This is a NeMo Gym 0.3 resource server for a single-step structured-output task. Each dataset row contains OpenAI Responses API input parameters and an `expected_output` LangGraph CLI object. The `/verify` endpoint compares the policy response with that object and returns a reward in `[-1, 1]`.

From the tutorial root:

```bash
uv run --project training ng_dump_config "+config_paths=[resources_servers/langgraph_cli/configs/langgraph_cli.yaml]"
uv run --project training ng_test +entrypoint=resources_servers/langgraph_cli
uv run --project training ng_run "+config_paths=[resources_servers/langgraph_cli/configs/langgraph_cli.yaml]"
```

In another terminal, check liveness:

```bash
curl http://127.0.0.1:8000/
```

The example dataset is at `data/example.jsonl`. The generation notebook writes larger train and validation datasets to the tutorial's ignored `data/langgraph_cli/` directory.
