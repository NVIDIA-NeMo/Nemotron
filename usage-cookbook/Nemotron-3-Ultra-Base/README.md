# **NVIDIA Nemotron 3 Ultra — Base Model**

**Announced at GTC San Jose 2026** · Best Open Base Model

Nemotron 3 Ultra is NVIDIA's largest open model: **550B total parameters with up to 55B active per token** via a hybrid Mamba-Transformer mixture-of-experts (MoE) architecture.

Similar to Nemotron 3 Super, it was pre-trained using NVFP4 and shares the same core technical innovations:

* **LatentMoE** — Compresses tokens into a low-rank latent space before routing, enabling 4× as many expert specialists for the same inference cost.  
* **Multi-Token Prediction (MTP)** — Predicts multiple future tokens in a single forward pass, improving chain-of-thought coherence and enabling built-in speculative decoding at inference time.  
* **1M Token Context Length** — Mamba-2 layers provide linear-time complexity over sequence length, making 1M-token context practical for long-document and agentic workloads.

Nemotron 3 Ultra is a **pre-training base checkpoint** — it has not undergone instruction tuning or post-training alignment. This means it is not meant to be used out of the box as an assistant or in a production pipeline.

It is designed to be the best possible **starting point for customization**: fine-tuning on domain data, reinforcement learning post-training, and custom instruction tuning pipelines. If you're looking for a model you can deploy directly, wait for the post-trained release.

---

## **Benchmark Results**

Measured on **NVIDIA GB200 NVL72** against GLM-4.5-355B-A32B and Kimi-K2-1026B-A33B, Nemotron 3 Ultra base model delivers up to 5x higher TPS at max throughput and leading accuracy for various agentic tasks:

| Benchmark | Ultra 550B-A55B | GLM-4.5-355B-A32B | Kimi-K2-1026B-A33B |
| ----- | ----- | ----- | ----- |
| MMLU Pro | **79.0** | 65.6 | 69.3 |
| MMLU | **89.1** | 86.3 | 88.0 |
| Code | **85.3** | 76.2 | 75.3 |
| Math | **85.4** | 72.1 | 79.5 |
| Common Sense | 81.0 | 81.3 | **81.6** |
| Multilingual | **89.0** | 83.3 | 84.2 |
| Peak Throughput | **5×** vs GLM | 1× | \~2.5× |

## **Availability**: 

Weights will become available with the full release of Nemotron 3 Ultra, expected to release in 1H 2026\.
