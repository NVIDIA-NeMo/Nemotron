---
library_name: transformers
license: other
license_name: nvidia-internal-scientific-research-and-development-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/
pipeline_tag: text-generation
language:
  - en
  - de
  - es
  - fr
  - it
  - ko
  - pt
  - ru
  - jp
  - zh
tags:
  - nvidia
  - pytorch
  - nemotron-h
---

# Nemotron-H-56B-Base-8K

## Model Overview 

NVIDIA Nemotron-H-56B-Base-8K is a large language model (LLM) developed by NVIDIA that is designed as a completion model for a given piece of text. It uses a hybrid model architecture that consists primarily of Mamba-2 and MLP layers combined with just ten Attention layers. The model features a context length of 8K. The supported languages include: English, German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, and Chinese. For more detailed information on the model architecture, training, and evaluation, please see the [project page](https://research.nvidia.com/labs/adlr/nemotronh/) and the [technical report](https://arxiv.org/abs/2504.03624).

For best performance on a given task, users are encouraged to customize the model using the [NeMo Framework](https://docs.nvidia.com/nemo-framework/index.html) suite of customization tools including Parameter-Efficient Fine-Tuning (P-tuning, Adapters, LoRA, and more), and Model Alignment (SFT, SteerLM, RLHF, and more) using [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner).

This model is for research and development only.

This model is part of the Nemotron-H Collection. You can find the models in this family here:
- [Nemotron-H-56B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-56B-Base-8K)
- [Nemotron-H-47B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-47B-Base-8K)
- [Nemotron-H-8B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K)

## License/Terms of Use

GOVERNING TERMS: Use of this model is governed by the [NVIDIA Internal Scientific Research and Development Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/).

**Model Developer:** NVIDIA

**Model Dates:**

October 2024 - March 2025

**Data Freshness:**

September 2024

The pretraining data has a cutoff date of September 2024. 


## Use Case:

This model is intended for developers and researchers building LLMs.

## Release Date: 

4/14/2025

## References

- [\[2504.03624\] Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models](https://arxiv.org/abs/2504.03624)

## Model Architecture
- Architecture Type: Hybrid Mamba-Transformer
- Network Architecture: Nemotron-H

This model has 56B model parameters.

## Input
- Input Type(s): Text 
- Input Format(s): String
- Input Parameters: One-Dimensional (1D): Sequences
- Other Properties Related to Input: Context length up to 8K. Supported languages include German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, Chinese and English.

## Output
- Output Type(s): Text 
- Output Format: String
- Output Parameters: One-Dimensional (1D): Sequences

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br> 

## Software Integration
- Runtime Engine(s): NeMo 24.12
- Supported Hardware Microarchitecture Compatibility: NVIDIA H100-80GB, NVIDIA A100
- Operating System(s): Linux

## Model Version
- v1.0

## Prompt Format

As this is a base model, no explicit prompt format is recommended or required. 

### Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer  = AutoTokenizer.from_pretrained("nvidia/Nemotron-H-56B-Base-8K", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("nvidia/Nemotron-H-56B-Base-8K", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

prompt = "When was NVIDIA founded?"

outputs = model.generate(**tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device))
print(tokenizer.decode(outputs[0]))
```

## Training, Testing, and Evaluation Datasets

### Training & Testing Datasets:

The training corpus for Nemotron-H-56B-Base-8K Base consists of English and multilingual text (German, Spanish, French, Italian, Korean, Portuguese, Russian, Japanese, Chinese and English), as well as code. Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. This model was also improved using synthetic data from Qwen (Built with Qwen). The corpus spans domains including legal, math, science, finance, and more. We also include a small portion of question-answering, and alignment style data to improve model accuracies.

**Data Collection for Training & Testing Datasets:**
Hybrid: Automated, Human, Synthetic

**Data Labeling for Training & Testing Datasets:**
Hybrid: Automated, Human, Synthetic

### Evaluation Datasets 

We used the datasets listed in the next section to evaluate Nemotron-H-56B-Base-8K Base. 

Data Collection for Evaluation Datasets:
Hybrid: Human, Synthetic

Data Labeling for Evaluation Datasets:
Hybrid: Human, Synthetic, Automatic

#### Commonsense Understanding Evaluations:

| ARC Challenge 25-shot | Hellaswag 10-shot | Winogrande 5-shot | CommonsenseQA 7-shot |
|-------------|--------------|-----------------|------------------|
| 94.97 | 89.00 | 84.45 | 86.73 |

- ARC (Ai2 reasoning challenge)-Challenge - The challenge set of questions from a benchmark that contains grade-school level, multiple-choice science questions to assess question answering ability of language models. [Dataset](https://huggingface.co/datasets/allenai/ai2_arc)
- Hellaswag - Tests the ability of a language model to correctly finish the provided context from a choice of possible options. [Dataset](https://huggingface.co/datasets/Rowan/hellaswag )
- Winogrande - Tests the ability to choose the right option for a given sentence which requires commonsense reasoning. [Dataset](https://huggingface.co/datasets/allenai/winogrande )
- CommonsenseQA - A multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers. [Dataset](https://huggingface.co/datasets/tau/commonsense_qa  )

#### Coding Evaluations:

| MBPP (sanitized) 3-shot | MBPP+ 0-shot | HumanEval 0-shot | HumanEval+ 0-shot |
|-------------|--------------|-----------------|------------------|
| 77.82 | 67.20| 60.37 | 54.27 |

- MBPP (Mostly Basic Python Programming Problems) - Evaluates ability to generate solutions for Python programming tasks. [Dataset](https://github.com/google-research/google-research/tree/master/mbpp)
- MBPP+ - Extended version of MBPP with additional validation. [Dataset](https://huggingface.co/datasets/evalplus/mbppplus)
- HumanEval - Tests code generation and completion abilities in Python. [Dataset](https://github.com/openai/human-eval)

#### Math Evaluations:


| GSM8K 8-shot CoT |  MATH 4-shot CoT |  MATH-Lvl 5 4-shot CoT  | MATH-500 4-shot CoT |
|--------------|------------|------------|------------|
| 93.71 | 59.42 | 35.19 | 57.37 |

- GSM8K (Grade School Math 8K) - Evaluates grade school level mathematical word problem solving. [Dataset](https://github.com/openai/grade-school-math)
- MATH - Tests mathematical ability across multiple difficulty levels and various subjects including: Prealgebra, Algebra, Number Theory, Counting and Probability, Geometry, Intermediate Algebra, and Precalculus. [Dataset](https://github.com/hendrycks/math)
- MATH Lvl 5 - Only the most difficult questions from the MATH dataset. [Dataset](https://github.com/hendrycks/math)
- MATH-500 - Tests advanced mathematical problem solving across algebra, geometry, and calculus. [Dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)


#### General Evaluations:


| MMLU-Pro 5-shot CoT | MMLU 5-shot|
|-------------------|------------------|
|60.51 |84.21 | 

- MMLU Pro - Evaluates language understanding models across a broad range of challenging, reasoning-focused questions across 14 diverse domains.
[Dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- MMLU - Tests knowledge across 57 subjects including science, humanities, math and more. [Dataset](https://github.com/hendrycks/test)

## Potential Known Risks for Usage

The model was trained on data that contains toxic language, unsafe content, and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

The model demonstrates weakness to indirect prompt injection via some encodings, including Base16, Hex/ASCII, and Braille, though is more resilient than other similar models to injections using the more common Base64 vector.

## Inference
- Engine: NeMo
- Test Hardware NVIDIA H100-80GB

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

For more detailed information on ethical considerations for this model, please see the Responsible Use Guide available at http://nvidia.com/nemotron-responsible-use.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

