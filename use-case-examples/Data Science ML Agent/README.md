# Data Science ML Agent

This example demonstrates how to build a <b>natural language-driven data science and machine learning agent</b> powered by <b>NVIDIA GPUs</b>.
The agent allows users to perform data exploration, model training, and hyperparameter optimization interactively using <b>RAPIDS cuDF</b> and <b>cuML</b> for GPU acceleration. 

## Overview

The Data Science ML Agent enables natural language interaction for common data science workflows, including dataset loading, target selection, model training, and performance optimization.
It combines the flexibility of LLMs with GPU-accelerated computation to simplify and speed up end-to-end machine learning pipelines.

## LLM Used

[NVIDIA Nemotron Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2): A compact, open-source large language model optimized for reasoning and data analysis tasks.

## Key Features

- Natural language interface for running data exploration and ML workflows
- GPU acceleration with cuDF (DataFrame operations) and cuML (ML algorithms)
- Support for CPU mode using pandas and scikit-learn
- Simple setup and execution through Streamlit interface
- Compatible with both small and large-scale datasets

## Requirements

- RAPIDS 25.10
- Python 3.10, 3.11, 3.12, or 3.13
- CUDA 12.0 or 13.0 compatible NVIDIA GPU (for GPU mode)
- NVIDIA API key ([get one here](https://build.nvidia.com/))
- streamlit, optuna, joblib

Please refer to the [official RAPIDS installation documentation](https://docs.rapids.ai/install/) for detailed instructions.

Installation Example:
```bash
conda create -n rapids-25.10 -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.10 python=3.11 'cuda-version=13.0'
```

## Example Usage

```bash
conda activate rapids-25.10

export NVIDIA_API_KEY=""

# GPU-accelerated mode
python -m cudf.pandas -m cuml.accel -m streamlit run user_interface.py

# CPU-based mode
streamlit run user_interface.py
```

Example queries:
```bash
load dataset Titanic-Dataset.csv/Titanic-Dataset-test.csv
set target variable to be 'Survived'
train classification/regression model
optimize svc with 50 trials
optimize forest regressor with 30 trials
show best model by r2
make inference for the test dataset
...
```

## 📊 Sample Dataset

This project provides sample datasets, you can download the Kaggle Titanic Datasets [train.csv](https://www.kaggle.com/competitions/titanic) and [test.csv](https://www.kaggle.com/competitions/titanic) into the data folder. You can also create **train-1M**, an extrapolated version scaled to 1M rows of the train.csv using the script extrapolation.py. 


**Note:**  
- Ensure you have the appropriate dependencies installed for each mode.  
- GPU mode requires a supported NVIDIA GPU and the RAPIDS ecosystem installed.
