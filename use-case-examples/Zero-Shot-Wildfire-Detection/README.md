# Supercharged VLM Inference with State-of-the-Art Zero-Shot Performance

*A practical guide to building and evaluating a wildfire detection system using NVIDIA-Nemotron-Nano-12B-v2-VL*

**By Aastha Jhunjhunwala, Akul Santhosh**

## TL;DR

* Evaluate [NVIDIA-Nemotron-Nano-12B-v2-VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) for wildfire detection with a complete end-to-end pipeline
* Process 410+ test images with comprehensive evaluation metrics in under 20 minutes
* Delivers transparent, explainable predictions using structured prompt design
* Includes visualizations, summary metrics, and exportable CSV/JSON outputs for downstream analysis

## Quick Links

* **HuggingFace Checkpoint**: [NVIDIA-Nemotron-Nano-12B-v2-VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)
* **Dataset**: [The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset?resource=download)
* **Model NIM**: [NVIDIA-Nemotron-Nano-12B-v2-VL NIM](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl)

## Prerequisites

* NVIDIA GPU with atleast 80GB VRAM (tested on 2xA100 80GB) 
* Docker with NVIDIA Container Runtime
* Python 3.12+ with Jupyter
* Wildfire test dataset (fire/no-fire images)
* Network access for HTTP image serving

## The Challenge: Explainable Wildfire Detection

Wildfires are devastating natural disasters that require rapid detection and response. While traditional computer vision models can classify images, they lack the ability to **explain their reasoning**, a critical requirement for emergency response systems where understanding *why* a model flagged an image as fire is as important as the classification itself.

Enter **NVIDIA Nemotron VLM**: a vision-language model that doesn't just classify - it reasons, explains, and provides transparent decision-making for robust, real-time wildfire detection and response.

## Why Vision-Language Models for Wildfire Detection?

Traditional image classifiers output a single label: "fire" or "no fire." But what if the model could tell you:

* "FIRE - The image shows dense smoke plumes rising from forest vegetation, indicating an active wildfire with visible heat distortion in the surrounding area."
* "NO_FIRE - Clear mountain landscape with normal atmospheric conditions. The visible haze appears to be natural fog rather than wildfire smoke."

This **explainability** is transformative for:

✅ **Emergency Response**: Operators understand *why* an alert was triggered  
✅ **False Positive Reduction**: Distinguish between wildfire smoke and normal fog/clouds  
✅ **Training & Debugging**: Understand model behavior and improve prompts  
✅ **Regulatory Compliance**: Provide auditable reasoning for critical decisions

## Our Evaluation Pipeline: From Images to Insights

We built a complete evaluation pipeline that processes wildfire test images through the Nemotron VLM and generates comprehensive metrics and visualizations.

## Step 1: Deploy the Nemotron VLM

We use NVIDIA NIM (NVIDIA Inference Microservices) to deploy the Nemotron Nano 12B v2 VL model:

```bash
export NGC_API_KEY=<your-ngc-api-key>
export LOCAL_NIM_CACHE=~/.cache/nim

docker run -it --rm \
    --gpus '"device=0,1"' \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:latest
```

The model exposes an OpenAI-compatible API at `http://localhost:8000/v1/chat/completions`.

## Step 2: Data Ingestion via HTTP Streaming

**Key Challenge**: The NIM container runs in Docker and requires HTTP/HTTPS URLs to access images-it cannot directly access local file paths from the host machine.

**Solution**: Start a lightweight HTTP server on the host machine to serve images:

```python
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socket

def start_http_server(directory, port):
    """Start HTTP server to serve images to Docker container."""
    class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
        
        def log_message(self, format, *args):
            pass  # Suppress logging
    
    server = HTTPServer(('0.0.0.0', port), QuietHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

# Start server
http_server = start_http_server("/path/to/wildfire/data", 8888)
```

Now images are accessible via: `http://<host-ip>:8888/test/fire/image.jpg`

## Step 3: Engineer Structured Prompts for Explainability

The key to getting useful explanations from the model is **structured prompt engineering**. We designed a prompt that forces the model to:

1. **Start with a clear label** (FIRE or NO_FIRE)
2. **Follow with detailed reasoning**
3. **Consider multiple fire indicators** (flames, smoke, signs of burning)

```python
system_prompt = (
    "You are a wildfire detection expert. Analyze the image and provide "
    "a classification with reasoning.\n\n"
    "START your response with one of these labels on the first line:\n"
    "FIRE\n"
    "NO_FIRE\n\n"
    "Then on the following lines, explain your reasoning.\n\n"
    "Classification criteria:\n"
    "- Use FIRE if you see: active flames, burning, wildfire smoke "
    "(dense plumes from vegetation/forest), or signs of active fire spread\n"
    "- Use NO_FIRE for: clear landscapes, normal clouds, fog, steam, "
    "or images with no fire indicators\n\n"
    "Example format:\n"
    "FIRE\n"
    "The image shows dense smoke rising from forest vegetation, "
    "indicating an active wildfire...\n\n"
    "Remember: START with FIRE or NO_FIRE on the first line."
)
```

**Why this works**:
* **First-line label**: Easy to parse programmatically
* **Reasoning after label**: Model can elaborate without affecting parsing
* **Clear criteria**: Reduces ambiguity in classification
* **Examples**: Guide the model's response format

## Step 4: Parse Structured Responses

We extract both the classification label and the reasoning explanation:

```python
def parse_model_response(model_response: str) -> tuple:
    """
    Parse model response to extract label and explanation.
    Returns: (label_int, label_str, explanation)
    """
    text = model_response.strip()
    lines = text.split('\n', 1)  # Split at first newline
    
    first_line = lines[0].strip().upper()
    explanation = lines[1].strip() if len(lines) > 1 else ""
    
    # Check first line for label
    if "FIRE" in first_line and "NO_FIRE" not in first_line:
        return 1, "fire", explanation
    elif "NO_FIRE" in first_line or "NO FIRE" in first_line:
        return 0, "no_fire", explanation
    
    # Fallback to checking entire response
    return 0, "no_fire", text
```

**Result**: Each prediction includes:
* `parsed_label`: fire/no_fire (for metrics)
* `explanation`: Model's reasoning (for human review)
* `full_response`: Complete raw output (for debugging)

## Step 5: Generate Comprehensive Evaluation Metrics

We compute standard classification metrics to evaluate model performance:

```python
# Confusion Matrix
TP = True Positives   # Fire correctly identified as fire
FP = False Positives  # No-fire incorrectly identified as fire
FN = False Negatives  # Fire incorrectly identified as no-fire
TN = True Negatives   # No-fire correctly identified as no-fire

# Performance Metrics
Precision = TP / (TP + FP)  # How accurate are fire predictions?
Recall = TP / (TP + FN)     # How many fires did we catch?
F1-Score = 2 * (Precision × Recall) / (Precision + Recall)
Accuracy = (TP + TN) / Total
```

**CSV Output** includes all predictions with explanations:

| image_name | ground_truth | parsed_label | explanation |
|------------|--------------|--------------|-------------|
| wildfire_001.jpg | fire | fire | Dense smoke plumes rising from forest vegetation... |
| landscape_045.jpg | nofire | nofire | Clear mountain landscape with normal atmospheric conditions... |

## Step 6: Visualize Results

We generate a comprehensive 6-panel visualization dashboard:

### 1. Confusion Matrix Heatmap
Visual representation of TP, FP, FN, TN with color coding

### 2. Classification Metrics Bar Chart
Precision, Recall, F1-Score, and Accuracy with labeled values

### 3. Ground Truth vs Predictions
Side-by-side comparison of actual vs predicted class distributions

### 4. Confusion Matrix Components
Individual bars for TN, FP, FN, TP with counts

### 5. Per-Class Performance
Precision and Recall broken down by class (Fire/No-Fire)

### 6. Summary Statistics
Text panel with dataset info, metrics, and error rates

```python
# Generated automatically with matplotlib + seaborn
fig = plt.figure(figsize=(16, 12))
# ... 6 subplots with metrics visualizations
plt.savefig('evaluation_visualizations.png', dpi=300)
```

**Output**: High-resolution PNG dashboard saved to outputs directory.

## Step 7: Optimize with Subset Testing

For rapid iteration, we added **subset testing** to evaluate on a small sample before running the full dataset:

```python
# Quick testing: Use 10 images (5 fire + 5 nofire)
SUBSET_SIZE = 10  

# Full evaluation: Use entire dataset
SUBSET_SIZE = None
```

**Benefits**:
* **Fast iteration**: Test prompt changes in ~30 seconds
* **Debug quickly**: Identify issues before full run
* **Save compute**: Don't waste GPU time on broken code

**Typical workflow**:
1. Set `SUBSET_SIZE = 10` → Test prompt engineering
2. Fix issues, verify outputs
3. Set `SUBSET_SIZE = None` → Run full evaluation

## Results: Explainable Wildfire Detection

### Example Predictions

**True Positive (Fire Correctly Detected)**:
```
Image: wildfire_mountain_2024.jpg
Parsed Label: FIRE
Explanation: The image shows active flames consuming vegetation on a 
hillside with dense smoke plumes rising vertically. The orange glow and 
heat distortion indicate an active wildfire with rapid spread potential. 
Multiple fire fronts visible with charred areas in the foreground.
```

**True Negative (No-Fire Correctly Identified)**:
```
Image: foggy_valley_morning.jpg
Parsed Label: NO_FIRE
Explanation: The image shows a valley filled with morning fog, which 
appears as a white layer. Unlike wildfire smoke, this fog is evenly 
distributed and lacks the dark gray/black coloration typical of 
combustion byproducts. No heat signatures or flames visible.
```

**False Negative (Missed Fire - Room for Improvement)**:
```
Image: early_stage_fire_smoke.jpg
Parsed Label: NO_FIRE (Incorrect)
Ground Truth: FIRE
Explanation: The image shows light gray smoke that could be natural fog 
or mist. No visible flames detected.
Analysis: Model missed early-stage fire with only smoke present. 
Consider fine-tuning with more early-stage fire examples.
```

### Evaluation Metrics

On our test dataset of 410 images (159 fire, 251 no-fire):

```
==================================================
EVALUATION METRICS (Test Set)
==================================================

Confusion Matrix:
  TP (True Positives):  142
  FP (False Positives): 18
  FN (False Negatives): 17
  TN (True Negatives):  233

Classification Metrics:
  Precision: 0.8875
  Recall:    0.8931
  F1-score:  0.8903
  Accuracy:  0.9146
==================================================
```

**Key Insights**:
* **91.5% Accuracy**: Strong overall performance
* **88.8% Precision**: Most fire alerts are genuine
* **89.3% Recall**: Catches most actual fires
* **Explainability**: Every prediction includes reasoning

## When to Use This Approach

### ✅ Ideal Use Cases

* **Emergency Response Systems**: Where explanations are critical for decision-making
* **Regulatory Compliance**: Auditable AI decisions required
* **Model Development**: Understanding model behavior to improve prompts
* **Human-in-the-Loop**: Operators need context to validate alerts
* **Edge Deployment**: When you can't retrain models but can adjust prompts

## Beyond Wildfire: Adapting to Your Domain

The evaluation framework is **domain-agnostic**. Adapt it for:

### 🏥 Medical Imaging
```python
system_prompt = "Analyze this medical image. START with NORMAL or ABNORMAL..."
```

### 🏭 Industrial Inspection
```python
system_prompt = "Inspect this component. START with PASS or FAIL..."
```

### 🌾 Agriculture
```python
system_prompt = "Assess crop health. START with HEALTHY or DISEASED..."
```

### 🚗 Autonomous Driving
```python
system_prompt = "Analyze driving conditions. START with SAFE or HAZARD..."
```

**Key Pattern**: 
1. Define clear binary/multi-class labels
2. Specify domain criteria in prompt
3. Request structured output format
4. Parse label + explanation
5. Evaluate with standard metrics

## Code Walkthrough: Key Components

### 1. Configuration
```python
# API Configuration
API_URL = "http://0.0.0.0:8000/v1/chat/completions"
MODEL_NAME = "nvidia/nemotron-nano-12b-v2-vl"

# Data paths
DATA_ROOT = Path("data/wildfire_dataset/test")
FIRE_DIR = DATA_ROOT / "fire"
NOFIRE_DIR = DATA_ROOT / "nofire"

# Testing mode
SUBSET_SIZE = None  # None for full dataset, or number for quick testing
```

### 2. Build API Payload
```python
def build_payload(model: str, image_url: str) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "max_tokens": 128,
        "temperature": 0,
    }
```

### 3. Process Images
```python
for img_path in tqdm(fire_images, desc="Fire images"):
    # Convert to HTTP URL
    image_url = image_path_to_url(img_path, SERVER_ROOT, HOST_IP, PORT)
    
    # Call API
    response = session.post(API_URL, json=build_payload(MODEL_NAME, image_url))
    model_response = response.json()["choices"][0]["message"]["content"]
    
    # Parse response
    pred_label, label_str, explanation = parse_model_response(model_response)
    
    # Store results
    results.append({
        "image_name": img_path.name,
        "parsed_label": label_str,
        "explanation": explanation,
        "full_response": model_response,
    })
```

### 4. Generate Visualizations
```python
# Confusion matrix heatmap
cm = np.array([[TN, FP], [FN, TP]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Metrics bar chart
metrics = [precision, recall, f1, accuracy]
plt.bar(['Precision', 'Recall', 'F1', 'Accuracy'], metrics)

# Save high-res output
plt.savefig('evaluation_visualizations.png', dpi=300)
```

## Best Practices We Learned

### 1. Prompt Engineering is Critical
* **Bad**: "Classify this image"
* **Good**: "START with FIRE or NO_FIRE on line 1, then explain..."
* **Result**: 95%+ structured output compliance

### 2. HTTP Server Must Use Host IP
* **Wrong**: `localhost:8888` (container can't reach host)
* **Right**: `10.185.124.93:8888` (host machine IP)
* **Detection**: Auto-detect with `socket.getsockname()`

### 3. Parse Robustly
* Check first line for label
* Fall back to full text search if needed
* Always handle edge cases (errors, malformed responses)

### 4. Start Small, Scale Up
* Test on 10 images first
* Verify outputs manually
* Then run full 410-image evaluation

### 5. Export Everything
* CSV with all predictions
* JSON with metrics summary  
* PNG with visualizations
* Full responses for debugging

## Performance Benchmarks

### Hardware: NVIDIA A100 (80GB)

| Metric | Value |
|--------|-------|
| Images processed | 410 |
| Total time | ~15 minutes |
| Time per image | ~2.3 seconds |


## Troubleshooting Common Issues

### Issue 1: 400 Bad Request
**Symptom**: All predictions return ERROR  
**Cause**: Docker container can't access image URLs  
**Fix**: Use host IP, not localhost; verify HTTP server is running

### Issue 2: Port Already in Use
**Symptom**: HTTP server fails to start  
**Cause**: Previous server still running on port 8888  
**Fix**: Auto-detect available ports with `find_available_port()`

### Issue 3: Model Ignores Structured Format
**Symptom**: Responses don't start with FIRE/NO_FIRE  
**Cause**: Prompt not explicit enough  
**Fix**: Add "Remember: START with FIRE or NO_FIRE on line 1"


## Conclusion

We've built a complete end-to-end pipeline for **explainable wildfire detection** using NVIDIA Nemotron VLM. Key takeaways:

✅ **Vision-language models** provide both classification and reasoning  
✅ **Structured prompts** enable programmatic parsing + human interpretability   
✅ **Comprehensive evaluation** generates metrics, visualizations, and exportable results   

The complete Jupyter notebook provides a **reusable template** for evaluating VLMs on any vision classification task requiring explainability.

## Get Started

Clone the repository and run the notebook:

```bash
# Clone repository
git clone <your-repo-url>
cd wildfire-nemo

# Install dependencies
pip install requests numpy pandas tqdm matplotlib seaborn

# Start Nemotron NIM container (see prerequisites)
docker run -it --rm --gpus all ...

# Launch Jupyter
jupyter notebook evaluate_wildfire_vlm.ipynb
```

All code is open-source and ready to adapt for your domain!

## Resources

* **Notebook**: [evaluate_wildfire_vlm.ipynb](link-to-notebook)
* **NVIDIA NIM**: [build.nvidia.com](https://build.nvidia.com/)
* **Nemotron Models**: [Nemotron Developer Page](https://developer.nvidia.com/nemotron)
* **Dataset**: [Wildfire Dataset 2N](link-to-dataset)

## References

* El-Madafri I, Peña M, Olmedo-Torre N. The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach. Forests. 2023; 14(9):1697. https://doi.org/10.3390/f14091697


***Last updated: 11-29-2025***

