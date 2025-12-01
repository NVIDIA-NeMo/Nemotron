# Build a Computer Use Agent with Synthetic Data

This project demonstrates how to build a computer use agent using synthetic data generated with NVIDIA's NeMo Data Designer.

## Setting Up Data Designer for Local Development

Before running the synthetic data generation notebook, you need to set up NeMo Data Designer locally using Docker.

### Prerequisites

- Docker and Docker Compose installed
- NGC CLI installed and configured
- NGC API key set in environment variable `NGC_CLI_API_KEY`

### Setup Steps

1. **Authenticate with NGC**:
   ```bash
   echo $NGC_CLI_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

2. **Download NeMo Data Designer Docker Compose files**:
   ```bash
   ngc registry resource download-version "nvidia/nemo-microservices/nemo-microservices-quickstart:25.11"
   cd nemo-microservices-quickstart_v25.11
   ```

3. **Set important environment variables**
   ```bash
   export NEMO_MICROSERVICES_IMAGE_REGISTRY="nvcr.io/nvidia/nemo-microservices"
   export NEMO_MICROSERVICES_IMAGE_TAG="25.11"
   export NIM_API_KEY=$NGC_CLI_API_KEY # This is the API key for build.nvidia.com created as part of the prerequisites
   ```

3. **Start the Data Designer services**:
   ```bash
   docker compose --profile data-designer up
   ```

   The Data Designer service will be available at `http://localhost:8000`.

4. **Verify the service is running**:
   ```bash
   curl http://localhost:8080/v1/nemo/dd/health
   ```

## Usage

Once Data Designer is running, you can use the Jupyter notebook `langgraph_cli_synthetic_data.ipynb` to generate synthetic datasets for training your computer use agent.

