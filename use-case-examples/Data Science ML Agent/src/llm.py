# llm.py - Simplified for beginners
import os
from typing import List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("NIM_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
TEMPERATURE = float(os.getenv("NIM_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("NIM_MAX_TOKENS", "4096"))

# Set up HTTP session with retries
def create_session():
    """Create a session with retry logic for better reliability."""
    session = requests.Session()
    
    # Retry Strategy for Resilient Communication
    retry_strategy = Retry(
        total=3,  # Try 3 times
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these errors
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

class LLMClient:
    """Simple client to talk to AI models."""
    
    def __init__(self):
        # Get API key from environment
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("Please set NVIDIA_API_KEY environment variable")
        
        self.base_url = BASE_URL.rstrip("/")
        self.model = MODEL
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.session = create_session()
 
    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        """Send messages to the AI and get a response."""
        
        # Prepare the request
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add tools if provided
        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = "auto"
        
        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Make the request
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_data,
                timeout=(10, 300)  # 10 seconds to connect, 300 seconds to read
            )
            
            # Check if request was successful
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with AI: {e}")

def create_client() -> LLMClient:
    """Create a new LLM client."""
    return LLMClient()