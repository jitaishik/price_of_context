import time

import openai
import requests

VLLM_BASE_URL = "http://localhost:8000/v1"

MODEL_PATHS = {
    "llama": "llama_model_path",
    "qwen": "qwen_model_path",
}


def get_vllm_client_for_model(model_name, base_url=VLLM_BASE_URL):
    """Return (model_url, client) for a locally vLLM-served "llama"/"qwen" model.

    `model_url` is None for unrecognized model names (e.g. "gpt", which is
    served by the real OpenAI API instead - see `configure_gpt`).
    """
    client = openai.OpenAI(base_url=base_url, api_key="dummy-key")
    return MODEL_PATHS.get(model_name), client


def get_vllm_client(base_url=VLLM_BASE_URL):
    """Generic OpenAI-compatible client pointed at a vLLM server."""
    return openai.OpenAI(base_url=base_url, api_key="dummy-key")


def configure_gpt(api_key=None):
    """Set the `openai` module's API key for direct (non-vLLM) GPT calls."""
    openai.api_key = api_key or "open_ai-key"


def call_vllm_text(base_url, model, messages, max_tokens=512, temperature=0.7, retries=3):
    """POST a chat completion directly to a vLLM server and return the raw
    text content, retrying with exponential backoff on failure."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"vLLM call failed after {retries} attempts: {e}") from e
