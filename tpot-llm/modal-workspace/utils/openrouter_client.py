# %%
import os
import time
from typing import List, Dict, Any, Union, Tuple
from openai import OpenAI, OpenAIError
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

# Add these type variables for our decorator
T = TypeVar("T")
P = ParamSpec("P")


def with_retries(
    max_attempts: int = 3, base_delay: float = 1.0
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that adds exponential backoff retries to a function

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries (will be exponentially increased)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt == max_attempts:
                        raise e
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    time.sleep(delay)
            raise RuntimeError("Should not reach here")

        return wrapper

    return decorator


def get_openrouter_client():
    """Get OpenRouter client with proper configuration"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://community-archive.org",
            "X-Title": "community-archive",
        },
    )


def handle_llm_error(e: Exception) -> Tuple[str, str]:
    """Handle LLM API errors and return error details

    Args:
        e: Exception that occurred

    Returns:
        Tuple of (error type, error message)
    """
    error_type = type(e).__name__
    if isinstance(e, OpenAIError):
        error_msg = str(e)
    else:
        error_msg = f"Unexpected error: {str(e)}"
    return error_msg


@with_retries(max_attempts=3)
def query_llm(
    message: str,
    model: str = "meta-llama/llama-3.3-70b-instruct",
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> Union[str, Tuple[str, str]]:
    """Query LLM through OpenRouter

    Args:
        message: Prompt to send
        model: Model to use
        max_tokens: Max tokens in response
        temperature: Sampling temperature

    Returns:
        Either model response text or tuple of (error_type, error_message)
    """
    try:
        client = get_openrouter_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return handle_llm_error(e)


@with_retries(max_attempts=3)
def chat_w_llm(
    messages: List[Dict[str, str]],
    system_prompt: str,
    model: str = "meta-llama/llama-3.3-70b-instruct",
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> Union[str, Tuple[str, str]]:
    """Chat with LLM using message history

    Args:
        messages: List of message dicts with 'role' and 'content'
        system_prompt: System prompt to prepend
        model: Model to use
        max_tokens: Max tokens in response
        temperature: Sampling temperature

    Returns:
        Either model response text or tuple of (error_type, error_message)
    """
    try:
        client = get_openrouter_client()
        formatted_messages = [
            {
                "role": m["role"],
                "content": (
                    f"{system_prompt}\n\n{m['content']}"
                    if m["role"] == "user" and messages.index(m) == 0
                    else m["content"]
                ),
            }
            for m in messages
        ]

        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        print(f"Model response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        return handle_llm_error(e)


@with_retries(max_attempts=3)
def get_available_models() -> List[Dict[str, Any]]:
    """Fetch available models from OpenRouter API

    Returns:
        List of model info dicts with fields like id, name, description
    """
    try:
        import requests

        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()

        models_data = response.json()
        models = []

        # Extract models from data array
        for model_info in models_data.get("data", []):
            models.append(
                {
                    "id": model_info.get("id"),
                    "name": model_info.get("name"),
                    "description": model_info.get("description", ""),
                    "context_length": model_info.get("context_length", 0),
                    "pricing": model_info.get("pricing", {}),
                }
            )

        return sorted(models, key=lambda x: x["name"])

    except Exception as e:
        print(f"Error fetching models: {e}")
        return []
