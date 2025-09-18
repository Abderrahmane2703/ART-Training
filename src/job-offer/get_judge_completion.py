from async_lru import alru_cache
import asyncio
from openai import AsyncAzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Configurable concurrency
MAX_CONCURRENCY = int(os.getenv("JUDGE_MAX_CONCURRENCY", "3"))

semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# Azure OpenAI configuration
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g., "https://your-resource.openai.azure.com/"
    api_version="2024-02-01",  # Azure API version
    # Disable client-level retries so we own the backoff behavior here
    max_retries=0,
)


@alru_cache(maxsize=1024)
async def get_judge_completion(
    prompt, temperature=0.0, max_tokens=600, timeout=30
) -> str:
    try:
        async with semaphore:
            completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=os.getenv("AZURE_DEPLOYMENT_NAME_GPT5", "gpt-35-turbo"),  # Your Azure deployment name
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Failure] get_judge_completion failed: {e}")
        return "ERROR: Get judge completion failed"


def clear_judge_cache():
    """Clear the cache for get_judge_completion."""
    get_judge_completion.cache_clear()


# -----------------------------
# GPT-5 Mini/Nano (Azure) helpers
# -----------------------------

# Separate client for GPT-5 reasoning models (newer API version)
client_gpt5_reasoning = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    # Disable client-level retries so we own the behavior
    max_retries=0,
)


@alru_cache(maxsize=1024)
async def get_judge_completion_gpt5_mini(
    prompt, max_completion_tokens=600, timeout=30
) -> str:
    """Judge using Azure GPT-5 Mini deployment.
    Honors env AZURE_DEPLOYMENT_NAME_GPT5_MINI, falls back to 'gpt-5-mini'.
    Mirrors logic from get_judge_completion_gpt5_mini.py.
    """
    try:
        async with semaphore:
            deployment = os.getenv("AZURE_DEPLOYMENT_NAME_GPT5_MINI", "gpt-5-mini")
            completion = await client_gpt5_reasoning.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise evaluator. Provide only the final visible answer. "
                            "Do not include internal reasoning. Keep outputs concise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=deployment,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"},
                timeout=timeout,
            )
        content = completion.choices[0].message.content or ""
        return content.strip()
    except Exception as e:
        print(f"[Failure] get_judge_completion_gpt5_mini failed: {e}")
        return "ERROR: Get judge completion (gpt5-mini) failed"


def clear_judge_cache_gpt5_mini():
    """Clear the cache for get_judge_completion_gpt5_mini."""
    get_judge_completion_gpt5_mini.cache_clear()


@alru_cache(maxsize=1024)
async def get_judge_completion_gpt5_nano(
    prompt, max_completion_tokens=600, timeout=30
) -> str:
    """Judge using Azure GPT-5 Nano deployment.
    Honors env AZURE_DEPLOYMENT_NAME_GPT5_NANO, falls back to 'gpt-5-nano'.
    Behavior mirrors GPT-5 Mini helper but targets nano.
    """
    try:
        async with semaphore:
            deployment = os.getenv("AZURE_DEPLOYMENT_NAME_GPT5_NANO", "gpt-5-nano")
            completion = await client_gpt5_reasoning.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise evaluator. Provide only the final visible answer. "
                            "Do not include internal reasoning. Keep outputs concise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=deployment,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "text"},
                timeout=timeout,
            )
        content = completion.choices[0].message.content or ""
        return content.strip()
    except Exception as e:
        print(f"[Failure] get_judge_completion_gpt5_nano failed: {e}")
        return "ERROR: Get judge completion (gpt5-nano) failed"


def clear_judge_cache_gpt5_nano():
    """Clear the cache for get_judge_completion_gpt5_nano."""
    get_judge_completion_gpt5_nano.cache_clear()


@alru_cache(maxsize=1024)
async def get_judge_completion_gpt5_strict(
    prompt, max_completion_tokens=800, timeout=30
) -> str:
    """Judge using Azure GPT-5 deployment with strict JSON output.
    Uses response_format={"type":"json_object"}.
    Honors env AZURE_DEPLOYMENT_NAME_GPT5, falls back to 'gpt-5'.
    """
    try:
        async with semaphore:
            deployment = os.getenv("AZURE_DEPLOYMENT_NAME_GPT5", "gpt-5")
            completion = await client_gpt5_reasoning.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You MUST return ONLY a strict JSON object. "
                            "Use double quotes for all keys and strings. "
                            "Do not include any text before or after the JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=deployment,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"},
                timeout=timeout,
            )
        content = completion.choices[0].message.content or ""
        return content.strip()
    except Exception as e:
        print(f"[Failure] get_judge_completion_gpt5_strict failed: {e}")
        return "ERROR: Get judge completion (gpt5-strict) failed"


def clear_judge_cache_gpt5_strict():
    """Clear the cache for get_judge_completion_gpt5_strict."""
    get_judge_completion_gpt5_strict.cache_clear()
