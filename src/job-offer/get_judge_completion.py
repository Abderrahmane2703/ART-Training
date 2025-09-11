from async_lru import alru_cache
import asyncio
from openai import AsyncAzureOpenAI
from openai import RateLimitError
import os
from dotenv import load_dotenv
import random

load_dotenv()

# Configurable concurrency and retry/backoff
MAX_CONCURRENCY = int(os.getenv("JUDGE_MAX_CONCURRENCY", "3"))
BASE_BACKOFF = float(os.getenv("JUDGE_BACKOFF_BASE", "2"))  # seconds
MAX_BACKOFF = float(os.getenv("JUDGE_BACKOFF_MAX", "30"))    # seconds
MAX_RETRIES = int(os.getenv("JUDGE_RETRIES", "6"))

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
    prompt, temperature=0.0, max_tokens=600, retries: int | None = None, timeout=30
) -> str:
    total_retries = retries if retries is not None else MAX_RETRIES
    for attempt in range(1, total_retries + 1):
        try:
            async with semaphore:
                completion = await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-35-turbo"),  # Your Azure deployment name
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            # Determine backoff duration
            retry_after: float | None = None
            # Try to inspect Retry-After header if present (not always available)
            try:
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    ra = e.response.headers.get("retry-after") or e.response.headers.get("Retry-After")
                    if ra is not None:
                        retry_after = float(ra)
            except Exception:
                pass

            # Fallback to exponential backoff with jitter
            if retry_after is None:
                retry_after = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** (attempt - 1)))
                retry_after += random.uniform(0, 0.5)

            if attempt < total_retries:
                print(
                    f"[Retry {attempt}/{total_retries}] get_judge_completion failed: {e}. "
                    f"Retrying in {retry_after:.1f}s..."
                )
                await asyncio.sleep(retry_after)
            else:
                print(
                    f"[Failure] get_judge_completion failed after {total_retries} attempts: {e}"
                )
                return "ERROR: Get judge completion failed"


def clear_judge_cache():
    """Clear the cache for get_judge_completion."""
    get_judge_completion.cache_clear()
    print("Judge cache cleared")
