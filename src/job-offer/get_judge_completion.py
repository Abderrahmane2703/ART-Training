from async_lru import alru_cache
import asyncio
from openai import AsyncAzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

semaphore = asyncio.Semaphore(20)

# Azure OpenAI configuration
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g., "https://your-resource.openai.azure.com/"
    api_version="2024-02-01",  # Azure API version
)


@alru_cache(maxsize=1024)
async def get_judge_completion(
    prompt, temperature=0.0, max_tokens=600, retries=3, timeout=10
) -> str:
    for attempt in range(1, retries + 1):
        try:
            async with semaphore:
                completion = await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-35-turbo"),  # Your Azure deployment name
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries:
                print(
                    f"[Retry {attempt}/{retries}] get_judge_completion failed: {e}. Retrying..."
                )
                await asyncio.sleep(3)
            else:
                print(
                    f"[Failure] get_judge_completion failed after {retries} attempts: {e}"
                )
                return "ERROR: Get judge completion failed"


def clear_judge_cache():
    """Clear the cache for get_judge_completion."""
    get_judge_completion.cache_clear()
    print("Judge cache cleared")
