import os
import time
import json
import math
import random
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI

# --------------------------
# Config
# --------------------------
API_VERSION = "2024-02-01"
# Sequential test to avoid 429s; keep it 1
SLEEP_BETWEEN_CALLS_SEC = 1.0

# Backoff for transient errors (e.g., 429)
BASE_BACKOFF = 2.0
MAX_BACKOFF = 30.0
MAX_RETRIES = 6
TIMEOUT_SEC = 30

# --------------------------
# Sample inputs (edit freely)
# --------------------------
JOB_TITLE = "Senior Database Engineer"
LANGUAGE = "en"
PROVIDED_SKILLS = ["SQL", "PostgreSQL", "ETL", "Python"]
GENERATED_OFFER = """<job_offer>
  <title>Senior Database Engineer</title>
  <overview>We are seeking a Senior Database Engineer to design and optimize large-scale data systems.</overview>
  <responsibilities>
    <item>Design database schemas and implement performant queries</item>
    <item>Optimize existing pipelines for reliability and throughput</item>
    <item>Collaborate with backend teams on data contracts</item>
    <item>Implement monitoring and alerting for data infrastructure</item>
  </responsibilities>
  <skills>
    <item>SQL</item>
    <item>ETL</item>
    <item>Python</item>
    <item>Data Modeling</item>
    <item>Performance Tuning</item>
  </skills>
  <nice_to_have>
    <item>AWS</item>
    <item>Kafka</item>
  </nice_to_have>
</job_offer>
"""

# --------------------------
# Prompt builders (mirrors your rollout prompts roughly)
# --------------------------
def language_prompt(generated_offer: str, lang: str) -> str:
    text_snippet = generated_offer[:300]
    readable = "English" if lang == "en" else "French" if lang == "fr" else lang
    return f"""Is this text written in {lang.upper()} language?

Text to check: {text_snippet}

Expected language: {lang} ({readable})

Respond ONLY in JSON format:
{{"answer": "YES" or "NO"}}"""

def xml_prompt(generated_offer: str) -> str:
    return f"""Is this valid XML format? Check if it has proper opening/closing tags.

Text to check:
{generated_offer}

Respond ONLY in JSON format:
{{"valid_xml": true or false, "has_required_tags": true or false}}"""

def context_inclusion_prompt(generated_offer: str, required_skills: list[str]) -> str:
    return f"""Evaluate skill inclusion with deduplication penalty.

Required skills that MUST be included: {', '.join(required_skills)}

Generated job offer:
{generated_offer}

Instructions:
1. Extract ALL skills mentioned in the job offer (from all sections)
2. For each required skill, check if it's present (exact or fuzzy match)
3. Identify duplicate/redundant skills
4. Calculate base_score = matched_required_skills / total_required_skills
5. Calculate deduplication_factor = 1 - (duplicate_count / total_extracted_skills)
6. Calculate final_score = base_score * deduplication_factor

Respond ONLY in JSON format:
{{
  "required_skills": ["skill1", "skill2"],
  "extracted_skills": ["skill1", "skill2", "..."],
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": [],
  "duplicate_groups": [["Python","Python Programming"]],
  "duplicate_count": 0,
  "base_score": 0.0,
  "deduplication_factor": 1.0,
  "final_score": 0.0
}}"""

def skill_relevance_prompt(generated_offer: str, job_title: str, provided_skills: list[str]) -> str:
    provided_str = ", ".join(provided_skills) if provided_skills else "None"
    return f"""For a {job_title} position:

1. Extract ALL skills mentioned in the job offer (both in Required Skills and Nice-to-Have sections)
2. EXCLUDE these provided skills from evaluation: {provided_str}
3. For each NEW skill added by the model, score it:
   - 1 if relevant to {job_title}
   - 0 if not relevant
4. Calculate final score = sum(skill_scores) / total_new_skills

Generated job offer:
{generated_offer}

Respond ONLY in JSON format:
{{
  "provided_skills": ["skill1","skill2"],
  "all_extracted_skills": ["skill1","skill2","..."],
  "new_skills_evaluation": [{{"skill":"skill_name","relevant":1}}],
  "total_new_skills": 0,
  "relevant_count": 0,
  "final_score": 0.0
}}"""

def completeness_prompt(generated_offer: str, job_title: str) -> str:
    return f"""Evaluate skill completeness for {job_title} position.

Review the generated job offer and identify essential skills that are missing.
Essential = skills that 80%+ of {job_title} job postings would include.

Generated job offer:
{generated_offer}

For each missing essential skill:
1. Explain why it's essential for this role
2. Rate its importance: 1.0 (critical), 0.5 (important), 0.25 (nice-to-have)

Calculate penalty = sum(importance_scores) / 10 (capped at 1.0)
Final score = 1.0 - penalty

Respond ONLY in JSON format:
{{
  "skills_present": ["skill1","skill2"],
  "missing_essentials": [{{"skill":"Git","importance":1.0,"reason":"Version control is critical"}}],
  "total_penalty": 0.0,
  "final_score": 1.0
}}"""

PROMPTS = {
    "language_consistency": (language_prompt(GENERATED_OFFER, LANGUAGE), 50),
    "xml_format": (xml_prompt(GENERATED_OFFER), 100),
    "context_inclusion": (context_inclusion_prompt(GENERATED_OFFER, PROVIDED_SKILLS), 400),
    "skill_relevance": (skill_relevance_prompt(GENERATED_OFFER, JOB_TITLE, PROVIDED_SKILLS), 300),
    "skill_completeness": (completeness_prompt(GENERATED_OFFER, JOB_TITLE), 400),
}

# --------------------------
# API helper
# -- supports both legacy 'max_tokens' and newer 'max_completion_tokens' variants.
def _create_chat_with_fallback(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
):
    """Attempt a chat.completions call with smart fallbacks:
    - Try 'max_tokens' first, then switch to 'max_completion_tokens' if unsupported.
    - If 'temperature' value is unsupported, drop it to use the model's default.
    Also handles the inverse case where 'max_completion_tokens' is rejected.
    """
    # Build kwargs dynamically so we can remove/replace parameters on the fly
    kwargs: dict[str, object] = {
        "model": deployment,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": timeout,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    # Start by trying legacy 'max_tokens'
    kwargs["max_tokens"] = max_tokens

    # If the prompt explicitly asks for JSON, request structured JSON output.
    # Some models (e.g., GPT-5 mini/nano) are more reliable with response_format.
    expects_json = "Respond ONLY in JSON format" in prompt
    if expects_json:
        kwargs["response_format"] = {"type": "json_object"}

    tried_completion_tokens = False
    tried_drop_temperature = False

    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            # Parse Azure error payload if present
            err_json = None
            try:
                resp = getattr(e, "response", None)
                err_json = resp.json() if resp is not None else None
            except Exception:
                err_json = None

            # Extract fields
            code = None
            param = None
            message = None
            if isinstance(err_json, dict):
                err_obj = err_json.get("error") or {}
                code = str(err_obj.get("code", ""))
                param = str(err_obj.get("param", "")) if err_obj.get("param") is not None else None
                message = str(err_obj.get("message", ""))

            # Fallback for unsupported max_tokens
            if (
                (code == "unsupported_parameter" or (message and "Unsupported parameter" in message))
                and param == "max_tokens"
                and not tried_completion_tokens
            ):
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = max_tokens
                tried_completion_tokens = True
                continue

            # Fallback for unsupported temperature value
            if (
                (code == "unsupported_value" or (message and "Unsupported value" in message))
                and param == "temperature"
                and not tried_drop_temperature
            ):
                # Drop temperature to let the model use its default (often required on mini/nano)
                kwargs.pop("temperature", None)
                tried_drop_temperature = True
                continue

            # Handle inverse: if 'max_completion_tokens' is rejected on older deployments
            if (
                (code == "unsupported_parameter" or (message and "Unsupported parameter" in message))
                and param == "max_completion_tokens"
                and "max_completion_tokens" in kwargs
            ):
                kwargs.pop("max_completion_tokens", None)
                kwargs["max_tokens"] = max_tokens
                continue

            # If response_format is not recognized/unsupported, remove it and retry
            if (
                (code == "unsupported_parameter" or (message and ("Unrecognized request argument" in message or "Unsupported parameter" in message)))
                and (param == "response_format" or "response_format" in (message or ""))
                and "response_format" in kwargs
            ):
                kwargs.pop("response_format", None)
                continue

            # No known fallback available; re-raise to outer backoff
            raise

# --------------------------
def _extract_text_from_response(resp) -> str:
    """Extract textual content from Azure Chat Completions response across
    different model variants (reasoning, mini/nano, multimodal).
    """
    try:
        choice = resp.choices[0]
        msg = getattr(choice, "message", None)
        if msg is None:
            txt = getattr(resp, "output_text", None)
            return (txt or "").strip()

        content = getattr(msg, "content", None)
        # Standard string content
        if isinstance(content, str):
            return content.strip()
        # Some SDK/model combos return list of content parts
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                    if t:
                        parts.append(str(t))
                else:
                    t = getattr(part, "text", None)
                    if t:
                        parts.append(str(t))
            if parts:
                return "\n".join(parts).strip()

        # Fallback: if there are tool/function call arguments
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            buf = []
            for tc in tool_calls:
                try:
                    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                    if fn:
                        args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
                        if args:
                            buf.append(str(args))
                except Exception:
                    continue
            if buf:
                return "\n".join(buf).strip()

        # Last resort: top-level output_text if present
        txt = getattr(resp, "output_text", None)
        return (txt or "").strip()
    except Exception:
        try:
            txt = getattr(resp, "output_text", None)
            return (txt or "").strip()
        except Exception:
            return ""

# --------------------------
def _responses_api_call(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    expects_json: bool,
):
    """Call Azure Responses API as a fallback and return (content, usage_meta)."""
    kwargs: dict[str, object] = {
        "model": deployment,
        "input": [{"role": "user", "content": prompt}],
        "timeout": timeout,
        # Responses API uses max_output_tokens
        "max_output_tokens": max_tokens,
    }
    if expects_json:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.responses.create(**kwargs)
    # Prefer output_text if available
    text = getattr(resp, "output_text", None)
    if text:
        text = text.strip()
    else:
        # Try to assemble from output contents
        try:
            output = getattr(resp, "output", None)
            parts = []
            if isinstance(output, list):
                for item in output:
                    content_list = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                    if isinstance(content_list, list):
                        for part in content_list:
                            t = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                            if t:
                                parts.append(str(t))
            text = "\n".join(parts).strip() if parts else ""
        except Exception:
            text = ""

    usage = getattr(resp, "usage", None)
    meta = {
        "prompt_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }
    return text or "", meta

# --------------------------
def chat_with_backoff(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    retries: int = MAX_RETRIES,
    base_backoff: float = BASE_BACKOFF,
    max_backoff: float = MAX_BACKOFF,
    timeout: int = TIMEOUT_SEC,
) -> Tuple[str, Dict[str, Any]]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.perf_counter()
            resp = _create_chat_with_fallback(
                client=client,
                deployment=deployment,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            t1 = time.perf_counter()
            content = _extract_text_from_response(resp)
            usage = getattr(resp, "usage", None)
            meta = {
                "latency_sec": round(t1 - t0, 3),
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            }
            # If content is empty but usage suggests tokens were generated, try Responses API fallback once
            if (not content) and (meta["completion_tokens"] and meta["completion_tokens"] > 0):
                expects_json = "Respond ONLY in JSON format" in prompt
                try:
                    content2, meta2 = _responses_api_call(
                        client=client,
                        deployment=deployment,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        expects_json=expects_json,
                    )
                    # If Responses API returns something, prefer it
                    if content2:
                        # Merge usage meta if available
                        for k, v in meta2.items():
                            if v is not None:
                                meta[k] = v
                        return content2.strip(), meta
                except Exception:
                    pass

            return (content or "").strip(), meta
        except Exception as e:
            last_err = e
            # Retry-After if available
            retry_after = None
            try:
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    ra = e.response.headers.get("retry-after") or e.response.headers.get("Retry-After")
                    if ra:
                        retry_after = float(ra)
            except Exception:
                pass
            # Exponential backoff with jitter
            if retry_after is None:
                backoff = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                backoff += random.uniform(0, 0.5)
            else:
                backoff = retry_after
            print(f"[{deployment}] attempt {attempt}/{retries} failed: {e}. Sleeping {backoff:.1f}s...")
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {retries} attempts. Last error: {last_err}")

# --------------------------
# Main
# --------------------------
def main():
    load_dotenv()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise SystemExit("Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env")

    deployments = {
        "gpt5": os.getenv("AZURE_DEPLOYMENT_NAME_GPT5"),
        "gpt5-mini": os.getenv("AZURE_DEPLOYMENT_NAME_GPT5_MINI"),
        "gpt5-nano": os.getenv("AZURE_DEPLOYMENT_NAME_GPT5_NANO"),
    }
    deployments = {k: v for k, v in deployments.items() if v}

    if not deployments:
        raise SystemExit("Please set at least one of AZURE_DEPLOYMENT_NAME_GPT5(_MINI/_NANO) in .env")

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=API_VERSION,
        max_retries=0,  # we handle retries ourselves
    )

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model_name, deployment in deployments.items():
        results[model_name] = {}
        print(f"\n==== Testing model: {model_name} (deployment={deployment}) ====")
        for test_name, (prompt, max_tokens) in PROMPTS.items():
            print(f"\n-- {test_name} --")
            try:
                content, meta = chat_with_backoff(
                    client, deployment, prompt, max_tokens=max_tokens, temperature=0.0
                )
                print(f"Latency: {meta['latency_sec']}s | tokens: {meta['total_tokens']}")
                # Print first 400 chars to keep output readable
                excerpt = content[:400].replace("\n", " ")
                print(f"Output excerpt: {excerpt}{'…' if len(content) > 400 else ''}")
                results[model_name][test_name] = {"ok": True, "meta": meta, "output": content}
            except Exception as e:
                print(f"ERROR: {e}")
                results[model_name][test_name] = {"ok": False, "error": str(e)}

            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    print("\n==== Summary (JSON) ====")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()