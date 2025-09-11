import os
import time
import json
import random
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI

# --------------------------
# Config
# --------------------------
API_VERSION = "2024-02-01"
SLEEP_BETWEEN_CALLS_SEC = 0.5
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
    return f"""Return ONLY a valid JSON object. Do not include code fences, markdown, or extra text.

Is this text written in {lang.upper()} language?

Text to check: {text_snippet}

Expected language: {lang} ({readable})

Respond ONLY in JSON format:
{{"answer": "YES" or "NO"}}"""

def xml_prompt(generated_offer: str) -> str:
    return f"""Return ONLY a valid JSON object. Do not include code fences, markdown, or extra text.

Is this valid XML format? Check if it has proper opening/closing tags.

Text to check:
{generated_offer}

Respond ONLY in JSON format:
{{"valid_xml": true or false, "has_required_tags": true or false}}"""

def context_inclusion_prompt(generated_offer: str, required_skills: list[str]) -> str:
    return f"""Return ONLY a valid JSON object. Do not include code fences, markdown, or extra text.

Evaluate skill inclusion with deduplication penalty.

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
    return f"""Return ONLY a valid JSON object. Do not include code fences, markdown, or extra text.

For a {job_title} position:

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
    return f"""Return ONLY a valid JSON object. Do not include code fences, markdown, or extra text.

Evaluate skill completeness for {job_title} position.

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
# Low-level helpers (Azure param compatibility)
# --------------------------

def _create_chat_with_fallback(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    # Use None so we don't send temperature unless required; some mini/nano only allow default
    temperature: float | None,
    max_tokens: int,
    timeout: int,
):
    """Attempt Chat Completions with smart fallbacks:
    - Start with max_tokens; switch to max_completion_tokens if unsupported.
    - Drop temperature if unsupported (use model default).
    - Remove response_format if rejected.
    """
    kwargs: Dict[str, Any] = {
        "model": deployment,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": timeout,
        # Avoid tool call-only responses; force text content when possible
        "tool_choice": "none",
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    kwargs["max_tokens"] = max_tokens

    expects_json = "Respond ONLY in JSON format" in prompt
    if expects_json:
        kwargs["response_format"] = {"type": "json_object"}

    tried_completion_tokens = False
    tried_drop_temperature = False

    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err_json = None
            try:
                resp = getattr(e, "response", None)
                err_json = resp.json() if resp is not None else None
            except Exception:
                err_json = None

            code = None
            param = None
            message = None
            if isinstance(err_json, dict):
                err_obj = err_json.get("error") or {}
                code = str(err_obj.get("code", ""))
                param = str(err_obj.get("param", "")) if err_obj.get("param") is not None else None
                message = str(err_obj.get("message", ""))

            # Swap max_tokens -> max_completion_tokens
            if (
                (code == "unsupported_parameter" or (message and "Unsupported parameter" in message))
                and param == "max_tokens"
                and not tried_completion_tokens
            ):
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = max_tokens
                tried_completion_tokens = True
                continue

            # Drop temperature if unsupported
            if (
                (code == "unsupported_value" or (message and "Unsupported value" in message))
                and param == "temperature"
                and not tried_drop_temperature
            ):
                kwargs.pop("temperature", None)
                tried_drop_temperature = True
                continue

            # Inverse: remove max_completion_tokens on older deployments
            if (
                (code == "unsupported_parameter" or (message and "Unsupported parameter" in message))
                and param == "max_completion_tokens"
                and "max_completion_tokens" in kwargs
            ):
                kwargs.pop("max_completion_tokens", None)
                kwargs["max_tokens"] = max_tokens
                continue

            # response_format unsupported
            if (
                (code == "unsupported_parameter" or (message and ("Unrecognized request argument" in message or "Unsupported parameter" in message)))
                and (param == "response_format" or "response_format" in (message or ""))
                and "response_format" in kwargs
            ):
                kwargs.pop("response_format", None)
                continue

            # Give up to caller
            raise


def _extract_text_from_response(resp) -> str:
    try:
        choice = resp.choices[0]
        msg = getattr(choice, "message", None)
        if msg is None:
            txt = getattr(resp, "output_text", None)
            return (txt or "").strip()

        content = getattr(msg, "content", None)
        # Structured outputs: some models set message.parsed when response_format is JSON
        parsed = getattr(msg, "parsed", None)
        if parsed is not None:
            try:
                import json as _json
                return _json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                else:
                    t = getattr(part, "text", None)
                if t:
                    parts.append(str(t))
            if parts:
                return "\n".join(parts).strip()

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            buf: list[str] = []
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

        txt = getattr(resp, "output_text", None)
        return (txt or "").strip()
    except Exception:
        try:
            txt = getattr(resp, "output_text", None)
            return (txt or "").strip()
        except Exception:
            return ""


def _responses_api_call(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    expects_json: bool,
):
    """Fallback to Responses API and extract output text."""
    kwargs: Dict[str, Any] = {
        "model": deployment,
        "input": [{"role": "user", "content": prompt}],
        "timeout": timeout,
        "max_output_tokens": max_tokens,
    }
    if expects_json:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.responses.create(**kwargs)

    text = getattr(resp, "output_text", None)
    if not text:
        # Try assembling from output parts
        parts: list[str] = []
        try:
            output = getattr(resp, "output", None)
            if isinstance(output, list):
                for item in output:
                    content_list = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                    if isinstance(content_list, list):
                        for part in content_list:
                            t = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                            if t:
                                parts.append(str(t))
        except Exception:
            pass
        text = "\n".join(parts) if parts else ""

    usage = getattr(resp, "usage", None)
    meta = {
        "prompt_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }
    return (text or "").strip(), meta


def chat_with_backoff(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None = None,
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
            # If empty content but tokens generated, try Responses API
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
                    if content2:
                        for k, v in meta2.items():
                            if v is not None:
                                meta[k] = v
                        return content2.strip(), meta
                except Exception:
                    pass

            return (content or "").strip(), meta
        except Exception as e:
            last_err = e
            # Retry-After header if any
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
    deployment = os.getenv("AZURE_DEPLOYMENT_NAME_GPT5_MINI")

    if not endpoint or not api_key:
        raise SystemExit("Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env")
    if not deployment:
        raise SystemExit("Please set AZURE_DEPLOYMENT_NAME_GPT5_MINI in your .env")

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=API_VERSION,
        max_retries=0,  # we handle retries ourselves
    )

    print(f"\n==== Testing model: gpt5-mini (deployment={deployment}) ====\n")
    results: Dict[str, Dict[str, Any]] = {}

    for test_name, (prompt, max_tokens) in PROMPTS.items():
        print(f"\n-- {test_name} --")
        try:
            # For mini, omit temperature to use model default
            content, meta = chat_with_backoff(
                client, deployment, prompt, max_tokens=max_tokens, temperature=None
            )
            print(f"Latency: {meta['latency_sec']}s | tokens: {meta['total_tokens']}")
            excerpt = content[:400].replace("\n", " ")
            print(f"Output excerpt: {excerpt}{'…' if len(content) > 400 else ''}")
            results[test_name] = {"ok": True, "meta": meta, "output": content}
        except Exception as e:
            print(f"ERROR: {e}")
            results[test_name] = {"ok": False, "error": str(e)}
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    print("\n==== Summary (JSON) ====")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
