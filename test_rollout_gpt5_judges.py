#!/usr/bin/env python3
"""
Rollout-like test for new judges:
- gpt-5-nano: language consistency, XML format
- gpt-5-mini: context inclusion, skill relevance
- existing judge (gpt-35 or your configured Azure deployment): skill completeness
"""
import os
import sys
import json
import asyncio
from dotenv import load_dotenv

# Make src/job-offer importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_OFFER_DIR = os.path.join(CURRENT_DIR, "src", "job-offer")
if JOB_OFFER_DIR not in sys.path:
    sys.path.append(JOB_OFFER_DIR)

from get_judge_completion import (
    get_judge_completion,  # existing judge
    get_judge_completion_gpt5_nano,
    get_judge_completion_gpt5_mini,
    get_judge_completion_gpt5_strict,
    clear_judge_cache_gpt5_nano,
    clear_judge_cache_gpt5_mini,
)

load_dotenv()


def clean_json_response(response: str) -> str:
    """Clean JSON response from markdown code blocks."""
    clean = response.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


async def run_tests():
    print("\n🧪 Rollout-like test for new judges (nano/mini + existing)")
    print("=" * 70)

    # Clear judge caches to avoid stale hits
    try:
        clear_judge_cache_gpt5_nano()
    except Exception:
        pass
    try:
        clear_judge_cache_gpt5_mini()
    except Exception:
        pass

    # Scenario
    job_title = os.getenv("TEST_JOB_TITLE", "Data Scientist")
    language = os.getenv("TEST_LANGUAGE", "en")
    provided_skills = [
        s.strip() for s in os.getenv("TEST_SKILLS", "Python, Machine Learning, SQL").split(",") if s.strip()
    ]

    generated_offer = f"""
<job_offer>
  <title>{job_title}</title>
  <overview>
    We are seeking a Data Scientist to drive data-driven decisions and build predictive models.
    You will collaborate with cross-functional teams to deploy scalable solutions.
  </overview>
  <responsibilities>
    <item>Develop and deploy machine learning models</item>
    <item>Analyze large datasets to extract insights</item>
    <item>Collaborate with data engineers and stakeholders</item>
    <item>Design A/B tests and measure impact</item>
    <item>Communicate findings through dashboards and reports</item>
  </responsibilities>
  <skills>
    <item>Python</item>
    <item>Machine Learning</item>
    <item>SQL</item>
    <item>ML</item>
    <item>Data Analytics</item>
    <item>Statistics</item>
    <item>Python Programming</item>
  </skills>
  <nice_to_have>
    <item>Cloud (AWS or GCP)</item>
    <item>Deep Learning</item>
  </nice_to_have>
</job_offer>
""".strip()

    scores = {}

    # 1) Language Consistency — nano
    language_prompt = f"""Is this text written in {language.upper()} language?

Text to check: {generated_offer[:300]}

Expected language: {language} ({'English' if language == 'en' else 'French' if language == 'fr' else language})

Respond ONLY in JSON format:
{{"answer": "YES" or "NO"}}"""

    lang_resp = await get_judge_completion_gpt5_nano(language_prompt, max_completion_tokens=250)
    try:
        result = json.loads(clean_json_response(lang_resp))
        scores["language_consistency"] = 1.0 if result.get("answer") == "YES" else 0.0
    except Exception:
        scores["language_consistency"] = 0.0
    print(f"• Language consistency: {scores['language_consistency']:.2f}")
    print("LLM language response (nano):")
    print(lang_resp)

    # 2) XML Format Validation — mini
    xml_prompt = f"""Is this valid XML format? Check if it has proper opening/closing tags.

Text to check:
{generated_offer}

Respond ONLY in JSON format:
{{"valid_xml": true or false, "has_required_tags": true or false}}"""

    xml_resp = await get_judge_completion_gpt5_mini(xml_prompt, max_completion_tokens=400)
    try:
        result = json.loads(clean_json_response(xml_resp))
        scores["xml_format"] = 1.0 if result.get("valid_xml", False) else 0.0
    except Exception:
        scores["xml_format"] = 0.0
    print(f"• XML format: {scores['xml_format']:.2f}")
    print("LLM XML response (mini):")
    print(xml_resp)

    # 3) Context Inclusion (with dedup penalty) — stric
    provided_skills_str = ", ".join(provided_skills)
    context_prompt = f"""Evaluate skill inclusion with deduplication penalty.

Required skills that MUST be included: {provided_skills_str}

Generated job offer:
{generated_offer}

Instructions:
1. Extract ALL skills mentioned in the job offer (from all sections)
2. For each required skill, check if it's present (exact or fuzzy match)
3. Identify duplicate/redundant skills (e.g., "Python" and "Python Programming", "ML" and "Machine Learning")
4. Calculate base_score = matched_required_skills / total_required_skills
5. Compute total_required_skills_extracted = total_required_skills + duplicate_count.
6. Calculate deduplication_factor = 1 - (duplicate_count / total_extracted_skills)
7. Calculate final_score = base_score * deduplication_factor

Respond ONLY in JSON format:
{{
  "required_skills": ["skill1", "skill2"],
  "extracted_skills": ["skill1", "skill2", ...],
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": [],
  "duplicate_groups": [["Python", "Python Programming"], ["ML", "Machine Learning"]],
  "duplicate_count": number,
  "base_score": 0.0-1.0,
  "deduplication_factor": 0.0-1.0,
  "final_score": 0.0-1.0
}}"""

    context_resp = await get_judge_completion_gpt5_strict(context_prompt, max_completion_tokens=800)
    try:
        result = json.loads(clean_json_response(context_resp))
        scores["context_inclusion"] = float(result.get("final_score", 0.0))
    except Exception:
        scores["context_inclusion"] = 0.0
    print(f"• Context inclusion: {scores['context_inclusion']:.2f}")
    print("LLM context inclusion response (strict):")
    print(context_resp)

    # 4) Skill Relevance — mini
    skill_relevance_prompt = f"""For a {job_title} position:

1. Extract ALL skills mentioned in the job offer (both in Required Skills and Nice-to-Have sections)
2. EXCLUDE these provided skills from evaluation: {provided_skills_str or 'None'}
3. Ignore the duplicate skills
4. For each NEW skill added by the model, score it:
   - 1 if relevant to {job_title}
   - 0 if not relevant
5. Calculate final score = sum(skill_scores) / total_new_skills
6. If there are zero new skills, set final_score = 1.0

Generated job offer:
{generated_offer}

Respond ONLY as a strict JSON object with a single field:
{{"final_score": 0.0-1.0}}"""

    rel_resp = await get_judge_completion_gpt5_mini(skill_relevance_prompt, max_completion_tokens=800)
    try:
        result = json.loads(clean_json_response(rel_resp))
        scores["skill_relevance"] = float(result.get("final_score", 0.5))
    except Exception:
        scores["skill_relevance"] = 0.5
    print(f"• Skill relevance: {scores['skill_relevance']:.2f}")
    print("LLM skill relevance response (mini):")
    print(rel_resp)

    # 5) Skill Completeness — existing judge
    completeness_prompt = f"""Evaluate skill completeness for {job_title} position.

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
  "skills_present": ["skill1", "skill2", ...],
  "missing_essentials": [
    {{"skill": "Git", "importance": 1.0, "reason": "Version control is critical for any developer role"}},
    {{"skill": "Testing", "importance": 0.5, "reason": "Most positions require testing knowledge"}}
  ],
  "total_penalty": 0.0-1.0,
  "final_score": 0.0-1.0
}}"""

    comp_resp = await get_judge_completion_gpt5_strict(completeness_prompt, max_completion_tokens=800)
    try:
        result = json.loads(clean_json_response(comp_resp))
        scores["skill_completeness"] = float(result.get("final_score", 0.5))
    except Exception:
        scores["skill_completeness"] = 0.5
    print(f"• Skill completeness: {scores['skill_completeness']:.2f}")
    print("LLM skill completeness response (existing judge):")
    print(comp_resp)

    final_score = (
        scores.get("language_consistency", 0.0) * 0.2
        + scores.get("xml_format", 0.0) * 0.2
        + scores.get("context_inclusion", 0.0) * 0.2
        + scores.get("skill_relevance", 0.0) * 0.2
        + scores.get("skill_completeness", 0.0) * 0.2
    )

    print("\nResults")
    print("-" * 70)
    print(f"Language consistency: {scores['language_consistency']:.2f}")
    print(f"XML format:          {scores['xml_format']:.2f}")
    print(f"Context inclusion:   {scores['context_inclusion']:.2f}")
    print(f"Skill relevance:     {scores['skill_relevance']:.2f}")
    print(f"Skill completeness:  {scores['skill_completeness']:.2f}")
    print(f"Final score:         {final_score:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_tests())
