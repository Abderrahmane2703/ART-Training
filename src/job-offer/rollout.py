import art
import openai
import random
import json
from pydantic import BaseModel
import time
import os

from get_judge_completion import (
    get_judge_completion,
    get_judge_completion_gpt5_mini,
    get_judge_completion_gpt5_nano,
    get_judge_completion_gpt5_strict,
)
from load_documents import JobContext

from openpipe.client import OpenPipe


op_client = OpenPipe()


class JobOfferScenario(BaseModel):
    context: JobContext
    step: int = 0


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


@art.retry(exceptions=(Exception,))
async def rollout(model: art.Model, scenario: JobOfferScenario) -> art.Trajectory:
    client = model.openai_client()

    # Job offer template
    template = """{{JOB_TITLE}}
Overview
{{ONE_OR_TWO_SENTENCES_OVERVIEW}}

Key Responsibilities
* {{RESPONSIBILITY_1}}
* {{RESPONSIBILITY_2}}
* {{RESPONSIBILITY_3}}
* {{RESPONSIBILITY_4}}
* {{RESPONSIBILITY_5}}

Required Skills & Qualifications
* {{REQUIRED_SKILL_1}}
* {{REQUIRED_SKILL_2}}
* {{REQUIRED_SKILL_3}}
* {{REQUIRED_SKILL_4}}
* {{REQUIRED_SKILL_5}}

Nice-to-Have
* {{NICE_TO_HAVE_1}}
* {{NICE_TO_HAVE_2}}
* {{NICE_TO_HAVE_3}}

Guidelines:
- Overview: 2-3 sentences describing the purpose of the role and its impact on the company
- Key Responsibilities: List 5-7 bullet points with action verbs (e.g., "Develop", "Manage", "Lead", "Optimize")
- Focus on outcomes and accountability, not just tasks
- Skills: Include provided skills and add relevant missing ones"""

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": f"""You are a specialized AI assistant that generates professional job offers in XML format.
You must follow this template structure and output valid XML.

Template:
{template}
""",
            }
        ],
        reward=0,
        metrics={
            "language_consistency": 0,
            "xml_format": 0,
            "context_inclusion": 0,
            "skill_relevance": 0,
            "skill_completeness": 0,
            "total_score": 0,
        },
    )

    # Build the generation prompt
    context_info = f"""Job Title: {scenario.context.job_title}
Language: {scenario.context.language}"""
    
    if scenario.context.skills:
        context_info += f"\nProvided Skills: {', '.join(scenario.context.skills)}"
    
    generation_prompt = f"""Generate a complete job offer based on this context:

{context_info}

Instructions:
1. Use the same language as provided ({scenario.context.language})
2. Include all provided skills and add relevant ones that are missing
3. Create 5-7 key responsibilities using action verbs
4. Output in valid XML format with these tags: <job_offer>, <title>, <overview>, <responsibilities>, <skills>, <nice_to_have>
5. Each responsibility, skill, and nice-to-have should be in its own tag

Generate the job offer now:"""

    trajectory.messages_and_choices.append(
        {"role": "user", "content": generation_prompt}
    )

    requested_at = int(time.time() * 1000)

    # Generate job offer
    messages = trajectory.messages()
    completion = await client.chat.completions.create(
        model=model.inference_model_name, messages=messages, max_tokens=1500
    )
    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)
    generated_offer = choice.message.content

    # Initialize scoring
    total_criteria = 5
    scores = {}

    # Criterion 1: Language Consistency
    language_prompt = f"""Is this text written in {scenario.context.language.upper()} language?

Text to check: {generated_offer[:300]}

Expected language: {scenario.context.language} ({'English' if scenario.context.language == 'en' else 'French' if scenario.context.language == 'fr' else scenario.context.language})

Respond ONLY in JSON format:
{{"answer": "YES" or "NO"}}"""

    language_response = await get_judge_completion_gpt5_nano(language_prompt, max_completion_tokens=250)
    try:
        result = json.loads(clean_json_response(language_response))
        scores["language_consistency"] = 1.0 if result["answer"] == "YES" else 0.0
    except:
        scores["language_consistency"] = 0.0  # Default to 0 if parsing fails

    # Criterion 2: XML Format Validation
    xml_prompt = f"""Is this valid XML format? Check if it has proper opening/closing tags.

Text to check:
{generated_offer}

Respond ONLY in JSON format:
{{"valid_xml": true or false, "has_required_tags": true or false}}"""

    xml_response = await get_judge_completion_gpt5_mini(xml_prompt, max_completion_tokens=400)
    try:
        result = json.loads(clean_json_response(xml_response))
        scores["xml_format"] = 1.0 if result.get("valid_xml", False) else 0.0
    except:
        scores["xml_format"] = 0.0

    # Criterion 3: Context Inclusion (check if provided skills are included)
    if scenario.context.skills:
        context_prompt = f"""Evaluate skill inclusion with deduplication penalty.

Required skills that MUST be included: {', '.join(scenario.context.skills)}

Generated job offer:
{generated_offer}

Instructions:
1. Extract ALL skills mentioned in the job offer (from all sections)
2. For each required skill, check if it's present (exact or fuzzy match)
3. Identify duplicate/redundant skills (e.g., "Python" and "Python Programming", "ML" and "Machine Learning")
4. Calculate base_score = matched_required_skills / total_required_skills
5. Calculate total_required_skills_extracted = total_required_skills + duplicate_count.
6. Calculate deduplication_factor = 1 - (duplicate_count / total_extracted_skills)
7. Calculate final_score = base_score * deduplication_factor

Example of duplicates to detect:
- "Python" + "Python" = duplicate
- "Python" + "Python Programming" = duplicate
- "Machine Learning" + "ML" = duplicate
- "Data Analysis" + "Data Analytics" = duplicate
- "Communication" + "Communication Skills" = duplicate

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

        context_response = await get_judge_completion_gpt5_strict(context_prompt, max_completion_tokens=800)
        try:
            result = json.loads(clean_json_response(context_response))
            scores["context_inclusion"] = result.get("final_score", 0.0)
        except:
            scores["context_inclusion"] = 0.0
    else:
        scores["context_inclusion"] = 1.0  # No skills to check

    # Criterion 4: Skill Relevance (only for NEW skills added by the model)
    provided_skills_str = ', '.join(scenario.context.skills) if scenario.context.skills else 'None'
    skill_relevance_prompt = f"""For a {scenario.context.job_title} position:

1. Extract ALL skills mentioned in the job offer (both in Required Skills and Nice-to-Have sections)
2. EXCLUDE these provided skills from evaluation: {provided_skills_str}
3. Ignore the duplicate skills
4. For each NEW skill added by the model, score it:
   - 1 if relevant to {scenario.context.job_title}
   - 0 if not relevant
5. Calculate final score = sum(skill_scores) / total_new_skills

Generated job offer:
{generated_offer}

Respond ONLY as a strict JSON object with a single field:
{{"final_score": 0.0-1.0}}"""

    skill_relevance_response = await get_judge_completion_gpt5_mini(skill_relevance_prompt, max_completion_tokens=800)
    try:
        result = json.loads(clean_json_response(skill_relevance_response))
        scores["skill_relevance"] = result.get("final_score", 0.5)
    except:
        scores["skill_relevance"] = 0.5

    # Criterion 5: Skill Completeness (check for missing obvious skills)
    completeness_prompt = f"""Evaluate skill completeness for {scenario.context.job_title} position.

Review the generated job offer and identify essential skills that are missing.
Essential = skills that 80%+ of {scenario.context.job_title} job postings would include.

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

    completeness_response = await get_judge_completion_gpt5_strict(completeness_prompt, max_completion_tokens=800)
    try:
        result = json.loads(clean_json_response(completeness_response))
        scores["skill_completeness"] = result.get("final_score", 0.5)
    except:
        scores["skill_completeness"] = 0.5  # Default if parsing fails

    # Calculate final score (weighted average)
    final_score = (
        scores["language_consistency"] * 0.2 +  # 20%
        scores["xml_format"] * 0.2 +            # 20%
        scores["context_inclusion"] * 0.2 +     # 20%
        scores["skill_relevance"] * 0.2 +       # 20%
        scores["skill_completeness"] * 0.2      # 20%
    )

    # Update trajectory metrics
    trajectory.metrics.update(scores)
    trajectory.metrics["total_score"] = final_score
    trajectory.reward = final_score * 10  # Scale to 0-10

    # Debug output (occasional)
    if random.random() < 0.05:
        print("\n" + "="*50)
        print(f"Job Title: {scenario.context.job_title}")
        print(f"Language: {scenario.context.language}")
        print(f"Provided Skills: {scenario.context.skills}")
        print("-"*50)
        print("Scores:")
        for key, value in scores.items():
            print(f"  {key}: {value:.2f}")
        print(f"Final Score: {final_score:.2f}")
        print("-"*50)
        print(f"Generated (first 500 chars):\n{generated_offer[:500]}...")
        print("="*50 + "\n")

    # Report to OpenPipe if configured
    if os.getenv("OPENPIPE_API_KEY"):
        try:
            op_client.report(
                requested_at=requested_at,
                received_at=int(time.time() * 1000),
                req_payload={
                    "model": model.name,
                    "messages": messages,
                    "metadata": {
                        "project": "job-offer-generation",
                        "step": scenario.step,
                        "language": scenario.context.language,
                        "job_title": scenario.context.job_title,
                        **scores,
                        "final_score": final_score,
                    },
                },
                resp_payload=completion,
                status_code=200,
            )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

    return trajectory