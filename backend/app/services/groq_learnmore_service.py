"""
Groq LLM Service for LearnMore Page — Skills & Improvements
Uses a dedicated GROQ_API_KEY_LEARNMORE to avoid shared rate limiting
with the roadmap/certification service.
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

GROQ_API_KEY_LEARNMORE = os.getenv("GROQ_API_KEY_LEARNMORE")

# Separate OpenAI client for LearnMore — avoids rate-limit contention
client = OpenAI(
    api_key=GROQ_API_KEY_LEARNMORE,
    base_url="https://api.groq.com/openai/v1"
)


def generate_skills_and_improvements(career_path: str, resume_text: str) -> Dict[str, Any]:
    """
    Analyze a resume against a predicted career path and return:
      - skills_driving_score: 4 skills the user already has that matched this career
      - improve_your_match:   4 gaps the user should close to increase their fit

    Args:
        career_path: The predicted career path (e.g. "Software Development Careers")
        resume_text: The raw resume text extracted from the PDF

    Returns:
        Dict with structure:
        {
          "skills_driving_score": [ "Proficient in Python & Django", ... ],
          "improve_your_match":   [ "Gain cloud deployment skills", ... ]
        }

    Raises:
        Exception: If API call fails or response is invalid
    """
    if not GROQ_API_KEY_LEARNMORE:
        raise Exception("GROQ_API_KEY_LEARNMORE not found in environment variables")

    prompt = f"""Act as a Career-Fit Analyst. Examine a resume and identify (a) existing skills that drove the career match, and (b) improvement areas that would increase the match score.

INPUT DATA:
- Predicted Career Path: [{career_path}]
- Raw Resume Text: {resume_text}

INSTRUCTIONS:
1. ANALYZE the resume — look for technologies, tools, soft skills, job titles, accomplishments.
2. IDENTIFY exactly 4 skills/competencies the user ALREADY has that contributed to matching [{career_path}]. Write each as a SHORT sentence describing the skill and its relevance.
3. IDENTIFY exactly 4 skill gaps — areas where the resume shows NO evidence and that are important for [{career_path}]. Write each as a SHORT sentence describing what to improve.
4. Be SPECIFIC — use real skill/tool/framework names.
5. For skills_driving_score, only list skills you found evidence for in the resume.
6. For improve_your_match, only list gaps — do NOT repeat items from skills_driving_score.

STRICT OUTPUT FORMAT:
Return ONLY a valid JSON object with two arrays of short sentence strings:
{{
  "skills_driving_score": [
    "Proficient in Python & Django development",
    "Experience building REST APIs",
    "Skilled in PostgreSQL database management",
    "Uses Git for version control"
  ],
  "improve_your_match": [
    "Learn cloud deployment with AWS or GCP",
    "Develop system design knowledge",
    "Set up CI/CD pipelines",
    "Build technical leadership experience"
  ]
}}

RULES:
- No markdown blocks or conversational text
- Return exactly 4 items in each array
- Each item is a SHORT sentence (4-8 words), self-explanatory
- Do NOT add any sub-fields or objects — just strings
- Ensure valid JSON"""

    try:
        response = client.responses.create(
            model="openai/gpt-oss-120b",
            input=[
                {
                    "role": "system",
                    "content": "You are an expert career analyst. Always respond with valid JSON only, no markdown, no extra text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            text={
                "format": {
                    "type": "json_object"
                }
            }
        )

        # Parse JSON
        try:
            result = json.loads(response.output_text)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON from Groq: {str(e)}. Response: {response.output_text[:300]}")

        # Validate structure
        if "skills_driving_score" not in result or "improve_your_match" not in result:
            raise Exception("Invalid response structure: missing required fields")

        if not isinstance(result["skills_driving_score"], list) or len(result["skills_driving_score"]) != 4:
            raise Exception(f"Expected 4 skills_driving_score items, got {len(result.get('skills_driving_score', []))}")

        if not isinstance(result["improve_your_match"], list) or len(result["improve_your_match"]) != 4:
            raise Exception(f"Expected 4 improve_your_match items, got {len(result.get('improve_your_match', []))}")

        # Validate each item is a string
        for i, item in enumerate(result["skills_driving_score"]):
            if not isinstance(item, str):
                raise Exception(f"skills_driving_score[{i}] must be a string, got {type(item).__name__}")

        for i, item in enumerate(result["improve_your_match"]):
            if not isinstance(item, str):
                raise Exception(f"improve_your_match[{i}] must be a string, got {type(item).__name__}")

        return result

    except Exception as e:
        raise Exception(f"Failed to generate skills insights: {str(e)}")
