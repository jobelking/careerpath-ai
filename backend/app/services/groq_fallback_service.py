"""
Groq LLM Fallback Service — Prediction Verification
Uses a dedicated GROQ_API_KEY_FALLBACK to verify uncertain predictions
when the ML model's raw confidence is below 10%.
"""

import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

GROQ_API_KEY_FALLBACK = os.getenv("GROQ_API_KEY_FALLBACK")

# Separate OpenAI client for fallback verification — avoids rate-limit contention
client = OpenAI(
    api_key=GROQ_API_KEY_FALLBACK,
    base_url="https://api.groq.com/openai/v1"
)


def verify_prediction(
    resume_text: str,
    predicted_career: str,
    top_predictions: List[Dict[str, Any]],
    all_career_paths: List[str]
) -> Dict[str, Any]:
    """
    Verify the top 3 uncertain ML predictions using a Groq LLM.

    When the ML model's top-1 raw confidence is below 10%, this function
    asks the LLM to review the resume and verify or correct all three
    top predictions.

    Args:
        resume_text: The raw resume text extracted from the PDF
        predicted_career: The ML model's top prediction
        top_predictions: The model's top 3 predictions with raw scores
            e.g. [{"career_path": "...", "raw_confidence": 5.2}, ...]
        all_career_paths: Full list of 26 career paths the model supports

    Returns:
        Dict with structure:
        {
            "verified_predictions": [
                {"position": 1, "original": "...", "is_correct": bool, "verified_career": "..."},
                {"position": 2, "original": "...", "is_correct": bool, "verified_career": "..."},
                {"position": 3, "original": "...", "is_correct": bool, "verified_career": "..."}
            ],
            "explanation": "Brief reason"
        }

    Raises:
        Exception: If API call fails or response is invalid
    """
    if not GROQ_API_KEY_FALLBACK:
        raise Exception("GROQ_API_KEY_FALLBACK not found in environment variables")

    # Format the top predictions for the prompt
    top_3 = top_predictions[:3]
    top_preds_text = "\n".join(
        f"  {i+1}. {p['career_path']} (raw confidence: {p['raw_confidence']:.1f}%)"
        for i, p in enumerate(top_3)
    )

    # Format all career paths
    career_list_text = "\n".join(f"  - {c}" for c in all_career_paths)

    # Build the positions JSON for the prompt example
    positions_example = []
    for i, p in enumerate(top_3):
        positions_example.append({
            "position": i + 1,
            "original": p["career_path"],
            "is_correct": True,
            "verified_career": p["career_path"]
        })
    positions_json = json.dumps(positions_example, indent=2)

    prompt = f"""Act as a Career Classification Expert. Your task is to verify whether a machine learning model's top 3 career path predictions are correct for a given resume.

CONTEXT:
The ML model predicted career paths for a resume but has LOW confidence (below 10% raw probability). Your job is to review the resume and determine whether EACH of the top 3 predictions is correct, or if a different career path from the allowed list is a better match for that ranking position.

INPUT DATA:
- ML Model's Top 3 Predictions:
{top_preds_text}

- Resume Text:
{resume_text[:3000]}

- All Allowed Career Paths:
{career_list_text}

INSTRUCTIONS:
1. READ the resume carefully — focus on skills, experience, job titles, technologies, and education.
2. For EACH of the top 3 predictions, evaluate whether it is reasonable and appropriate for that ranking position.
3. If a prediction is correct or close enough for its position, set "is_correct" to true.
4. If a DIFFERENT career path from the allowed list is clearly a better fit for that position, set "is_correct" to false and set "verified_career" to the better career path.
5. Each verified_career must be UNIQUE — do not assign the same career to multiple positions.
6. Provide a brief 1-sentence overall explanation.

CRITICAL RULES:
- Every "verified_career" MUST be EXACTLY one of the career paths from the "All Allowed Career Paths" list — use the exact same spelling and casing.
- Do NOT invent new career paths.
- Each verified_career across all 3 positions must be DIFFERENT (no duplicates).
- If the model's prediction is roughly correct (even if not perfect), consider it correct.
- Only correct it when you are confident a different path is clearly better for that position.

STRICT OUTPUT FORMAT:
Return ONLY a valid JSON object:
{{
  "verified_predictions": {positions_json},
  "explanation": "Brief overall explanation of your reasoning."
}}

RULES:
- No markdown blocks or conversational text
- Return valid JSON only
- "is_correct" must be a boolean
- "verified_career" must match an entry from the allowed career paths list exactly
- "explanation" must be a single sentence
- All 3 positions must be included in the response"""

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
        if "verified_predictions" not in result:
            raise Exception("Invalid response structure: missing 'verified_predictions'")

        verified = result["verified_predictions"]
        if not isinstance(verified, list) or len(verified) < 1:
            raise Exception(f"Expected at least 1 verified prediction, got {len(verified) if isinstance(verified, list) else 0}")

        # Validate and sanitize each verified prediction
        seen_careers = set()
        for i, vp in enumerate(verified):
            if "is_correct" not in vp or "verified_career" not in vp:
                raise Exception(f"Prediction {i+1} missing required fields")

            if not isinstance(vp["is_correct"], bool):
                raise Exception(f"Prediction {i+1}: 'is_correct' must be boolean")

            # Validate verified_career is in the allowed list
            career = vp["verified_career"]
            if career not in all_career_paths:
                # Try case-insensitive match
                match = None
                for cp in all_career_paths:
                    if cp.lower() == career.lower():
                        match = cp
                        break
                if match:
                    vp["verified_career"] = match
                else:
                    # Fallback: keep the original prediction for this position
                    original = top_3[i]["career_path"] if i < len(top_3) else career
                    print(f"Warning: LLM returned unknown career '{career}' for position {i+1}, keeping original '{original}'.")
                    vp["is_correct"] = True
                    vp["verified_career"] = original

            # Check for duplicates
            if vp["verified_career"] in seen_careers:
                # Duplicate — revert to original
                original = top_3[i]["career_path"] if i < len(top_3) else vp["verified_career"]
                vp["is_correct"] = True
                vp["verified_career"] = original

            seen_careers.add(vp["verified_career"])

            # Ensure position and original fields are set
            vp["position"] = i + 1
            if "original" not in vp and i < len(top_3):
                vp["original"] = top_3[i]["career_path"]

        # Ensure explanation field exists
        if "explanation" not in result:
            result["explanation"] = ""

        return result

    except Exception as e:
        raise Exception(f"Failed to verify prediction: {str(e)}")
