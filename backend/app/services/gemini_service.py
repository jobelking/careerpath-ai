"""
Gemini API Service
Handles communication with Google's Gemini LLM for generating personalized learning roadmaps
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def generate_learning_roadmap(career_path: str, resume_text: str) -> Dict[str, Any]:
    """
    Generate a personalized learning roadmap using Gemini LLM.
    
    Args:
        career_path: The predicted career path (e.g., "Software Development Careers")
        resume_text: The raw resume text extracted from the PDF
        
    Returns:
        Dict with structure: { "analysis_summary": str, "modules": [...] }
        
    Raises:
        Exception: If API call fails or response is invalid
    """
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not found in environment variables")
    
    # Build the prompt using the Industry Education Specialist instructions
    prompt = f"""Act as an Industry Education Specialist. Your task is to generate a personalized fundamental learning roadmap for a user who has been predicted for the [{career_path}] career path.

INPUT DATA:
- Predicted Career Path: [{career_path}]
- Raw Resume Text: {resume_text}

INSTRUCTIONS:
1. NARRATIVE ANALYSIS: Analyze the raw resume text to identify the user's current seniority, existing technical skills, and industry experience. 
2. GAP ANALYSIS: Identify the "knowledge gaps" between their current resume and the requirements of the [{career_path}]. 
3. PERSONALIZATION: Do not suggest learning topics they clearly already possess. If they have the basics, suggest intermediate "pillar" concepts for this path.
4. CONTENT: Create 4 distinct modules. Each module must focus on a core technical or theoretical pillar of the [{career_path}].

STRICT OUTPUT FORMAT:
Return ONLY a JSON object with this exact structure:
{{
  "analysis_summary": "A 1-sentence summary of why this roadmap was chosen based on their resume.",
  "modules": [
    {{
      "id": 1,
      "title": "Module Name",
      "description": "2-sentence summary of the module's importance to the [{career_path}].",
      "key_concepts": ["Concept 1", "Concept 2", "Concept 3"]
    }}
  ]
}}

RULES:
- No conversational text or markdown blocks.
- Ensure the JSON is valid for parsing.
- Focus on "Fundamentals" but scale the difficulty based on the user's resume.
- Return exactly 4 modules.
- Each module should have exactly 3 key concepts."""

    try:
        # Initialize Gemini client with API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        
        # Extract the generated text
        generated_text = response.text.strip()
        
        # Clean up the response (remove markdown code blocks if present)
        if generated_text.startswith("```json"):
            generated_text = generated_text[7:]  # Remove ```json
        if generated_text.startswith("```"):
            generated_text = generated_text[3:]  # Remove ```
        if generated_text.endswith("```"):
            generated_text = generated_text[:-3]  # Remove trailing ```
        generated_text = generated_text.strip()
        
        # Parse the JSON response
        try:
            roadmap = json.loads(generated_text)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON from Gemini: {str(e)}. Response: {generated_text[:200]}")
        
        # Validate structure
        if "analysis_summary" not in roadmap or "modules" not in roadmap:
            raise Exception("Invalid roadmap structure: missing required fields")
        
        if not isinstance(roadmap["modules"], list) or len(roadmap["modules"]) != 4:
            raise Exception(f"Expected 4 modules, got {len(roadmap.get('modules', []))}")
        
        # Validate each module
        for i, module in enumerate(roadmap["modules"]):
            required_fields = ["id", "title", "description", "key_concepts"]
            for field in required_fields:
                if field not in module:
                    raise Exception(f"Module {i+1} missing required field: {field}")
            
            if not isinstance(module["key_concepts"], list) or len(module["key_concepts"]) != 3:
                raise Exception(f"Module {i+1} must have exactly 3 key_concepts")
        
        return roadmap
        
    except Exception as e:
        # Re-raise with context
        raise Exception(f"Failed to generate learning roadmap: {str(e)}")
