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
    
    # Build the prompt to return improvement areas with curated resources
    prompt = f"""Act as an Industry Education Specialist. Your task is to identify skill gaps and recommend curated learning resources for a user predicted for the [{career_path}] career path.

INPUT DATA:
- Predicted Career Path: [{career_path}]
- Raw Resume Text: {resume_text}

INSTRUCTIONS:
1. ANALYZE the resume to identify current skills, seniority, and experience.
2. IDENTIFY 4 key skill gaps between their current profile and the [{career_path}] requirements.
3. For each gap, provide:
   - Skill name (concise)
   - Brief reason why this skill is needed (1 sentence max)
   - 2-4 curated learning resources (videos, courses, or articles)
4. PRIORITIZE resources from reputable providers (YouTube, Coursera, Udemy, freeCodeCamp, Medium, etc.)
5. Do NOT suggest skills they clearly already possess.

URL FORMAT INSTRUCTIONS:
- Generate direct platform search URLs using each platform's search functionality
- Use the following URL formats:
  * YouTube: https://www.youtube.com/results?search_query=[Topic]
  * Coursera: https://www.coursera.org/search?query=[Topic]
  * Udemy: https://www.udemy.com/courses/search/?q=[Topic]
  * Medium: https://medium.com/search?q=[Topic]
  * freeCodeCamp: https://www.freecodecamp.org/news/search/?query=[Topic]
- For [Topic], use the skill name plus relevant keywords (URL-encoded with + for spaces)
- Example: For a Python tutorial on YouTube, use: https://www.youtube.com/results?search_query=Python+Tutorial

STRICT OUTPUT FORMAT:
Return ONLY a valid JSON object:
{{
  "analysis_summary": "One-sentence summary of their current level and gaps.",
  "improvement_areas": [
    {{
      "id": 1,
      "skill": "Skill Name",
      "why": "Why this skill is important for {career_path}.",
      "resources": [
        {{
          "title": "Resource Title",
          "type": "video",
          "provider": "YouTube",
          "url": "https://www.youtube.com/results?search_query=Skill+Name+Tutorial"
        }},
        {{
          "title": "Course Title",
          "type": "course",
          "provider": "Coursera",
          "url": "https://www.coursera.org/search?query=Skill+Name+Course"
        }}
      ]
    }}
  ]
}}

RULES:
- No markdown blocks or conversational text
- Return exactly 4 improvement areas
- Each area must have 2-4 resources
- Resource types: "video", "course", or "article" only
- All URLs must use the direct platform search format (YouTube, Coursera, Udemy, etc.)
- Ensure valid JSON"""

    try:
        # Initialize Gemini client with API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Call Gemini API with structured JSON output
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )
        
        # Extract the generated JSON (already pure JSON due to response_mime_type)
        generated_text = response.text.strip()
        
        # Parse the JSON response
        try:
            roadmap = json.loads(generated_text)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON from Gemini: {str(e)}. Response: {generated_text[:200]}")
        
        # Validate structure
        if "analysis_summary" not in roadmap or "improvement_areas" not in roadmap:
            raise Exception("Invalid roadmap structure: missing required fields")
        
        if not isinstance(roadmap["improvement_areas"], list) or len(roadmap["improvement_areas"]) != 4:
            raise Exception(f"Expected 4 improvement areas, got {len(roadmap.get('improvement_areas', []))}")
        
        # Validate each improvement area
        for i, area in enumerate(roadmap["improvement_areas"]):
            required_fields = ["id", "skill", "why", "resources"]
            for field in required_fields:
                if field not in area:
                    raise Exception(f"Area {i+1} missing required field: {field}")
            
            if not isinstance(area["resources"], list) or len(area["resources"]) < 1:
                raise Exception(f"Area {i+1} must have at least 1 resource")
            
            # Validate each resource
            for j, resource in enumerate(area["resources"]):
                required_resource_fields = ["title", "type", "provider", "url"]
                for field in required_resource_fields:
                    if field not in resource:
                        raise Exception(f"Area {i+1}, Resource {j+1} missing field: {field}")
                
                if resource["type"] not in ["video", "course", "article"]:
                    raise Exception(f"Area {i+1}, Resource {j+1} has invalid type: {resource['type']}")

        
        return roadmap
        
    except Exception as e:
        # Re-raise with context
        raise Exception(f"Failed to generate learning roadmap: {str(e)}")


def generate_certifications(career_path: str, resume_text: str) -> Dict[str, Any]:
    """
    Generate personalized certification recommendations using Gemini LLM.
    
    Args:
        career_path: The predicted career path (e.g., "Software Development Careers")
        resume_text: The raw resume text extracted from the PDF
        
    Returns:
        Dict with structure: { "summary": str, "certifications": [...] }
        
    Raises:
        Exception: If API call fails or response is invalid
    """
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not found in environment variables")
    
    # Build the prompt to return certification recommendations
    prompt = f"""Act as a Career Certification Advisor. Your task is to recommend industry-recognized certifications for a user predicted for the [{career_path}] career path.

INPUT DATA:
- Predicted Career Path: [{career_path}]
- Raw Resume Text: {resume_text}

INSTRUCTIONS:
1. ANALYZE the resume to identify current skills, experience level, and expertise gaps.
2. RECOMMEND 4-6 industry-recognized certifications that would advance their career in [{career_path}].
3. For each certification, provide:
   - Certification name
   - Provider/issuing organization
   - Level (beginner, intermediate, advanced)
   - Why this certification is valuable for their profile (1-2 sentences)
   - Estimated study duration (e.g., "2-3 months", "4-6 weeks")
4. PRIORITIZE certifications from reputable providers (AWS, Google, Microsoft, CompTIA, CISSP, PMI, etc.)
5. ORDER certifications from most relevant to least relevant based on their resume.

URL FORMAT INSTRUCTIONS:
- Generate Google search URLs for each certification
- Format: https://www.google.com/search?q=[Certification+Name+certification]
- Example: https://www.google.com/search?q=AWS+Solutions+Architect+Associate+certification

STRICT OUTPUT FORMAT:
Return ONLY a valid JSON object:
{{
  "summary": "One-sentence overview of certification strategy for {career_path}.",
  "certifications": [
    {{
      "id": 1,
      "name": "Certification Name",
      "provider": "Provider/Organization",
      "level": "intermediate",
      "why": "Why this certification is valuable for their profile.",
      "estimated_duration": "2-3 months",
      "search_url": "https://www.google.com/search?q=Certification+Name+certification"
    }}
  ]
}}

RULES:
- No markdown blocks or conversational text
- Return exactly 4-6 certifications
- Level must be one of: "beginner", "intermediate", "advanced"
- All search_url values must use the Google search format
- Ensure valid JSON"""

    try:
        # Initialize Gemini client with API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Call Gemini API with structured JSON output
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )
        
        # Extract the generated JSON (already pure JSON due to response_mime_type)
        generated_text = response.text.strip()
        
        # Parse the JSON response
        try:
            certifications_data = json.loads(generated_text)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON from Gemini: {str(e)}. Response: {generated_text[:200]}")
        
        # Validate structure
        if "summary" not in certifications_data or "certifications" not in certifications_data:
            raise Exception("Invalid certifications structure: missing required fields")
        
        if not isinstance(certifications_data["certifications"], list) or not (4 <= len(certifications_data["certifications"]) <= 6):
            raise Exception(f"Expected 4-6 certifications, got {len(certifications_data.get('certifications', []))}")
        
        # Validate each certification
        for i, cert in enumerate(certifications_data["certifications"]):
            required_fields = ["id", "name", "provider", "level", "why", "estimated_duration", "search_url"]
            for field in required_fields:
                if field not in cert:
                    raise Exception(f"Certification {i+1} missing required field: {field}")
            
            if cert["level"] not in ["beginner", "intermediate", "advanced"]:
                raise Exception(f"Certification {i+1} has invalid level: {cert['level']}")

        return certifications_data
        
    except Exception as e:
        # Re-raise with context
        raise Exception(f"Failed to generate certifications: {str(e)}")