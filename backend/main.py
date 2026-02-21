"""
FastAPI Backend for CareerPath AI
Handles resume upload, text extraction, and career path prediction
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from typing import Dict, List
from app.prediction.predictor import CareerPathPredictor
from app.utils.pdf_extractor import extract_text_from_pdf
from app.services.groq_service import generate_learning_roadmap, generate_certifications
from app.services.jsearch_service import search_jobs as jsearch_search_jobs
import tempfile

# Initialize FastAPI app
app = FastAPI(
    title="CareerPath AI API",
    description="AI-powered career path prediction from resumes",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (loads model on startup)
predictor = CareerPathPredictor()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "CareerPath AI API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with model status"""
    return {
        "status": "healthy",
        "model_loaded": predictor.is_loaded(),
        "model_classes": len(predictor.classes) if predictor.is_loaded() else 0
    }


@app.post("/api/predict")
async def predict_career_path(file: UploadFile = File(...)):
    """
    Predict career path from uploaded resume
    
    Args:
        file: Uploaded resume file (PDF format)
    
    Returns:
        JSON with predicted career path, confidence, and top predictions
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported. Please upload a PDF resume."
            )
        
        # Validate file size (10MB max)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds 10MB limit. Your file is {file_size / (1024*1024):.2f}MB."
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Extract text from PDF
            resume_text = extract_text_from_pdf(tmp_path)
            
            if not resume_text or len(resume_text.strip()) < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract sufficient text from the PDF. Please ensure the PDF contains readable text (not scanned images)."
                )
            
            # Get prediction from model (return all classes, max 100 safe limit)
            prediction_result = predictor.predict(resume_text, top_n=100)

            # Use all predictions returned by the model
            top_preds = prediction_result["top_predictions"]
            
            # Prepare detailed predictions with raw scores
            detailed_predictions = []
            for pred in top_preds:
                raw_score = pred["confidence"]
                
                detailed_predictions.append({
                    "career_path": pred["career_path"],
                    "raw_confidence": round(raw_score * 100, 2)
                })

            # Get main prediction confidence
            main_pred_raw = prediction_result["confidence"]
            
            # Get extracted keywords from resume
            extracted_keywords = prediction_result.get("extracted_keywords", [])
            total_distinctive_keywords = prediction_result.get("total_distinctive_keywords", 0)

            return JSONResponse(content={
                "success": True,
                "prediction": prediction_result["prediction"],
                "raw_confidence": round(main_pred_raw * 100, 2),
                "top_predictions": detailed_predictions,
                "extracted_keywords": extracted_keywords,
                "total_distinctive_keywords": total_distinctive_keywords,
                "resume_text": resume_text,
                "filename": file.filename
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing resume: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your resume: {str(e)}"
        )


@app.post("/api/learning-roadmap")
async def generate_roadmap(request: Dict):
    """
    Generate a personalized learning roadmap using Groq LLM
    
    Request body:
        {
            "career_path": str,
            "resume_text": str
        }
    
    Returns:
        JSON roadmap with analysis_summary and 4 modules
    """
    try:
        career_path = request.get("career_path")
        resume_text = request.get("resume_text")
        
        if not career_path or not resume_text:
            raise HTTPException(
                status_code=400,
                detail="Both career_path and resume_text are required"
            )
        
        # Generate roadmap using Groq
        roadmap = generate_learning_roadmap(career_path, resume_text)
        
        return JSONResponse(content={
            "success": True,
            "roadmap": roadmap
        })
        
    except Exception as e:
        print(f"Error generating roadmap: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate learning roadmap: {str(e)}"
        )


@app.post("/api/certifications")
async def generate_certifications_endpoint(request: Dict):
    """
    Generate personalized certification recommendations using Groq LLM
    
    Request body:
        {
            "career_path": str,
            "resume_text": str
        }
    
    Returns:
        JSON certifications with summary and certification list
    """
    try:
        career_path = request.get("career_path")
        resume_text = request.get("resume_text")
        
        if not career_path or not resume_text:
            raise HTTPException(
                status_code=400,
                detail="Both career_path and resume_text are required"
            )
        
        # Generate certifications using Groq
        certifications = generate_certifications(career_path, resume_text)
        
        return JSONResponse(content={
            "success": True,
            "certifications": certifications
        })
        
    except Exception as e:
        print(f"Error generating certifications: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate certifications: {str(e)}"
        )


@app.post("/api/jobs")
async def search_jobs_endpoint(request: Dict):
    """
    Search for jobs using JSearch API
    
    Request body:
        {
            "query": str,
            "location": str (optional, default: "Philippines"),
            "page": int (optional, default: 1),
            "resultsPerPage": int (optional, default: 10),
            "datePosted": str (optional),
            "employmentType": str (optional),
            "remoteOnly": bool (optional)
        }
    
    Returns:
        JSON with job search results
    """
    try:
        query = request.get("query")
        
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query parameter is required"
            )
        
        # Call JSearch service
        result = jsearch_search_jobs(
            query=query,
            location=request.get("location", "Philippines"),
            page=request.get("page", 1),
            results_per_page=request.get("resultsPerPage", 10),
            date_posted=request.get("datePosted"),
            employment_type=request.get("employmentType"),
            remote_only=request.get("remoteOnly", False)
        )
        
        return JSONResponse(content={
            "success": True,
            **result
        })
        
    except Exception as e:
        print(f"Error searching jobs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search jobs: {str(e)}"
        )


@app.get("/api/careers")
async def get_available_careers():
    """Get list of all available career paths the model can predict"""
    try:
        if not predictor.is_loaded():
            raise HTTPException(
                status_code=500,
                detail="Model not loaded"
            )
        
        return {
            "success": True,
            "careers": sorted(predictor.classes),
            "total": len(predictor.classes)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching careers: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting CareerPath AI API server...")
    print("API will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
