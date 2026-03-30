"""
Pydantic models for Prediction History request/response validation.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Any


class SaveHistoryRequest(BaseModel):
    prediction_result: str
    input_data: Optional[str] = None
    confidence_score: Optional[float] = None
    top_predictions: Optional[List[Any]] = None
    filename: Optional[str] = None
    extracted_keywords: Optional[List[Any]] = None
    total_distinctive_keywords: Optional[int] = None


class UpdateHistoryRequest(BaseModel):
    """Partial update — only the fields provided will be written."""
    learning_roadmap: Optional[Any] = None
    certification_data: Optional[Any] = None


class HistoryRecord(BaseModel):
    id: int
    user_id: int
    prediction_result: str
    input_data: Optional[str] = None
    confidence_score: Optional[float] = None
    top_predictions: Optional[List[Any]] = None
    filename: Optional[str] = None
    extracted_keywords: Optional[List[Any]] = None
    total_distinctive_keywords: Optional[int] = None
    learning_roadmap: Optional[Any] = None
    certification_data: Optional[Any] = None
    has_resume: Optional[bool] = False
    date_created: Optional[datetime] = None

    class Config:
        from_attributes = True


class SaveHistoryResponse(BaseModel):
    success: bool
    id: Optional[int] = None
    message: Optional[str] = None


class UserHistoryResponse(BaseModel):
    success: bool
    history: List[HistoryRecord]
    total: int
