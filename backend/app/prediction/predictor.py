"""
Career Path Prediction Service
Loads the trained ML model and provides prediction functionality
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.prediction.inference_classifier import InferenceCareerPathClassifier


class CareerPathPredictor:
    """
    Service class for career path prediction.
    Loads trained model and provides prediction methods.
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_name: str = 'advanced_career_path_cnb',
        eager_load: Optional[bool] = None,
        enable_location_removal: Optional[bool] = None
    ):
        """
        Initialize the predictor and load the trained model.
        
        Args:
            model_dir: Directory containing model files (defaults to data/trained_models)
            model_name: Base name of model files
        """
        if model_dir is None:
            # Default to the trained_models directory
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_dir = os.path.join(backend_dir, 'data', 'trained_models')
        
        self.model_dir = model_dir
        self.model_name = model_name
        self.classifier = None
        self.classes = []

        if eager_load is None:
            eager_load = os.getenv("MODEL_EAGER_LOAD", "false").lower() == "true"
        if enable_location_removal is None:
            enable_location_removal = os.getenv("ENABLE_LOCATION_REMOVAL_INFERENCE", "false").lower() == "true"

        self.eager_load = eager_load
        self.enable_location_removal = enable_location_removal

        if self.eager_load:
            self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk."""
        try:
            print(f"Loading model from: {self.model_dir}")
            
            # Check if model files exist
            required_files = [
                f'{self.model_name}_classifier.pkl',
                f'{self.model_name}_vectorizer.pkl',
                f'{self.model_name}_classes.pkl',
                f'{self.model_name}_synonyms.pkl'
            ]
            
            for file in required_files:
                file_path = os.path.join(self.model_dir, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Model file not found: {file_path}")
            
            # Load model using lightweight inference wrapper
            self.classifier = InferenceCareerPathClassifier.load_model(
                self.model_dir, 
                self.model_name,
                enable_location_removal=self.enable_location_removal
            )
            
            # Store classes
            self.classes = self.classifier.classes_
            
            print(f"✓ Model loaded successfully!")
            print(f"✓ Number of career paths: {len(self.classes)}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise Exception(f"Failed to load model: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.classifier is not None and self.classes is not None

    def _ensure_loaded(self):
        """Lazy-load model on first inference request."""
        if not self.is_loaded():
            self._load_model()
    
    def predict(self, resume_text: str, top_n: int = 5) -> Dict:
        """
        Predict career path from resume text.
        
        Args:
            resume_text: Raw resume text
            top_n: Number of top predictions to return
            
        Returns:
            Dictionary containing:
                - prediction: Top predicted career path
                - confidence: Confidence score (0-1)
                - top_predictions: List of top N predictions with scores
        """
        self._ensure_loaded()
        
        if not resume_text or len(resume_text.strip()) < 10:
            raise ValueError("Resume text is too short or empty")
        
        try:
            # Preprocess the text using the model's preprocessing
            preprocessed_text = self.classifier.preprocess_text(resume_text)
            
            # Vectorize the text
            text_vectorized = self.classifier.vectorizer.transform([preprocessed_text])
            
            # Get prediction probabilities
            probabilities = self.classifier.classifier.predict_proba(text_vectorized)[0]
            
            # Get top prediction
            top_idx = np.argmax(probabilities)
            top_prediction_raw = self.classes[top_idx]
            top_prediction_display = self.classifier.get_display_name(top_prediction_raw)
            top_confidence = probabilities[top_idx]
            
            # Get top N predictions
            top_indices = np.argsort(probabilities)[-top_n:][::-1]
            top_predictions = [
                {
                    "career_path": self.classifier.get_display_name(self.classes[idx]),
                    "confidence": float(probabilities[idx])
                }
                for idx in top_indices
            ]
            
            # Extract keywords from resume that are DISTINCTIVE for the predicted career path
            feature_names = self.classifier.vectorizer.get_feature_names_out()
            tfidf_scores = text_vectorized.toarray()[0]
            
            # Get feature weights from the Naive Bayes classifier
            feature_weights = self.classifier.classifier.feature_log_prob_
            mean_weights = np.mean(feature_weights, axis=0)
            
            # Get non-zero features (keywords present in resume)
            non_zero_indices = np.where(tfidf_scores > 0)[0]

            def extract_distinctive_keywords(class_index: int) -> List[Dict[str, float]]:
                class_weights = feature_weights[class_index]
                distinctiveness = class_weights - mean_weights

                career_distinctive_keywords: List[Dict[str, float]] = []
                for idx in non_zero_indices:
                    if distinctiveness[idx] > 0:  # Only include career-distinctive keywords
                        career_distinctive_keywords.append({
                            "keyword": feature_names[idx],
                            "score": float(distinctiveness[idx]),
                            "tfidf": float(tfidf_scores[idx])
                        })

                return sorted(
                    career_distinctive_keywords,
                    key=lambda x: x["score"],
                    reverse=True
                )

            extracted_keywords_by_path = {}
            total_distinctive_keywords_by_path = {}
            for idx in top_indices[:3]:
                display_name = self.classifier.get_display_name(self.classes[idx])
                full_keywords = extract_distinctive_keywords(idx)
                extracted_keywords_by_path[display_name] = full_keywords[:12]
                total_distinctive_keywords_by_path[display_name] = len(full_keywords)

            extracted_keywords = extracted_keywords_by_path.get(top_prediction_display, [])
            total_distinctive_keywords = total_distinctive_keywords_by_path.get(top_prediction_display, 0)
            
            return {
                "prediction": top_prediction_display,
                "confidence": float(top_confidence),
                "top_predictions": top_predictions,
                "extracted_keywords": extracted_keywords,
                "extracted_keywords_by_path": extracted_keywords_by_path,
                "total_distinctive_keywords": total_distinctive_keywords,
                "total_distinctive_keywords_by_path": total_distinctive_keywords_by_path
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, resume_texts: List[str], top_n: int = 5) -> List[Dict]:
        """
        Predict career paths for multiple resumes.
        
        Args:
            resume_texts: List of resume texts
            top_n: Number of top predictions to return for each
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text, top_n) for text in resume_texts]
    
    def get_classes(self) -> List[str]:
        """Get list of all possible career paths."""
        return list(self.classes) if self.classes is not None else []