"""
Extract Top Keywords Per Career Class

This script extracts the most important words/features that the model
uses to classify each career path. It analyzes the TF-IDF vectorizer
and Naive Bayes classifier to find features with the highest weights
for each class.

Usage:
    cd backend
    python scripts/extract_class_keywords.py

Output:
    - Prints top keywords per class to console
    - Saves to data/trained_models/class_keywords.json
"""

import os
import sys
import json
import numpy as np

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.insert(0, backend_dir)

from app.prediction.predictor import CareerPathPredictor


def extract_class_keywords(predictor, top_n: int = 20) -> dict:
    """
    Extract top keywords for each career class based on Naive Bayes feature weights.
    
    For Complement Naive Bayes, we find features that are MOST DISTINCTIVE for each class
    by comparing the class weight to the mean weight across all classes.
    Features with the LARGEST POSITIVE difference (class weight - mean weight) are
    the most indicative of that class (they appear less in complement classes).
    
    Args:
        predictor: Loaded CareerPathPredictor instance
        top_n: Number of top keywords to extract per class
        
    Returns:
        Dictionary mapping career display names to their top keywords
    """
    classifier = predictor.classifier
    
    # Get feature names from vectorizer
    feature_names = classifier.vectorizer.get_feature_names_out()
    
    # Get the Naive Bayes classifier
    nb_classifier = classifier.classifier
    
    # For ComplementNB, feature_log_prob_ contains log probabilities
    # Shape: (n_classes, n_features)
    feature_weights = nb_classifier.feature_log_prob_
    
    # Calculate mean weight for each feature across all classes
    mean_weights = np.mean(feature_weights, axis=0)
    
    # Get class names
    classes = nb_classifier.classes_
    
    results = {}
    
    print("=" * 80)
    print("EXTRACTING TOP KEYWORDS PER CAREER CLASS")
    print("=" * 80)
    print(f"Total classes: {len(classes)}")
    print(f"Total features: {len(feature_names)}")
    print(f"Top keywords per class: {top_n}")
    print("=" * 80)
    
    for idx, class_name in enumerate(classes):
        # Get display name
        display_name = classifier.get_display_name(class_name)
        
        # Get feature weights for this class
        class_weights = feature_weights[idx]
        
        # Calculate distinctiveness: how different is this class's weight from the mean?
        # For CNB: higher weight = feature appears LESS in complement (other classes)
        # So higher weight relative to mean = more distinctive to THIS class
        distinctiveness = class_weights - mean_weights
        
        # Get indices of most distinctive features (highest positive difference)
        top_indices = np.argsort(distinctiveness)[-top_n:][::-1]
        
        # Extract keywords with their distinctiveness scores
        keywords = []
        for i in top_indices:
            keywords.append({
                "keyword": feature_names[i],
                "distinctiveness": float(distinctiveness[i]),
                "weight": float(class_weights[i])
            })
        
        results[display_name] = {
            "internal_name": class_name,
            "top_keywords": keywords
        }
        
        # Print to console
        print(f"\n{display_name}")
        print("-" * 60)
        keyword_list = [k["keyword"] for k in keywords[:10]]
        print(f"  {', '.join(keyword_list)}")
    
    return results


def main():
    print("\n" + "=" * 80)
    print("CAREER PATH KEYWORD EXTRACTION TOOL")
    print("=" * 80)
    
    # Initialize predictor (loads model)
    print("\nLoading trained model...")
    predictor = CareerPathPredictor()
    
    if not predictor.is_loaded():
        print("ERROR: Model not loaded. Please train the model first.")
        sys.exit(1)
    
    print(f"✓ Model loaded with {len(predictor.classes)} career paths")
    
    # Extract keywords
    results = extract_class_keywords(predictor, top_n=20)
    
    # Save to JSON
    output_dir = os.path.join(backend_dir, 'data', 'trained_models')
    output_path = os.path.join(output_dir, 'class_keywords.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"✓ Keywords saved to: {output_path}")
    print(f"✓ Total classes: {len(results)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
