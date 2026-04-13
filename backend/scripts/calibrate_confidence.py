
import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.prediction.predictor import CareerPathPredictor

def calibrate_confidence():
    print("="*80)
    print("CONFIDENCE CALIBRATION SCRIPT")
    print("="*80)

    # 1. Load saved test data from training
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_path = os.path.join(base_dir, 'data', 'trained_models', 'test_split_data.pkl')
    
    if not os.path.exists(test_data_path):
        print(f"Error: Test split data not found at {test_data_path}")
        print("Please run the training script first to generate test split data.")
        return

    print(f"Loading test split data from: {test_data_path}")
    test_data = joblib.load(test_data_path)
    
    X_test_text = test_data['X_test_text']
    y_test = test_data['y_test']
    
    print(f"Test samples loaded: {len(X_test_text)}")
    print(f"Unique classes in test set: {len(set(y_test))}")
    
    # 2. Load Model
    print("\nLoading trained model...")
    try:
        predictor = CareerPathPredictor(eager_load=True)
    except Exception as e:
        print(f"Error loading predictor: {e}")
        return

    # 3. Run Predictions
    print("Running predictions on test set...")
    
    results = []
    correct_count = 0
    
    for i, text in enumerate(X_test_text):
        true_label = y_test[i]
        
        try:
            # Use the internal classifier for prediction
            preprocessed_text = predictor.classifier.preprocess_text(text)
            text_vectorized = predictor.classifier.vectorizer.transform([preprocessed_text])
            probabilities = predictor.classifier.classifier.predict_proba(text_vectorized)[0]
            
            top_idx = np.argmax(probabilities)
            top_class = predictor.classes[top_idx]  # raw class name
            confidence = probabilities[top_idx]
            
            is_correct = (top_class == true_label)
            if is_correct:
                correct_count += 1
                
            results.append({
                'confidence': confidence,
                'correct': is_correct,
                'true': true_label,
                'pred': top_class
            })
            
        except Exception as e:
            print(f"Error predicting sample {i}: {e}")
            continue
            
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(X_test_text)}...")

    print(f"\nOverall Accuracy on Test Set: {correct_count/len(X_test_text):.4f}")
    
    # 4. Binning and Analysis
    df_res = pd.DataFrame(results)
    
    # Bins: 0-5%, 5-10%, 10-15%, 15-20%, 20-30%, 30-50%, 50-100%
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    labels_bins = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '50-100%']
    
    df_res['bin'] = pd.cut(df_res['confidence'], bins=bins, labels=labels_bins, include_lowest=True)
    
    print("\n" + "="*80)
    print("CALIBRATION RESULTS")
    print("="*80)
    print(f"{'Confidence Bin':<15} | {'Count':<8} | {'Accuracy':<10} | {'Avg Conf':<10}")
    print("-" * 55)
    
    for bin_label in labels_bins:
        bin_data = df_res[df_res['bin'] == bin_label]
        count = len(bin_data)
        if count > 0:
            accuracy = bin_data['correct'].mean()
            avg_conf = bin_data['confidence'].mean()
            print(f"{bin_label:<15} | {count:<8} | {accuracy:.2%}   | {avg_conf:.2%}")
        else:
            print(f"{bin_label:<15} | {0:<8} | N/A        | N/A")

if __name__ == "__main__":
    calibrate_confidence()
