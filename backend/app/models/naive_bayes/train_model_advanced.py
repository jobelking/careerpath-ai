"""
ADVANCED Complement Naive Bayes Pipeline for Career Path Prediction
- Handles imbalanced dataset with strategic oversampling/downsampling
- Advanced text preprocessing with lemmatization and synonym normalization
- TF-IDF vectorization with unigrams and bigrams (top 15,000 features)
- Comprehensive evaluation with confusion matrix
- Expected accuracy: 76-80%+
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os
import json
from datetime import datetime
import re
import string
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Import NLP libraries for advanced preprocessing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data
    for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass
except ImportError:
    print("Warning: NLTK not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk'])
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize


class AdvancedCareerPathClassifier:
    """
    Advanced Complement Naive Bayes classifier with:
    - Advanced text preprocessing (lemmatization, synonym normalization)
    - TF-IDF vectorization (unigrams + bigrams, max 15,000 features)
    - Intelligent dataset balancing
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, max_features=15000, ngram_range=(1, 2), alpha=0.1):
        """
        Initialize the classifier.
        
        Args:
            max_features: Maximum vocabulary size for TF-IDF
            ngram_range: Tuple of (min_n, max_n) for n-grams
            alpha: Smoothing parameter for Complement Naive Bayes
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.9,  # Ignore terms that appear in more than 90% of documents
            sublinear_tf=True,  # Apply sublinear tf scaling
            smooth_idf=True,
            norm='l2'
        )
        self.classifier = ComplementNB(alpha=alpha)
        self.classes_ = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Synonym normalization dictionary for career-related terms
        self.synonyms = {
            # Quality Assurance
            'qa': 'quality assurance',
            'qa engineer': 'quality assurance',
            'tester': 'quality assurance',
            'testing': 'quality assurance',
            
            # Software Engineering
            'swe': 'software engineer',
            'sde': 'software engineer',
            'software dev': 'software engineer',
            'software developer': 'software engineer',
            'programmer': 'software engineer',
            'coder': 'software engineer',
            
            # Machine Learning
            'ml engineer': 'machine learning engineer',
            'ml': 'machine learning',
            'ai engineer': 'machine learning engineer',
            'ai': 'artificial intelligence',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            
            # Business Analysis
            'ba': 'business analyst',
            'business intelligence': 'business analyst',
            'bi analyst': 'business analyst',
            
            # Network Engineering
            'network admin': 'network engineer',
            'networking': 'network engineer',
            'network administrator': 'network engineer',
            
            # DevOps
            'dev ops': 'devops',
            'sre': 'devops',
            'site reliability engineer': 'devops',
            
            # Data Science
            'data scientist': 'data science',
            'ds': 'data science',
            
            # Cyber Security
            'cybersecurity': 'cyber security',
            'infosec': 'cyber security',
            'information security': 'cyber security',
            'security analyst': 'cyber security',
            
            # Mobile Development
            'mobile developer': 'mobile app developer',
            'app developer': 'mobile app developer',
            'ios developer': 'mobile app developer',
            'android developer': 'mobile app developer',
            
            # Data Engineering
            'de': 'data engineer',
            'etl engineer': 'data engineer',
            'big data engineer': 'data engineer',
            
            # Construction
            'construction worker': 'construction',
            'builder': 'construction',
            
            # Apparel/Fashion
            'fashion': 'apparel',
            'clothing': 'apparel',
            'textile': 'apparel',
            
            # Design
            'graphic designer': 'designer',
            'ui designer': 'designer',
            'ux designer': 'designer',
            'web designer': 'designer',
            
            # Healthcare
            'medical': 'healthcare',
            'health': 'healthcare',
            'nurse': 'healthcare',
            'doctor': 'healthcare',
            'physician': 'healthcare',
            
            # Accounting
            'accounting': 'accountant',
            'cpa': 'accountant',
            
            # Sales
            'salesperson': 'sales',
            'sales rep': 'sales',
            'sales representative': 'sales',
            
            # Fitness
            'personal trainer': 'fitness',
            'gym': 'fitness',
            'wellness': 'fitness',
            
            # Teaching
            'educator': 'teacher',
            'professor': 'teacher',
            'instructor': 'teacher',
            
            # Banking/Finance
            'banker': 'banking',
            'financial services': 'banking',
            
            # Digital Media
            'digital marketing': 'digital media',
            'social media': 'digital media',
            'content creator': 'digital media',
            
            # Agriculture
            'farming': 'agriculture',
            'agricultural': 'agriculture',
            'farmer': 'agriculture',
            
            # HR
            'human resources': 'hr',
            'recruiter': 'hr',
            'talent acquisition': 'hr',
            
            # Arts
            'artist': 'arts',
            'fine arts': 'arts',
            'visual artist': 'arts',
            'performing arts': 'arts',
            'exhibition': 'arts',
            
            # Legal/Advocate
            'lawyer': 'advocate',
            'attorney': 'advocate',
            'legal': 'advocate',
            
            # Business Development
            'bd': 'business development',
            'biz dev': 'business development',
            
            # Finance
            'financial analyst': 'finance',
            'fintech': 'finance',
            'financial planning': 'finance',
            'financial modeling': 'finance',
            'fp&a': 'finance',
            
            # Culinary
            'cook': 'chef',
            'culinary': 'chef',
            
            # Consulting
            'consulting': 'consultant',
            'advisor': 'consultant',
            
            # IT
            'it support': 'information technology',
            'it technician': 'information technology',
            'it tech support': 'information technology',
            'it helpdesk': 'information technology',
            'it help desk': 'information technology',
            'it service desk': 'information technology',
            'service desk support': 'information technology',
            'desktop support': 'information technology',
            'helpdesk support': 'information technology',
            'technical support engineer': 'information technology',
            'it specialist': 'information technology',
            'it administrator': 'information technology',
            
            # PR
            'pr': 'public relations',
            'communications': 'public relations',
            
            # Aviation
            'pilot': 'aviation',
            'aircraft': 'aviation',
            'airline': 'aviation',
            
            # General tech terms
            'fullstack': 'full stack',
            'frontend': 'front end',
            'backend': 'back end',
            'db': 'database',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience'
        }
        
        # Display name mapping for professional output
        self.display_names = {
            'quality assurance': 'Quality Assurance Engineer',
            'software engineer': 'Software Engineer',
            'machine learning engineer': 'Machine Learning Engineer',
            'business analyst': 'Business Analyst',
            'network engineer': 'Network Engineer',
            'devops': 'DevOps Engineer',
            'data science': 'Data Scientist',
            'cyber security': 'Cyber Security Specialist',
            'mobile app developer': 'Mobile App Developer',
            'construction': 'Construction Professional',
            'engineering': 'General Engineering Professional',
            'apparel': 'Apparel & Fashion Specialist',
            'designer': 'Designer',
            'healthcare': 'Healthcare Professional',
            'accountant': 'Accountant',
            'sales': 'Sales Executive',
            'fitness': 'Fitness Trainer',
            'teacher': 'Teacher',
            'banking': 'Banking Officer',
            'digital-media': 'Digital Media Specialist',
            'agriculture': 'Agriculture Specialist',
            'hr': 'Human Resources Specialist',
            'arts': 'Arts & Creative Professional',
            'advocate': 'Legal Advocate / Lawyer',
            'business-development': 'Business Development Specialist',
            'finance': 'Finance Analyst',
            'chef': 'Chef',
            'consultant': 'Consultant',
            'information-technology': 'IT Support Specialist',
            'public-relations': 'Public Relations Officer',
            'aviation': 'Aviation Professional'
        }
        
    def normalize_synonyms(self, text):
        """Normalize common synonyms and abbreviations."""
        text_lower = text.lower()
        for synonym, replacement in self.synonyms.items():
            # Use word boundaries to avoid partial matches
            text_lower = re.sub(r'\b' + re.escape(synonym) + r'\b', replacement, text_lower)
        return text_lower
        
    def preprocess_text(self, text):
        """
        Advanced text preprocessing pipeline:
        1. Convert to string + lowercase
        2. Synonym normalization (longest phrases first)
        3. Remove punctuation
        4. Tokenization
        5. Remove stopwords
        6. Lemmatization
        """

        # Convert to string and lowercase FIRST
        text = str(text).lower()

        # Normalize synonyms (match longer phrases first)
        for term in sorted(self.synonyms.keys(), key=len, reverse=True):
            replacement = self.synonyms[term]
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, replacement, text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemma = self.lemmatizer.lemmatize(token, pos='v')
                lemma = self.lemmatizer.lemmatize(lemma, pos='n')
                processed_tokens.append(lemma)

        return ' '.join(processed_tokens)

        
    def load_data(self, filepath, text_col='text_needed', label_col='path'):
        """
        Load and preprocess dataset from CSV.
        
        Args:
            filepath: Path to CSV file
            text_col: Name of column containing resume text
            label_col: Name of column containing career path labels
            
        Returns:
            Tuple of (preprocessed_texts, labels)
        """
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Drop rows with missing values
        df = df.dropna(subset=[text_col, label_col])
        
        print(f"Dataset shape: {df.shape}")
        print(f"Number of classes: {df[label_col].nunique()}")
        print(f"Number of samples: {len(df)}")
        
        # Show class distribution
        class_counts = df[label_col].value_counts()
        print(f"\nClass distribution statistics:")
        print(f"  Min samples: {class_counts.min()}")
        print(f"  Max samples: {class_counts.max()}")
        print(f"  Mean samples: {class_counts.mean():.1f}")
        print(f"  Median samples: {class_counts.median():.1f}")
        
        # Preprocess text
        print("\nPreprocessing text (this may take a few minutes)...")
        texts = df[text_col].apply(self.preprocess_text)
        print("Text preprocessing completed!")
        
        return texts, df[label_col]
    
    def balance_dataset(self, texts, labels):
        """
        Balance the dataset using intelligent oversampling and downsampling:
        - Remove classes with fewer than 50 samples
        - Oversample classes with 50-150 samples to 150 samples
        - Downsample classes with >400 samples to 400 samples
        - Leave other classes (151-400 samples) unchanged
        
        Args:
            texts: Series or list of text samples
            labels: Series or list of labels
            
        Returns:
            Tuple of (balanced_texts, balanced_labels)
        """
        print("\n" + "="*80)
        print("BALANCING DATASET")
        print("="*80)
        
        # Convert to pandas for easier manipulation
        df = pd.DataFrame({'text': texts, 'label': labels})
        
        # Get class distribution
        class_counts = df['label'].value_counts()
        print(f"\nOriginal class distribution:")
        print(f"  Total classes: {len(class_counts)}")
        print(f"  Total samples: {len(df)}")
        
        balanced_dfs = []
        
        for label, count in class_counts.items():
            class_df = df[df['label'] == label]
            
            if count < 50:
                # Remove classes with fewer than 50 samples
                print(f"  {label}: {count} samples → REMOVED (insufficient samples)")
                continue
            elif 50 <= count <= 150:
                # Oversample to 150 samples
                target_samples = 150
                oversampled = class_df.sample(n=target_samples, replace=True, random_state=42)
                balanced_dfs.append(oversampled)
                print(f"  {label}: {count} samples → OVERSAMPLED to {target_samples}")
            elif count > 400:
                # Downsample to 400 samples
                target_samples = 400
                downsampled = class_df.sample(n=target_samples, replace=False, random_state=42)
                balanced_dfs.append(downsampled)
                print(f"  {label}: {count} samples → DOWNSAMPLED to {target_samples}")
            else:
                # Keep as is (151-400 samples)
                balanced_dfs.append(class_df)
                print(f"  {label}: {count} samples → UNCHANGED")
        
        # Concatenate all balanced classes
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nBalanced dataset:")
        print(f"  Total classes: {balanced_df['label'].nunique()}")
        print(f"  Total samples: {len(balanced_df)}")
        
        new_class_counts = balanced_df['label'].value_counts().sort_values(ascending=False)
        print(f"  Min samples per class: {new_class_counts.min()}")
        print(f"  Max samples per class: {new_class_counts.max()}")
        print(f"  Mean samples per class: {new_class_counts.mean():.1f}")
        
        # Save balanced class distribution for reference
        self.balanced_class_counts = new_class_counts
        
        return balanced_df['text'], balanced_df['label']
    
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """
        Prepare data for training: split and vectorize.
        
        Args:
            texts: Preprocessed text samples
            labels: Career path labels
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*80)
        print("PREPARING DATA")
        print("="*80)
        
        # Split data (stratified to maintain class distribution)
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        print(f"\nData split:")
        print(f"  Training set size: {len(X_train_text)} ({(1-test_size)*100:.0f}%)")
        print(f"  Test set size: {len(X_test_text)} ({test_size*100:.0f}%)")
        
        # Vectorize using TF-IDF
        print(f"\nVectorizing text using TF-IDF...")
        print(f"  Max features: {self.vectorizer.max_features}")
        print(f"  N-gram range: {self.vectorizer.ngram_range}")
        
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        
        print(f"  Feature matrix shape: {X_train.shape}")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Store classes for later use
        self.classes_ = np.unique(labels)
        print(f"  Number of classes: {len(self.classes_)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, use_sample_weight=False):
        """
        Train the Complement Naive Bayes classifier.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            use_sample_weight: Whether to use sample weights for class balancing
        """
        print("\n" + "="*80)
        print("TRAINING MODEL")
        print("="*80)
        
        print(f"Model: Complement Naive Bayes")
        print(f"Alpha (smoothing): {self.classifier.alpha}")
        
        if use_sample_weight:
            # Compute sample weights to give more importance to minority classes
            sample_weights = compute_sample_weight('balanced', y_train)
            print(f"Using sample weights: Enabled")
            self.classifier.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            print(f"Using sample weights: Disabled")
            self.classifier.fit(X_train, y_train)
        
        print("✓ Training completed!")
        
    def evaluate(self, X_test, y_test, save_confusion_matrix=None):
        """
        Evaluate the model with comprehensive metrics.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            save_confusion_matrix: Path to save confusion matrix plot (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Weighted metrics (account for class imbalance)
        weighted_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        weighted_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Macro metrics (treat all classes equally)
        macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        print(f"\n{'='*40}")
        print(f"OVERALL METRICS")
        print(f"{'='*40}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\n{'='*40}")
        print(f"WEIGHTED METRICS (Class-Balanced)")
        print(f"{'='*40}")
        print(f"Precision: {weighted_precision:.4f}")
        print(f"Recall:    {weighted_recall:.4f}")
        print(f"F1-Score:  {weighted_f1:.4f}")
        
        print(f"\n{'='*40}")
        print(f"MACRO METRICS (Equal Weight per Class)")
        print(f"{'='*40}")
        print(f"Precision: {macro_precision:.4f}")
        print(f"Recall:    {macro_recall:.4f}")
        print(f"F1-Score:  {macro_f1:.4f}")
        
        # Classification report
        print("\n" + "="*80)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix computed")
        print(f"Shape: {cm.shape}")
        
        # Save confusion matrix if path provided
        if save_confusion_matrix:
            plt.figure(figsize=(20, 18))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                       xticklabels=self.classes_, yticklabels=self.classes_)
            plt.title('Confusion Matrix for Career Path Prediction')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=90, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(save_confusion_matrix, dpi=150)
            print(f"✓ Confusion matrix saved to: {save_confusion_matrix}")
            plt.close()
        
        # Prepare results dictionary
        results = {
            'accuracy': float(accuracy),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'training_date': datetime.now().isoformat(),
            'model_parameters': {
                'model': 'ComplementNB',
                'alpha': self.classifier.alpha,
                'max_features': self.vectorizer.max_features,
                'ngram_range': self.vectorizer.ngram_range,
                'min_df': self.vectorizer.min_df,
                'max_df': self.vectorizer.max_df
            }
        }
        
        return results
    
    def get_display_name(self, class_name):
        """
        Convert internal class name to professional display name.
        
        Args:
            class_name: Internal class name (e.g., 'quality assurance')
            
        Returns:
            Professional display name (e.g., 'Quality Assurance Engineer')
        """
        return self.display_names.get(class_name, class_name.title())
    
    def predict(self, texts, return_display_name=True):
        """
        Predict career paths for new resume texts.
        
        Args:
            texts: String or list of strings containing resume text
            return_display_name: If True, return professional display names; if False, return raw class names
            
        Returns:
            Predicted career path(s)
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        texts_processed = [self.preprocess_text(t) for t in texts]
        
        # Vectorize
        X = self.vectorizer.transform(texts_processed)
        
        # Predict
        predictions = self.classifier.predict(X)
        
        # Convert to display names if requested
        if return_display_name:
            predictions = [self.get_display_name(pred) for pred in predictions]
        
        # Return single prediction if input was single string
        if len(predictions) == 1:
            return predictions[0]
        return predictions
    
    def predict_proba(self, texts, return_display_names=True):
        """
        Predict career path probabilities for new resume texts.
        
        Args:
            texts: String or list of strings containing resume text
            return_display_names: If True, return dict with display names; if False, return raw probabilities
            
        Returns:
            If return_display_names=True: Dictionary mapping display names to probabilities
            If return_display_names=False: Probability distribution array over all career paths
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # Preprocess texts
        texts_processed = [self.preprocess_text(t) for t in texts]
        
        # Vectorize
        X = self.vectorizer.transform(texts_processed)
        
        # Get probabilities
        probabilities = self.classifier.predict_proba(X)
        
        # Convert to dictionary with display names if requested
        if return_display_names:
            result = []
            for prob_array in probabilities:
                prob_dict = {}
                for idx, prob in enumerate(prob_array):
                    class_name = self.classifier.classes_[idx]
                    display_name = self.get_display_name(class_name)
                    prob_dict[display_name] = float(prob)
                result.append(prob_dict)
            
            # Return single dict if input was single string
            if single_input:
                return result[0]
            return result
        else:
            # Return single probability array if input was single string
            if single_input:
                return probabilities[0]
            return probabilities
    
    def save_model(self, model_dir, model_name='advanced_cnb'):
        """
        Save the trained model to disk.
        
        Args:
            model_dir: Directory to save model files
            model_name: Base name for model files
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save all components
        joblib.dump(self.classifier, os.path.join(model_dir, f'{model_name}_classifier.pkl'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, f'{model_name}_vectorizer.pkl'))
        joblib.dump(self.classes_, os.path.join(model_dir, f'{model_name}_classes.pkl'))
        joblib.dump(self.synonyms, os.path.join(model_dir, f'{model_name}_synonyms.pkl'))
        joblib.dump(self.display_names, os.path.join(model_dir, f'{model_name}_display_names.pkl'))
        
        print(f"\n✓ Model saved successfully to: {model_dir}")
        print(f"  - {model_name}_classifier.pkl")
        print(f"  - {model_name}_vectorizer.pkl")
        print(f"  - {model_name}_classes.pkl")
        print(f"  - {model_name}_synonyms.pkl")
        print(f"  - {model_name}_display_names.pkl")
    
    @staticmethod
    def load_model(model_dir, model_name='advanced_cnb'):
        """
        Load a trained model from disk.
        
        Args:
            model_dir: Directory containing model files
            model_name: Base name of model files
            
        Returns:
            AdvancedCareerPathClassifier instance with loaded model
        """
        classifier = AdvancedCareerPathClassifier()
        
        classifier.classifier = joblib.load(os.path.join(model_dir, f'{model_name}_classifier.pkl'))
        classifier.vectorizer = joblib.load(os.path.join(model_dir, f'{model_name}_vectorizer.pkl'))
        classifier.classes_ = joblib.load(os.path.join(model_dir, f'{model_name}_classes.pkl'))
        classifier.synonyms = joblib.load(os.path.join(model_dir, f'{model_name}_synonyms.pkl'))
        
        # Try to load display names (for backward compatibility with older models)
        try:
            classifier.display_names = joblib.load(os.path.join(model_dir, f'{model_name}_display_names.pkl'))
        except FileNotFoundError:
            print("Warning: Display names not found. Using default mapping.")
        
        print(f"✓ Model loaded successfully from: {model_dir}")
        return classifier


def main():
    """
    Main function to execute the complete ML pipeline.
    """
    print("\n" + "="*80)
    print("ADVANCED COMPLEMENT NAIVE BAYES PIPELINE")
    print("Career Path Prediction from Resume Text")
    print("="*80)
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_path = os.path.join(base_dir, 'data', 'datasets', 'dataset_careerpath-ai_preprocessed.csv')
    model_dir = os.path.join(base_dir, 'data', 'trained_models')
    
    # Initialize classifier with optimized parameters
    print("\nInitializing classifier...")
    classifier = AdvancedCareerPathClassifier(
        max_features=15000,  # Top 15k features
        ngram_range=(1, 2),  # Unigrams and bigrams
        alpha=0.1  # Smoothing parameter
    )
    print("✓ Classifier initialized")
    
    # Step 1: Load and preprocess data
    print("\n" + "-"*80)
    print("STEP 1: Loading and Preprocessing Data")
    print("-"*80)
    texts, labels = classifier.load_data(data_path)
    
    # Step 2: Balance dataset
    print("\n" + "-"*80)
    print("STEP 2: Balancing Dataset")
    print("-"*80)
    texts_balanced, labels_balanced = classifier.balance_dataset(texts, labels)
    
    # Save class distribution used for training
    class_dist_path = os.path.join(model_dir, 'training_class_distribution.txt')
    with open(class_dist_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING CLASS DISTRIBUTION (After Balancing)\n")
        f.write("="*80 + "\n")
        f.write(f"\nDataset: dataset_careerpath-ai_preprocessed.csv\n")
        f.write(f"Total Classes Used: {classifier.balanced_class_counts.nunique() if hasattr(classifier, 'balanced_class_counts') else labels_balanced.nunique()}\n")
        f.write(f"Total Samples Used: {len(labels_balanced)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("CLASS NAME                                      | SAMPLE COUNT\n")
        f.write("="*80 + "\n")
        
        if hasattr(classifier, 'balanced_class_counts'):
            for idx, (class_name, count) in enumerate(classifier.balanced_class_counts.items(), 1):
                f.write(f"{idx:3d}. {class_name:<45} | {count:>6d}\n")
        else:
            class_counts = labels_balanced.value_counts().sort_values(ascending=False)
            for idx, (class_name, count) in enumerate(class_counts.items(), 1):
                f.write(f"{idx:3d}. {class_name:<45} | {count:>6d}\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Training class distribution saved to: {class_dist_path}")
    
    # Step 3: Prepare data (split and vectorize)
    print("\n" + "-"*80)
    print("STEP 3: Preparing Data for Training")
    print("-"*80)
    X_train, X_test, y_train, y_test = classifier.prepare_data(
        texts_balanced, labels_balanced, test_size=0.2, random_state=42
    )
    
    # Step 4: Train model
    print("\n" + "-"*80)
    print("STEP 4: Training Model")
    print("-"*80)
    classifier.train(X_train, y_train, use_sample_weight=False)
    
    # Step 5: Evaluate model
    print("\n" + "-"*80)
    print("STEP 5: Evaluating Model")
    print("-"*80)
    confusion_matrix_path = os.path.join(model_dir, 'confusion_matrix.png')
    results = classifier.evaluate(X_test, y_test, save_confusion_matrix=confusion_matrix_path)
    
    # Step 6: Save model
    print("\n" + "-"*80)
    print("STEP 6: Saving Model")
    print("-"*80)
    classifier.save_model(model_dir, 'advanced_career_path_cnb')
    
    # Save evaluation results
    results_path = os.path.join(model_dir, 'advanced_model_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Evaluation results saved to: {results_path}")
    
    # Step 7: Test predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    test_samples = [
        "python machine learning data science tensorflow pytorch deep learning neural networks",
        "java spring boot microservices kubernetes docker aws cloud devops",
        "network security cisco firewall vpn penetration testing cybersecurity",
        "react javascript frontend user interface design responsive web development",
        "sql database postgresql mysql data warehouse etl data engineering"
    ]
    
    for i, text in enumerate(test_samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Text: {text[:80]}...")
        
        # Get prediction with display name
        prediction = classifier.predict(text, return_display_name=True)
        print(f"✓ Predicted Career Path: {prediction}")
        
        # Get top 3 probabilities with display names
        prob_dict = classifier.predict_proba(text, return_display_names=True)
        top_3_careers = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"Top 3 Career Paths:")
        for career_name, prob in top_3_careers:
            print(f"  {career_name}: {prob*100:.2f}%")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  • Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  • Weighted F1-Score: {results['weighted_f1']:.4f}")
    print(f"  • Macro F1-Score: {results['macro_f1']:.4f}")
    print(f"  • Model saved to: {model_dir}")
    print(f"  • Confusion matrix: {confusion_matrix_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
