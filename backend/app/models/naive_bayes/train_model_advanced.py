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
import sys
import json
from datetime import datetime
import re
import string
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, backend_dir)

# Import NER location removal utilities
from app.preprocessing.ner_location_remover import NERLocationRemover, LocationLeakageValidator

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
        
        # Initialize NER location remover (prevents geographic leakage)
        print("Initializing NER location remover...")
        self.location_remover = NERLocationRemover(placeholder="<LOCATION>")
        print("✓ NER location remover initialized")
        
        # Initialize location leakage validator
        self.leakage_validator = LocationLeakageValidator()
        
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
            'machine learning engineer': 'datascience',
            'ml engineer': 'datascience',
            'ml': 'machine learning',
            'ai engineer': 'datascience',
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
            'data science': 'datascience',
            'data scientist': 'datascience',
            'ds': 'datascience',
            
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
            'account executive': 'sales',
            'sales manager': 'sales',
            'client acquisition': 'sales',
            
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
            'agronomist': 'agriculture',
            'farm manager': 'agriculture',
            'crop specialist': 'agriculture',
            'soil scientist': 'agriculture',
            'livestock': 'agriculture',
            
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
            'management consultant': 'consultant',
            'strategy consultant': 'consultant',
            'advisory services': 'consultant',
            
            # IT
            'it support': 'it-support',
            'it technician': 'it-support',
            'it tech support': 'it-support',
            'it helpdesk': 'it-support',
            'it help desk': 'it-support',
            'it service desk': 'it-support',
            'service desk support': 'it-support',
            'desktop support': 'it-support',
            'helpdesk support': 'it-support',
            'technical support engineer': 'it-support',
            'it specialist': 'it-support',
            'it administrator': 'it-support',
            
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
        
        # Display name mapping for professional output (Career Path focused)
        self.display_names = {
            'quality assurance': 'Quality Assurance & Testing Careers',
            'software engineer': 'Software Development Careers',
            'business analyst': 'Business Analysis Careers',
            'network engineer': 'Network Administration Careers',
            'devops': 'DevOps & Site Reliability Careers',
            'datascience': 'Data Science & AI Careers',
            'cyber security': 'Cybersecurity Careers',
            'mobile app developer': 'Mobile Development Careers',
            'construction': 'Construction Careers',
            'engineering': 'Engineering Careers',
            'healthcare': 'Healthcare Careers',
            'sales': 'Sales Careers',
            'fitness': 'Fitness & Wellness Careers',
            'teacher': 'Education & Teaching Careers',
            'digital-media': 'Digital Media & Marketing Careers',
            'agriculture': 'Agriculture & Agribusiness Careers',
            'hr': 'Human Resources Careers',
            'advocate': 'Law & Legal Services Careers',
            'business-development': 'Business Development Careers',
            'chef': 'Culinary Arts Careers',
            'consultant': 'Consulting & Advisory Careers',
            'it-support': 'IT Support & Services Careers',
            'public-relations': 'Public Relations & Communications Careers',
            'aviation': 'Aviation & Aerospace Careers',
            # Merged classes (Feb 2026)
            'finance-accounting': 'Finance & Accounting Careers',
            'design-creative': 'Design & Creative Careers'
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
        0. Remove geographic entities using NER (PREVENTS LOCATION LEAKAGE)
        1. Convert to string + lowercase
        2. Synonym normalization (longest phrases first)
        3. Remove punctuation
        4. Tokenization
        5. Remove stopwords
        6. Lemmatization
        """

        # STEP 0: Remove geographic entities BEFORE any other processing
        # This prevents location-based bias in the model
        text = str(text)
        text = self.location_remover.remove_locations(text)
        
        # Convert to lowercase
        text = text.lower()

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
    
    def balance_dataset(self, texts, labels, use_augmentation: bool = False):
        """
        Balance the dataset using intelligent oversampling and downsampling.
        
        NEW STRATEGY (Feb 2026) - Optimized for macro F1 and minority recall:
        - Remove classes with fewer than 50 samples
        - Oversample classes with 50-149 samples to 200 samples
        - Downsample classes with >400 samples to 400 samples
        - Leave other classes (200-400 samples) unchanged
        - Optionally use text augmentation for oversampling
        
        Args:
            texts: Series or list of text samples
            labels: Series or list of labels
            use_augmentation: If True, use synonym replacement for oversampling
                            (back-translation is too slow for training)
            
        Returns:
            Tuple of (balanced_texts, balanced_labels)
        """
        # Configuration thresholds
        MIN_SAMPLES = 50      # Minimum to include class
        TARGET_FLOOR = 200    # Oversample minority classes to this
        TARGET_CEILING = 400  # Downsample dominant classes to this
        
        print("\n" + "="*80)
        print("BALANCING DATASET")
        print("="*80)
        print(f"\nBalancing strategy:")
        print(f"  Floor: {TARGET_FLOOR} samples (oversample minority classes)")
        print(f"  Ceiling: {TARGET_CEILING} samples (downsample dominant classes)")
        print(f"  Remove: classes with <{MIN_SAMPLES} samples")
        print(f"  Augmentation: {'ENABLED (synonym replacement)' if use_augmentation else 'DISABLED (random oversampling)'}")
        
        # Convert to pandas for easier manipulation
        df = pd.DataFrame({'text': texts, 'label': labels})
        
        # Get class distribution
        class_counts = df['label'].value_counts()
        print(f"\nOriginal class distribution:")
        print(f"  Total classes: {len(class_counts)}")
        print(f"  Total samples: {len(df)}")
        print(f"  Class ratio (max/min): {class_counts.max() / class_counts.min():.2f}x")
        
        # Initialize augmenter if needed
        augmenter = None
        if use_augmentation:
            try:
                from app.utils.text_augmenter import TextAugmenter
                augmenter = TextAugmenter(use_gpu=False, back_translation=False)
                print("  Text augmenter initialized (synonym replacement mode)")
            except ImportError as e:
                print(f"  Warning: Could not load augmenter ({e}). Using random oversampling.")
                use_augmentation = False
        
        balanced_dfs = []
        stats = {
            'removed': [],       # List of (class_name, original_count)
            'oversampled': [],   # List of (class_name, original_count, new_count)
            'downsampled': [],   # List of (class_name, original_count, new_count)
            'unchanged': []      # List of (class_name, count)
        }
        
        for label, count in class_counts.items():
            class_df = df[df['label'] == label]
            
            if count < MIN_SAMPLES:
                # Remove classes with fewer than MIN_SAMPLES samples
                print(f"  {label}: {count} → REMOVED (< {MIN_SAMPLES})")
                stats['removed'].append((label, count))
                continue
                
            elif count < TARGET_FLOOR:
                # Oversample to TARGET_FLOOR samples
                if use_augmentation and augmenter:
                    # Use augmentation for diversity
                    original_texts = class_df['text'].tolist()
                    augmented_texts = augmenter.augment_batch(
                        original_texts, 
                        target_count=TARGET_FLOOR,
                        technique='synonym',
                        max_augment_ratio=4.0
                    )
                    augmented_df = pd.DataFrame({
                        'text': augmented_texts,
                        'label': [label] * len(augmented_texts)
                    })
                    balanced_dfs.append(augmented_df)
                    print(f"  {label}: {count} → AUGMENTED to {TARGET_FLOOR}")
                else:
                    # Random oversampling (with replacement)
                    oversampled = class_df.sample(n=TARGET_FLOOR, replace=True, random_state=42)
                    balanced_dfs.append(oversampled)
                    print(f"  {label}: {count} → OVERSAMPLED to {TARGET_FLOOR}")
                stats['oversampled'].append((label, count, TARGET_FLOOR))
                
            elif count > TARGET_CEILING:
                # Downsample to TARGET_CEILING samples
                downsampled = class_df.sample(n=TARGET_CEILING, replace=False, random_state=42)
                balanced_dfs.append(downsampled)
                print(f"  {label}: {count} → DOWNSAMPLED to {TARGET_CEILING}")
                stats['downsampled'].append((label, count, TARGET_CEILING))
                
            else:
                # Keep as is (TARGET_FLOOR to TARGET_CEILING samples)
                balanced_dfs.append(class_df)
                print(f"  {label}: {count} → UNCHANGED")
                stats['unchanged'].append((label, count))
        
        # Concatenate all balanced classes
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        new_class_counts = balanced_df['label'].value_counts().sort_values(ascending=False)
        new_ratio = new_class_counts.max() / new_class_counts.min()
        
        print(f"\nBalanced dataset:")
        print(f"  Total classes: {balanced_df['label'].nunique()}")
        print(f"  Total samples: {len(balanced_df)}")
        print(f"  Min samples per class: {new_class_counts.min()}")
        print(f"  Max samples per class: {new_class_counts.max()}")
        print(f"  Mean samples per class: {new_class_counts.mean():.1f}")
        print(f"  New class ratio: {new_ratio:.2f}x (was {class_counts.max() / class_counts.min():.2f}x)")
        print(f"\nBalancing summary:")
        print(f"  Classes oversampled: {len(stats['oversampled'])}")
        print(f"  Classes downsampled: {len(stats['downsampled'])}")
        print(f"  Classes unchanged: {len(stats['unchanged'])}")
        print(f"  Classes removed: {len(stats['removed'])}")
        
        # Save balanced class distribution and stats for reference
        self.balanced_class_counts = new_class_counts
        self.balancing_stats = stats  # Save for export
        
        return balanced_df['text'], balanced_df['label']
    
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """
        Prepare data for training with CORRECT pipeline order:
        1. Split data into train/test FIRST (test set never touched)
        2. Balance/oversample ONLY the training set
        3. Fit vectorizer on balanced training set
        
        Args:
            texts: Preprocessed text samples
            labels: Career path labels
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, X_test_text)
            - X_test_text is included for saving raw test data for calibration
        """
        print("\n" + "="*80)
        print("PREPARING DATA (Split → Balance Train Only → Vectorize)")
        print("="*80)
        
        # STEP 1: Split data FIRST (stratified to maintain class distribution)
        print("\n[STEP 1] Splitting data into train/test sets...")
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        print(f"  Training set size (before balancing): {len(X_train_text)} ({(1-test_size)*100:.0f}%)")
        print(f"  Test set size (untouched): {len(X_test_text)} ({test_size*100:.0f}%)")
        
        # STEP 2: Balance ONLY the training set (test set remains untouched!)
        # Using augmentation for minority classes to improve recall
        print("\n[STEP 2] Balancing ONLY the training set...")
        X_train_text_balanced, y_train_balanced = self.balance_dataset(
            X_train_text, y_train, use_augmentation=False
        )
        
        print(f"\n  Training set size (after balancing): {len(X_train_text_balanced)}")
        print(f"  Test set remains: {len(X_test_text)} (never oversampled!)")
        
        # STEP 3: Fit vectorizer on BALANCED training set
        print("\n[STEP 3] Vectorizing text using TF-IDF...")
        print(f"  Max features: {self.vectorizer.max_features}")
        print(f"  N-gram range: {self.vectorizer.ngram_range}")
        
        # Fit on balanced training data, transform both
        X_train = self.vectorizer.fit_transform(X_train_text_balanced)
        X_test = self.vectorizer.transform(X_test_text)
        
        print(f"  Training feature matrix shape: {X_train.shape}")
        print(f"  Test feature matrix shape: {X_test.shape}")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Store classes for later use (from original labels, not balanced)
        self.classes_ = np.unique(labels)
        print(f"  Number of classes: {len(self.classes_)}")
        
        # Update y_train to the balanced version
        y_train = y_train_balanced
        
        return X_train, X_test, y_train, y_test, X_test_text
    
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
    # Using merged dataset (26 classes) - finance/accountant/banking merged, arts/designer/apparel merged
    data_path = os.path.join(base_dir, 'data', 'datasets', 'merged_dataset_careerpath-ai_preprocessed.csv')
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
    
    # Step 2: Prepare data (Split FIRST → Balance Train Only → Vectorize)
    # NOTE: Balancing is now done INSIDE prepare_data, only on training set!
    print("\n" + "-"*80)
    print("STEP 2: Preparing Data (Split → Balance Train Only → Vectorize)")
    print("-"*80)
    X_train, X_test, y_train, y_test, X_test_text = classifier.prepare_data(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Save class distribution used for training (after balancing)
    class_dist_path = os.path.join(model_dir, 'training_class_distribution.txt')
    with open(class_dist_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING CLASS DISTRIBUTION (After Balancing - Train Set Only)\n")
        f.write("="*80 + "\n")
        f.write(f"\nDataset: merged_dataset_careerpath-ai_preprocessed.csv\n")
        f.write(f"Pipeline: Split → Balance Train Only → Vectorize\n")
        f.write(f"Total Classes Used: {len(classifier.balanced_class_counts) if hasattr(classifier, 'balanced_class_counts') else y_train.nunique() if hasattr(y_train, 'nunique') else len(set(y_train))}\n")
        f.write(f"Total Training Samples (after balancing): {len(y_train)}\n")
        f.write(f"Total Test Samples (untouched): {len(y_test)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Write balancing strategy info
        if hasattr(classifier, 'balancing_stats'):
            stats = classifier.balancing_stats
            
            # Removed classes section
            f.write("\n" + "="*80 + "\n")
            f.write(f"REMOVED CLASSES ({len(stats['removed'])} classes with <50 samples)\n")
            f.write("="*80 + "\n")
            if stats['removed']:
                for class_name, count in stats['removed']:
                    f.write(f"  ✗ {class_name}: {count} samples\n")
            else:
                f.write("  (None - all classes met minimum threshold)\n")
            
            # Oversampled classes section
            f.write("\n" + "="*80 + "\n")
            f.write(f"OVERSAMPLED CLASSES ({len(stats['oversampled'])} classes boosted to 200)\n")
            f.write("="*80 + "\n")
            if stats['oversampled']:
                for class_name, orig, new in stats['oversampled']:
                    f.write(f"  ↑ {class_name}: {orig} → {new} samples\n")
            else:
                f.write("  (None)\n")
            
            # Downsampled classes section
            f.write("\n" + "="*80 + "\n")
            f.write(f"DOWNSAMPLED CLASSES ({len(stats['downsampled'])} classes reduced to 400)\n")
            f.write("="*80 + "\n")
            if stats['downsampled']:
                for class_name, orig, new in stats['downsampled']:
                    f.write(f"  ↓ {class_name}: {orig} → {new} samples\n")
            else:
                f.write("  (None)\n")
            
            # Unchanged classes section
            f.write("\n" + "="*80 + "\n")
            f.write(f"UNCHANGED CLASSES ({len(stats['unchanged'])} classes with 200-400 samples)\n")
            f.write("="*80 + "\n")
            if stats['unchanged']:
                for class_name, count in stats['unchanged']:
                    f.write(f"  = {class_name}: {count} samples\n")
            else:
                f.write("  (None)\n")
        
        # Final balanced class distribution
        f.write("\n" + "="*80 + "\n")
        f.write("FINAL BALANCED CLASS DISTRIBUTION\n")
        f.write("="*80 + "\n")
        f.write("CLASS NAME                                      | SAMPLE COUNT\n")
        f.write("-"*80 + "\n")
        
        if hasattr(classifier, 'balanced_class_counts'):
            for idx, (class_name, count) in enumerate(classifier.balanced_class_counts.items(), 1):
                f.write(f"{idx:3d}. {class_name:<45} | {count:>6d}\n")
        else:
            if hasattr(y_train, 'value_counts'):
                class_counts = y_train.value_counts().sort_values(ascending=False)
            else:
                from collections import Counter
                class_counts = pd.Series(Counter(y_train)).sort_values(ascending=False)
            for idx, (class_name, count) in enumerate(class_counts.items(), 1):
                f.write(f"{idx:3d}. {class_name:<45} | {count:>6d}\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Training class distribution saved to: {class_dist_path}")
    
    # Save test split data for calibration script
    test_data_path = os.path.join(model_dir, 'test_split_data.pkl')
    test_data = {
        'X_test_text': X_test_text.tolist() if hasattr(X_test_text, 'tolist') else list(X_test_text),
        'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
    }
    joblib.dump(test_data, test_data_path)
    print(f"✓ Test split data saved to: {test_data_path}")
    
    # Step 3: Train model
    print("\n" + "-"*80)
    print("STEP 3: Training Model")
    print("-"*80)
    classifier.train(X_train, y_train, use_sample_weight=False)
    
    # Step 3.5: VALIDATE NO GEOGRAPHIC LEAKAGE (Critical!)
    print("\n" + "-"*80)
    print("STEP 3.5: Validating Geographic Leakage Prevention")
    print("-"*80)
    print("Checking vocabulary for location-related terms...")
    
    try:
        # Assert no leakage - will raise exception if any location terms found
        classifier.leakage_validator.assert_no_leakage(
            classifier.vectorizer.vocabulary_,
            fail_on_leakage=True  # Abort training if leakage detected
        )
        
        # Additional feature name check
        feature_names = classifier.vectorizer.get_feature_names_out()
        is_valid, leaked_features = classifier.leakage_validator.validate_features(feature_names)
        
        if not is_valid:
            print(f"\n{'='*80}")
            print(f"WARNING: {len(leaked_features)} location terms found in features!")
            print(f"{'='*80}")
            print(f"Leaked features: {', '.join(sorted(leaked_features)[:20])}")
            if len(leaked_features) > 20:
                print(f"... and {len(leaked_features) - 20} more")
            print(f"{'='*80}\n")
            raise AssertionError(
                f"Training aborted: {len(leaked_features)} location terms found in features. "
                f"Geographic leakage must be eliminated before deployment."
            )
        
        print("✓ No geographic leakage detected - validation PASSED")
        
    except AssertionError as e:
        print(f"\n❌ TRAINING ABORTED: {str(e)}")
        print("Please investigate and fix location leakage before proceeding.")
        raise
    
    # Step 4: Evaluate model
    print("\n" + "-"*80)
    print("STEP 4: Evaluating Model")
    print("-"*80)
    confusion_matrix_path = os.path.join(model_dir, 'confusion_matrix.png')
    results = classifier.evaluate(X_test, y_test, save_confusion_matrix=confusion_matrix_path)
    
    # Step 5: Save model
    print("\n" + "-"*80)
    print("STEP 5: Saving Model")
    print("-"*80)
    classifier.save_model(model_dir, 'advanced_career_path_cnb')
    
    # Save evaluation results
    results_path = os.path.join(model_dir, 'advanced_model_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Evaluation results saved to: {results_path}")
    
    # Step 6: Test predictions
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
