# CareerPath-AI: Resume-Based Career Path Classification System

## 1. Project Overview

CareerPath-AI is a production-grade machine learning system designed to automate the classification of candidate resumes into distinct career paths. By leveraging Natural Language Processing (NLP) and Complement Naive Bayes (CNB) algorithms, the system analyzes resume content to predict the most suitable professional domain with high confidence.

**Key capabilities:**
- Automated parsing of PDF resumes.
- Intelligent handling of class imbalances in career datasets.
- High-precision classification across diverse career domains (e.g., Software Engineering, Data Science, Finance).
- Production-ready REST API and React-based frontend.

This solution significantly reduces manual screening time for recruiters and hiring managers, enabling data-driven talent acquisition strategies.

## 2. System Architecture

The system follows a modular data flow pipeline:

1.  **Input**: PDF Resume Upload (Frontend)
2.  **Extraction**: Text extraction from PDF (Backend API)
3.  **Preprocessing**: Cleaning, normalization, and tokenization
4.  **Vectorization**: TF-IDF transformation
5.  **Inference**: Complement Naive Bayes Model
6.  **Output**: Predicted Career Path with Confidence Scores

## 3. Machine Learning Approach

### Algorithm
The core classification engine uses **Complement Naive Bayes (CNB)**. CNB is chosen specifically for its superior performance on imbalanced datasets compared to standard Multinomial Naive Bayes. It estimates parameters using data from all classes *except* the target class, mitigating the bias often seen towards majority classes.

### Feature Engineering
- **Text Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF).
- **N-Grams**: Unigrams and Bigrams (1, 2) to capture context (e.g., "machine learning" vs "machine" and "learning").
- **Vocabulary Size**: Top 15,000 features based on frequency.
- **Preprocessing**:
    - **Normalization**: Lowercasing and synonym mapping (e.g., "swe" → "software engineer").
    - **Cleaning**: Removal of punctuation, URLs, and email addresses.
    - **Tokenization**: Standard word tokenization.
    - **Lemmatization**: Noun and verb lemmatization to reduce inflectional forms.
    - **Stopword Removal**: Filtering of common English stopwords.

### Handling Class Imbalance
The training pipeline employs a strategic balancing approach:
- **Oversampling**: Minority classes (50-150 samples) are oversampled to 150 samples.
- **Downsampling**: Majority classes (>400 samples) are downsampled to 400 samples.
- **Thresholding**: Classes with fewer than 50 samples are excluded to ensure statistical significance.

### Evaluation Metrics
Model performance is evaluated using:
- **Accuracy**: Overall correctness.
- **Weighted Precision, Recall, and F1-Score**: To account for class distribution.
- **Macro F1-Score**: To assess performance across all classes equally.
- **Confusion Matrix**: Visual analysis of misclassifications.

## 4. Dataset Structure

The system expects a CSV dataset for training with the following schema:

| Column | Description |
| :--- | :--- |
| `text_needed` | The raw text content of the resume. |
| `path` | The target label (career path), e.g., "Software Engineer", "Data Scientist". |

**Constraints:**
- Rows with missing values in required columns are dropped.
- Minimum data requirements per class apply (see "Handling Class Imbalance").

## 5. Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Install Python dependencies
pip install -r ../requirements.txt

# 3. Download required NLTK data (handled automatically on first run, or manually)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Frontend Setup

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies
npm install
```

## 6. Usage Guide

### Running the Application

**Backend API:**
Start the FastAPI server (default: `http://localhost:8000`):

```bash
# From the root directory
python backend/main.py
```

**Frontend Interface:**
Launch the React development server (default: `http://localhost:5173`):

```bash
# From the frontend directory
npm run dev
```

### Model Training
To retrain the model with new data:

```bash
python backend/app/models/naive_bayes/train_model_advanced.py
```
*Ensure your dataset is placed at `backend/data/datasets/` and configured correctly in the script.*

## 7. Model Performance

The current production model achieves the following benchmarks:

- **Accuracy**: 76-80% (on held-out test set)
- **Weighted F1-Score**: ~0.78
- **Macro F1-Score**: ~0.75

*Note: Performance varies by class support. Detailed evaluation metrics, including the confusion matrix, are generated in `backend/data/trained_models/` after training.*

## 8. Limitations and Known Issues

- **PDF Only**: The system currently accepts only PDF files.
- **Text Extraction**: Scanned PDFs (images) are not supported; the file must contain selectable text.
- **Language**: Optimized for English language resumes only.
- **Domain Scope**: Predictions are limited to the career paths present in the training dataset.

## 9. Future Improvements

- **Deep Learning**: Implementation of Transformer-based models (BERT/RoBERTa) for contextual embedding.
- **OCR Integration**: Tesseract support for scanned document processing.
- **Multilingual Support**: Training on non-English corpora.
- **Skill Extraction**: granular entity recognition for specific skills (e.g., "Python", "Project Management").

## 10. Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── models/         # ML model definitions and training scripts
│   │   ├── prediction/     # Inference logic
│   │   ├── preprocessing/  # Text cleaning and transformation
│   │   └── routes/         # API endpoints
│   ├── data/               # Datasets and serialized models
│   └── main.py             # API entry point
├── frontend/
│   ├── src/                # React source code
│   └── package.json
├── docs/                   # Documentation and diagrams
└── requirements.txt        # Python dependencies
```

## 11. License

This project is licensed under the MIT License. See the LICENSE file for details.
