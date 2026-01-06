# CareerPath-AI: Resume-Based Career Path Classification System

## Overview
CareerPath-AI classifies resume content into likely career paths to help recruiters, hiring managers, and developers screen candidates and route profiles to the right teams. The system ingests PDF resumes, extracts text, applies deterministic preprocessing, and serves probability-ranked career path predictions through a FastAPI service. It is designed for reproducibility, deterministic inference, and transparent model evaluation.

## System Architecture
- **Input:** PDF resume uploaded via API or frontend.
- **Preprocessing:**
  - Text extraction from PDF, validation for size and minimum readable content.
  - Normalization (lowercasing, punctuation stripping), tokenization, stop-word removal, lemmatization, and domain-specific synonym mapping.
- **Model:** TF-IDF vectorization (unigrams + bigrams, max 15,000 features) followed by a Complement Naive Bayes classifier with smoothing (`alpha=0.1`).
- **Output:** JSON response with top-N career paths, normalized confidence scores, and available class list endpoint.

## Machine Learning Approach
- **Algorithm:** Complement Naive Bayes optimized for high-dimensional sparse text classification.
- **Feature Engineering:**
  - TF-IDF with sublinear term frequency scaling, unigrams/bigrams, `min_df=2`, `max_df=0.9` to reduce noise.
  - Lemmatization via NLTK and synonym normalization for role titles (e.g., QA → quality assurance, SWE → software engineer).
- **Handling Class Imbalance:**
  - Training-time class balancing: remove labels with <50 samples, oversample minority classes to 150 samples, downsample majority classes above 400 samples, and shuffle.
  - Sample weighting during model fit using `compute_sample_weight('balanced', y_train)` to retain minority influence after vectorization.
- **Evaluation Metrics:**
  - Overall accuracy plus weighted/macro precision, recall, and F1 to reflect both class-frequency-aware and class-equal performance.
  - Confusion matrix persisted for error analysis; classification report saved as JSON for reproducible benchmarking.

## Dataset Structure and Constraints
- Location: `backend/data/datasets/` contains raw and preprocessed CSVs.
- Schema: `path` (career path label) and `text_needed` (raw or cleaned resume text).
- Constraints:
  - Training pipeline drops classes with <50 samples.
  - Oversampling and downsampling targets 150–400 samples per class; expect skewed raw data but balanced training splits.
  - Input resumes should contain machine-readable text; scanned images without OCR will be rejected.


### Supported Career Paths
The current model ships with the following 31 career path labels, sourced from
`backend/data/trained_models/advanced_career_path_cnb_display_names.pkl`:
- Accountant
- Agriculture Specialist
- Apparel & Fashion Specialist
- Arts & Creative Professional
- Aviation Professional
- Banking Officer
- Business Analyst
- Business Development Specialist
- Chef
- Construction Professional
- Consultant
- Cyber Security Specialist
- Data Scientist
- Designer
- DevOps Engineer
- Digital Media Specialist
- Finance Analyst
- Fitness Trainer
- General Engineering Professional
- Healthcare Professional
- Human Resources Specialist
- IT Support Specialist
- Legal Advocate / Lawyer
- Machine Learning Engineer
- Mobile App Developer
- Network Engineer
- Public Relations Officer
- Quality Assurance Engineer
- Sales Executive
- Software Engineer
- Teacher

## Installation & Setup
```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: ensure NLTK data is available (downloaded on demand during training)
python - <<'PY'
import nltk
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']:
    nltk.download(resource)
PY
```

## Usage
### Run API locally
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
API docs will be available at `http://localhost:8000/docs`.

### Predict via cURL
```bash
curl -X POST \
  -F "file=@/path/to/resume.pdf" \
  http://localhost:8000/api/predict
```
Response includes `prediction`, `confidence`, and `top_predictions` (probability-normalized top N).

### Frontend integration
The `frontend/` directory hosts a Vite-based UI that uploads PDFs to the FastAPI backend (default CORS allows ports 5173 and 3000). Install Node.js 18+, run `npm install`, then `npm run dev` from `frontend/`.

## Model Performance Summary
Latest trained model (`advanced_career_path_cnb`) metrics on held-out test split:
- Accuracy: 81.64%
- Weighted Precision / Recall / F1: 0.83 / 0.82 / 0.81
- Macro Precision / Recall / F1: 0.83 / 0.80 / 0.80
- Notable strengths: high F1 for mobile app developer (~0.95) and quality assurance (~0.93). Improvement areas: information-technology macro recall (~0.48) and consultant class recall (~0.43).

## Limitations and Known Issues
- PDF ingestion assumes text-based documents; image-only resumes require external OCR.
- Model vocabulary and synonym map focus on technology and adjacent domains; atypical or emerging roles may be misclassified.
- Complement Naive Bayes favors sparse TF-IDF features and may underperform on highly context-dependent phrasing compared to transformer models.
- Class list is fixed to training labels; unseen categories are mapped to nearest known role rather than “other”.

## Future Improvements
- Replace or ensemble with transformer encoders (e.g., BERT/RoBERTa) for richer contextual understanding.
- Add automated OCR preprocessing for scanned PDFs.
- Expand labeled datasets, especially underrepresented career paths, and automate data quality checks.
- Implement active learning loop to incorporate recruiter feedback into periodic retraining.

## Project Structure
```
backend/
  app/
    models/naive_bayes/      # Training pipeline and model serialization
    prediction/              # Runtime predictor loading serialized artifacts
    preprocessing/           # Text cleaning utilities
    routes/                  # FastAPI route definitions
    utils/                   # PDF parsing and helpers
  data/
    datasets/                # Raw and preprocessed CSVs
    trained_models/          # Serialized model, vectorizer, class metadata, metrics
  main.py                    # FastAPI entrypoint
frontend/                    # Vite-based frontend
docker/                      # Dockerfiles and compose snippets
requirements.txt             # Python dependencies
```

## License
This repository currently has no published license. Obtain explicit permission before using the code or model artifacts in production or commercial contexts.
