"""
Lightweight inference classifier for CareerPath AI.
This module intentionally avoids importing training-time dependencies.
"""

import os
import re
import string
import joblib

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None
    word_tokenize = None


class InferenceCareerPathClassifier:
    """Inference-only classifier wrapper with lightweight preprocessing."""

    def __init__(self, enable_location_removal: bool = False):
        self.enable_location_removal = enable_location_removal
        self.location_remover = None

        self.classifier = None
        self.vectorizer = None
        self.classes_ = None
        self.synonyms = {}
        self.display_names = {}

        self.lemmatizer = None
        self.stop_words = set()
        self._init_nlp()

        if self.enable_location_removal:
            from app.preprocessing.ner_location_remover import NERLocationRemover
            self.location_remover = NERLocationRemover(placeholder="<LOCATION>")

    def _init_nlp(self):
        if not nltk or not stopwords or not WordNetLemmatizer:
            return

        for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass

        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.lemmatizer = None
            self.stop_words = set()

        generic_resume_terms = {
            "company", "company name", "name", "name location", "location",
            "bachelor", "master", "degree", "graduate", "college", "university",
            "year", "month", "experience",
            "microsoft", "office", "excel", "word", "powerpoint",
        }
        self.stop_words.update(generic_resume_terms)

    @staticmethod
    def load_model(model_dir: str, model_name: str = "advanced_cnb", enable_location_removal: bool = False):
        classifier = InferenceCareerPathClassifier(enable_location_removal=enable_location_removal)

        classifier.classifier = joblib.load(os.path.join(model_dir, f"{model_name}_classifier.pkl"))
        classifier.vectorizer = joblib.load(os.path.join(model_dir, f"{model_name}_vectorizer.pkl"))
        classifier.classes_ = joblib.load(os.path.join(model_dir, f"{model_name}_classes.pkl"))
        classifier.synonyms = joblib.load(os.path.join(model_dir, f"{model_name}_synonyms.pkl"))

        try:
            classifier.display_names = joblib.load(os.path.join(model_dir, f"{model_name}_display_names.pkl"))
        except FileNotFoundError:
            classifier.display_names = {}

        return classifier

    def get_display_name(self, class_name: str) -> str:
        return self.display_names.get(class_name, class_name.title())

    def preprocess_text(self, text: str) -> str:
        text = str(text)

        if self.location_remover is not None:
            text = self.location_remover.remove_locations(text)

        text = text.lower()

        text = re.sub(
            r"\b(january|february|march|april|may|june|july|august|september|"
            r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
            "", text, flags=re.I
        )
        text = re.sub(r"\b(19|20)\d{2}\b", "", text)
        text = re.sub(r"[-\u2013]?\s*\b(present|current|till|ongoing)\b", "", text, flags=re.I)
        text = re.sub(r"\b\d+(st|nd|rd|th)\b", "", text, flags=re.I)
        text = re.sub(
            r"\b(summary|objective|profile|references?|curriculum vitae|"
            r"date of birth|dob|nationality|gender|marital status|religion|hobbies?)\b",
            "", text, flags=re.I
        )
        text = re.sub(r"(?<![a-z\d])\d{1,3}(?![a-z\d])", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        for term in sorted(self.synonyms.keys(), key=len, reverse=True):
            replacement = self.synonyms[term]
            pattern = r"\b" + re.escape(term) + r"\b"
            text = re.sub(pattern, replacement, text)

        text = text.translate(str.maketrans("", "", string.punctuation))

        try:
            tokens = word_tokenize(text) if word_tokenize else text.split()
        except Exception:
            tokens = text.split()

        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                if self.lemmatizer is not None:
                    lemma = self.lemmatizer.lemmatize(token, pos="v")
                    lemma = self.lemmatizer.lemmatize(lemma, pos="n")
                    processed_tokens.append(lemma)
                else:
                    processed_tokens.append(token)

        return " ".join(processed_tokens)