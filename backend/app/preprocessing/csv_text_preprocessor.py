import pandas as pd
import re
import string
from typing import List, Optional
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, backend_dir)

# Import NER location removal utilities
from app.preprocessing.ner_location_remover import NERLocationRemover

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


class TextPreprocessor:
    """
    Advanced text preprocessing class for career path AI model.
    Matches the preprocessing used in train_model_advanced.py:
    - NER location removal (prevents geographic bias)
    - Lemmatization
    - Synonym normalization
    - Extended stopwords
    """
    
    def __init__(self):
        """Initialize the advanced text preprocessor."""
        self.stopwords = self._get_stopwords()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add generic resume terms that cause class confusion
        generic_resume_terms = {
            'company', 'company name', 'name', 'name location', 'location',
            'bachelor', 'master', 'degree', 'graduate', 'college', 'university',
            'year', 'month', 'experience',
            'microsoft', 'office', 'excel', 'word', 'powerpoint',
        }
        self.stop_words.update(generic_resume_terms)
        
        # Initialize NER location remover
        print("Initializing NER location remover...")
        self.location_remover = NERLocationRemover(placeholder="<LOCATION>")
        print("✓ NER location remover initialized")
        
        # Synonym normalization dictionary
        self.synonyms = self._get_synonyms()
    
    def _get_stopwords(self) -> set:
        """Get common English stopwords (kept for backward compatibility)."""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
    
    def _get_synonyms(self) -> dict:
        """Get synonym normalization dictionary for career-related terms."""
        return {
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
            'de': 'datascience',
            'etl engineer': 'datascience',
            'big data engineer': 'datascience',
            
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
            
            # Sales
            'salesperson': 'sales',
            'sales rep': 'sales',
            'sales representative': 'sales',
            'account executive': 'sales',
            'quota': 'sales',
            'cold call': 'sales',
            'prospecting': 'sales',
            'crm': 'sales',
            
            # HR
            'human resources': 'hr',
            'recruiter': 'hr',
            'talent acquisition': 'hr',
            
            # Design
            'graphic designer': 'design-creative',
            'ui designer': 'design-creative',
            'ux designer': 'design-creative',
            'designer': 'design-creative',
            'figma': 'design-creative',
            'photoshop': 'design-creative',
            
            # IT Support
            'it support': 'it-support',
            'helpdesk': 'it-support',
            'technical support': 'it-support',
            
            # General tech terms
            'fullstack': 'full stack',
            'frontend': 'front end',
            'backend': 'back end',
            'db': 'database',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience'
        }
    
    def normalize_synonyms(self, text: str) -> str:
        """Normalize common synonyms and abbreviations."""
        text_lower = text.lower()
        # Process longest phrases first to avoid partial matches
        for synonym in sorted(self.synonyms.keys(), key=len, reverse=True):
            replacement = self.synonyms[synonym]
            pattern = r'\b' + re.escape(synonym) + r'\b'
            text_lower = re.sub(pattern, replacement, text_lower)
        return text_lower
    
    def preprocess(self, text: str, remove_stops: bool = True, 
                   min_word_length: int = 3) -> str:
        """
        Advanced preprocessing pipeline matching train_model_advanced.py:
        0. Remove geographic entities using NER (PREVENTS LOCATION LEAKAGE)
        1. Convert to string + lowercase
        2. Synonym normalization
        3. Remove punctuation
        4. Tokenization
        5. Remove stopwords
        6. Lemmatization
        
        Args:
            text: Raw input text
            remove_stops: Whether to remove stopwords
            min_word_length: Minimum length for words to keep (default: 3)
            
        Returns:
            Fully preprocessed text
        """
        # STEP 0: Remove geographic entities BEFORE any other processing
        text = str(text)
        if pd.isna(text) or text == "":
            return ""
        
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
            if token not in self.stop_words and len(token) >= min_word_length:
                # Lemmatize as verb first, then as noun
                lemma = self.lemmatizer.lemmatize(token, pos='v')
                lemma = self.lemmatizer.lemmatize(lemma, pos='n')
                processed_tokens.append(lemma)
        
        return ' '.join(processed_tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text_needed',
                            target_column: str = 'path', 
                            remove_stops: bool = True) -> pd.DataFrame:
        """
        Preprocess an entire DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text to preprocess
            target_column: Name of target/label column
            remove_stops: Whether to remove stopwords
            
        Returns:
            DataFrame with preprocessed text
        """
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Remove duplicates
        df_processed = df_processed.drop_duplicates()
        
        # Remove rows with missing values
        df_processed = df_processed.dropna(subset=[text_column, target_column])
        
        # Apply preprocessing to text column
        print(f"Preprocessing {len(df_processed)} rows...")
        df_processed[text_column] = df_processed[text_column].apply(
            lambda x: self.preprocess(x, remove_stops=remove_stops, min_word_length=3)
        )
        
        # Remove empty text after preprocessing
        df_processed = df_processed[df_processed[text_column].str.strip() != '']
        
        # Reset index
        df_processed = df_processed.reset_index(drop=True)
        
        return df_processed


def preprocess_dataset(input_path: str, output_path: str, 
                       text_column: str = 'text_needed',
                       target_column: str = 'path',
                       remove_stops: bool = True) -> None:
    """
    Load, preprocess, and save a dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save preprocessed CSV file
        text_column: Name of text column
        target_column: Name of target column
        remove_stops: Whether to remove stopwords
    """
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Print class distribution
    print("\nClass distribution (before preprocessing):")
    print(df[target_column].value_counts())
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess dataset
    df_processed = preprocessor.preprocess_dataframe(
        df, 
        text_column=text_column,
        target_column=target_column,
        remove_stops=remove_stops
    )
    
    print(f"\nProcessed dataset shape: {df_processed.shape}")
    print("\nClass distribution (after preprocessing):")
    print(df_processed[target_column].value_counts())
    
    # Save preprocessed data
    df_processed.to_csv(output_path, index=False)
    print(f"\nPreprocessed dataset saved to: {output_path}")
    
    # Show sample
    print("\n" + "="*80)
    print("SAMPLE OF PREPROCESSED DATA:")
    print("="*80)
    for idx in range(min(3, len(df_processed))):
        print(f"\nClass: {df_processed.iloc[idx][target_column]}")
        print(f"Text (first 200 chars): {df_processed.iloc[idx][text_column][:200]}...")
    print("="*80)


class PreprocessorGUI:
    """Simple GUI application for text preprocessing."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Text Preprocessor")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.text_column = tk.StringVar(value="text_needed")
        self.target_column = tk.StringVar(value="path")
        self.remove_stopwords = tk.BooleanVar(value=True)
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="CSV Text Preprocessor", 
            font=("Arial", 16, "bold"),
            pady=20
        )
        title_label.pack()
        
        # Input file section
        input_frame = tk.LabelFrame(self.root, text="Input CSV File", padx=10, pady=10)
        input_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side="left", padx=5)
        tk.Button(input_frame, text="Browse...", command=self.browse_input).pack(side="left")
        
        # Output file section
        output_frame = tk.LabelFrame(self.root, text="Output CSV File (optional - auto-generated if empty)", padx=10, pady=10)
        output_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side="left", padx=5)
        tk.Button(output_frame, text="Browse...", command=self.browse_output).pack(side="left")
        
        # Column names section
        columns_frame = tk.LabelFrame(self.root, text="Column Settings", padx=10, pady=10)
        columns_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(columns_frame, text="Text Column Name:").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(columns_frame, textvariable=self.text_column, width=30).grid(row=0, column=1, pady=5, padx=5)
        
        tk.Label(columns_frame, text="Target Column Name:").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(columns_frame, textvariable=self.target_column, width=30).grid(row=1, column=1, pady=5, padx=5)
        
        # Options section
        options_frame = tk.LabelFrame(self.root, text="Preprocessing Options", padx=10, pady=10)
        options_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Checkbutton(
            options_frame, 
            text="Remove stopwords", 
            variable=self.remove_stopwords
        ).pack(anchor="w")
        
        # Process button
        self.process_button = tk.Button(
            self.root, 
            text="Process Dataset", 
            command=self.process_dataset,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            pady=10,
            cursor="hand2"
        )
        self.process_button.pack(pady=20, padx=20, fill="x")
        
        # Status section
        self.status_text = tk.Text(self.root, height=8, width=70, state="disabled", wrap="word")
        self.status_text.pack(padx=20, pady=(0, 20))
        
        # Scrollbar for status
        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side="right", fill="y")
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
    
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_path.set(filename)
            # Always auto-generate output path for new input
            base, ext = os.path.splitext(filename)
            self.output_path.set(f"{base}_preprocessed{ext}")
    
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output CSV File",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
    
    def log_status(self, message):
        """Add message to status text box."""
        self.status_text.config(state="normal")
        self.status_text.insert("end", message + "\n")
        self.status_text.see("end")
        self.status_text.config(state="disabled")
        self.root.update_idletasks()
    
    def process_dataset(self):
        """Process the dataset in a separate thread."""
        # Validate inputs
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input CSV file!")
            return
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input file does not exist!")
            return
        
        # Auto-generate output path if not provided
        output = self.output_path.get()
        if not output:
            base, ext = os.path.splitext(self.input_path.get())
            output = f"{base}_preprocessed{ext}"
            self.output_path.set(output)
        
        # Clear status
        self.status_text.config(state="normal")
        self.status_text.delete(1.0, "end")
        self.status_text.config(state="disabled")
        
        # Disable button during processing
        self.process_button.config(state="disabled", text="Processing...")
        
        # Run processing in separate thread
        thread = threading.Thread(target=self._process_worker)
        thread.daemon = True
        thread.start()
    
    def _process_worker(self):
        """Worker thread for processing dataset."""
        try:
            self.log_status(f"Loading dataset from: {self.input_path.get()}")
            df = pd.read_csv(self.input_path.get())
            self.log_status(f"Original dataset shape: {df.shape}")
            
            # Print class distribution
            self.log_status("\nClass distribution (before preprocessing):")
            class_dist = df[self.target_column.get()].value_counts()
            for cls, count in class_dist.items():
                self.log_status(f"  {cls}: {count}")
            
            # Initialize preprocessor
            preprocessor = TextPreprocessor()
            
            # Preprocess dataset
            self.log_status(f"\nPreprocessing {len(df)} rows...")
            df_processed = preprocessor.preprocess_dataframe(
                df, 
                text_column=self.text_column.get(),
                target_column=self.target_column.get(),
                remove_stops=self.remove_stopwords.get()
            )
            
            self.log_status(f"Processed dataset shape: {df_processed.shape}")
            self.log_status("\nClass distribution (after preprocessing):")
            class_dist = df_processed[self.target_column.get()].value_counts()
            for cls, count in class_dist.items():
                self.log_status(f"  {cls}: {count}")
            
            # Save preprocessed data
            df_processed.to_csv(self.output_path.get(), index=False)
            self.log_status(f"\n✓ Preprocessed dataset saved to: {self.output_path.get()}")
            
            # Show sample
            self.log_status("\n" + "="*60)
            self.log_status("SAMPLE OF PREPROCESSED DATA:")
            for idx in range(min(2, len(df_processed))):
                self.log_status(f"\nClass: {df_processed.iloc[idx][self.target_column.get()]}")
                text_sample = df_processed.iloc[idx][self.text_column.get()][:150]
                self.log_status(f"Text: {text_sample}...")
            self.log_status("="*60)
            
            # Success message
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", 
                f"Dataset preprocessed successfully!\n\nSaved to:\n{self.output_path.get()}"
            ))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.log_status(f"\n✗ {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        finally:
            # Re-enable button
            self.root.after(0, lambda: self.process_button.config(
                state="normal", 
                text="Process Dataset"
            ))


def run_gui():
    """Launch the GUI application."""
    root = tk.Tk()
    app = PreprocessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
