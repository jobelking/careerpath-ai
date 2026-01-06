import pandas as pd
import re
import string
from typing import List, Optional
import os


class TextPreprocessor:
    """
    Text preprocessing class for career path AI model.
    Handles cleaning, normalization, and transformation of resume text data.
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.stopwords = self._get_stopwords()
    
    def _get_stopwords(self) -> set:
        """Get common English stopwords."""
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
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers (optional - comment out if numbers are important)
        # text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def remove_short_words(self, text: str, min_length: int = 2) -> str:
        """
        Remove words shorter than specified length.
        
        Args:
            text: Input text
            min_length: Minimum word length to keep
            
        Returns:
            Text with short words removed
        """
        words = text.split()
        filtered_words = [word for word in words if len(word) >= min_length]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str, remove_stops: bool = True, 
                   min_word_length: int = 2) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw input text
            remove_stops: Whether to remove stopwords
            min_word_length: Minimum length for words to keep
            
        Returns:
            Fully preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords if requested
        if remove_stops:
            text = self.remove_stopwords(text)
        
        # Remove short words
        text = self.remove_short_words(text, min_word_length)
        
        return text
    
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
            lambda x: self.preprocess(x, remove_stops=remove_stops)
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


if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_file = os.path.join(base_dir, 'data', 'datasets', 'finance_resume.csv')
    output_file = os.path.join(base_dir, 'data', 'datasets', 'finance_resume_preprocessed.csv')
    
    # Run preprocessing
    preprocess_dataset(
        input_path=input_file,
        output_path=output_file,
        text_column='text_needed',
        target_column='path',
        remove_stops=True
    )
