"""
PDF Text Extraction Utility
Extracts text content from PDF resume files
"""

import PyPDF2
from typing import Optional
import os


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be read or processed
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("File must be a PDF")
    
    try:
        text_content = []
        
        # Open and read PDF
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Get number of pages
            num_pages = len(pdf_reader.pages)
            
            if num_pages == 0:
                raise Exception("PDF has no pages")
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text:
                    text_content.append(text)
        
        # Combine all text
        full_text = "\n".join(text_content)
        
        # Clean up text
        full_text = full_text.strip()
        
        # Remove excessive whitespace
        full_text = " ".join(full_text.split())
        
        return full_text
        
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def validate_pdf_content(text: str, min_length: int = 50) -> bool:
    """
    Validate that extracted text has sufficient content.
    
    Args:
        text: Extracted text
        min_length: Minimum required text length
        
    Returns:
        True if valid, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check minimum length
    if len(text.strip()) < min_length:
        return False
    
    # Check for actual words (not just special characters)
    words = text.split()
    if len(words) < 10:
        return False
    
    return True
