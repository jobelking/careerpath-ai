"""
Text preprocessing module for CareerPath-AI.

This module provides utilities for cleaning and preprocessing resume text data
for machine learning model training and prediction.
"""

from .text_preprocessor import TextPreprocessor, preprocess_dataset

__all__ = ['TextPreprocessor', 'preprocess_dataset']
