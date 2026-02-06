"""
Text Augmentation Module for Career Path Classification

Provides text augmentation techniques to increase diversity of minority classes
without naive duplication that causes overfitting.

Techniques:
1. Back-translation (EN → DE/FR → EN) - Most diversity
2. Synonym replacement using WordNet - Lightweight fallback
3. Random word swap/delete - Simple augmentation

Author: CareerPath-AI Team
Date: Feb 2026
"""

import random
import re
from typing import List, Optional, Tuple
import logging

# Optional imports for advanced augmentation
try:
    from transformers import MarianMTModel, MarianTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
try:
    from nltk.corpus import wordnet
    import nltk
    # Ensure wordnet is downloaded
    try:
        wordnet.synsets('test')
        WORDNET_AVAILABLE = True
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextAugmenter:
    """
    Text augmentation for career path training data.
    
    Strategies (in order of preference):
    1. Back-translation: Best for semantic diversity
    2. Synonym replacement: Lightweight alternative
    3. Simple augmentation: Word swap, shuffle
    """
    
    def __init__(self, use_gpu: bool = False, back_translation: bool = True):
        """
        Initialize the text augmenter.
        
        Args:
            use_gpu: Whether to use GPU for translation models
            back_translation: Whether to load back-translation models (slow init)
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.back_translation_enabled = back_translation and TRANSFORMERS_AVAILABLE
        
        # Lazy load translation models
        self._en_de_model = None
        self._en_de_tokenizer = None
        self._de_en_model = None
        self._de_en_tokenizer = None
        
        if back_translation and not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers not available. Back-translation disabled.")
        
        if not WORDNET_AVAILABLE:
            logger.warning("WordNet not available. Synonym replacement disabled.")
    
    def _load_translation_models(self):
        """Lazy load translation models for back-translation."""
        if self._en_de_model is None:
            logger.info("Loading English→German translation model...")
            model_name = 'Helsinki-NLP/opus-mt-en-de'
            self._en_de_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self._en_de_model = MarianMTModel.from_pretrained(model_name).to(self.device)
            
            logger.info("Loading German→English translation model...")
            model_name = 'Helsinki-NLP/opus-mt-de-en'
            self._de_en_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self._de_en_model = MarianMTModel.from_pretrained(model_name).to(self.device)
            
            logger.info("Translation models loaded successfully.")
    
    def _translate(self, text: str, model, tokenizer) -> str:
        """Translate text using a MarianMT model."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, num_beams=4)
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def back_translate(self, text: str) -> str:
        """
        Augment text via back-translation (EN → DE → EN).
        
        This produces semantically equivalent but syntactically different text.
        """
        if not self.back_translation_enabled:
            return self.synonym_replacement(text)
        
        self._load_translation_models()
        
        try:
            # EN → DE
            german = self._translate(text, self._en_de_model, self._en_de_tokenizer)
            # DE → EN
            english = self._translate(german, self._de_en_model, self._de_en_tokenizer)
            return english
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}. Falling back to synonym replacement.")
            return self.synonym_replacement(text)
    
    def synonym_replacement(self, text: str, n_replacements: int = 3) -> str:
        """
        Replace random words with synonyms from WordNet.
        
        Args:
            text: Input text
            n_replacements: Number of words to replace
        
        Returns:
            Augmented text with synonym replacements
        """
        if not WORDNET_AVAILABLE:
            return self.simple_augment(text)
        
        words = text.split()
        if len(words) < 3:
            return text
        
        # Find words that have synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            # Skip short words and common words
            if len(word) < 4:
                continue
            synsets = wordnet.synsets(word.lower())
            if synsets:
                replaceable_indices.append(i)
        
        if not replaceable_indices:
            return text
        
        # Replace n random words
        n_to_replace = min(n_replacements, len(replaceable_indices))
        indices_to_replace = random.sample(replaceable_indices, n_to_replace)
        
        for idx in indices_to_replace:
            word = words[idx].lower()
            synsets = wordnet.synsets(word)
            if synsets:
                # Get all lemmas from all synsets
                synonyms = []
                for syn in synsets[:3]:  # Limit to first 3 synsets
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != word:
                            synonyms.append(synonym)
                
                if synonyms:
                    # Replace with random synonym
                    replacement = random.choice(synonyms)
                    # Preserve original case
                    if words[idx][0].isupper():
                        replacement = replacement.capitalize()
                    words[idx] = replacement
        
        return ' '.join(words)
    
    def simple_augment(self, text: str) -> str:
        """
        Simple augmentation: word swap, random delete.
        Used as fallback when other methods unavailable.
        """
        words = text.split()
        if len(words) < 4:
            return text
        
        augmented = words.copy()
        
        # Random word swap (2 pairs)
        for _ in range(2):
            if len(augmented) >= 4:
                idx1, idx2 = random.sample(range(len(augmented)), 2)
                augmented[idx1], augmented[idx2] = augmented[idx2], augmented[idx1]
        
        return ' '.join(augmented)
    
    def augment(self, text: str, technique: str = 'auto') -> str:
        """
        Augment a single text sample.
        
        Args:
            text: Input text
            technique: 'back_translation', 'synonym', 'simple', or 'auto'
        
        Returns:
            Augmented text
        """
        if technique == 'back_translation':
            return self.back_translate(text)
        elif technique == 'synonym':
            return self.synonym_replacement(text)
        elif technique == 'simple':
            return self.simple_augment(text)
        else:  # auto
            if self.back_translation_enabled:
                return self.back_translate(text)
            elif WORDNET_AVAILABLE:
                return self.synonym_replacement(text)
            else:
                return self.simple_augment(text)
    
    def augment_batch(
        self, 
        texts: List[str], 
        target_count: int,
        technique: str = 'auto',
        max_augment_ratio: float = 3.0
    ) -> List[str]:
        """
        Augment a batch of texts to reach target_count.
        
        Args:
            texts: Original texts (will be included in output)
            target_count: Desired total number of samples
            technique: Augmentation technique to use
            max_augment_ratio: Maximum ratio of augmented to original
        
        Returns:
            List of original + augmented texts
        """
        original_count = len(texts)
        if original_count >= target_count:
            return list(texts)
        
        # Calculate how many augmented samples we need
        needed = target_count - original_count
        max_augmented = int(original_count * max_augment_ratio)
        
        if needed > max_augmented:
            logger.warning(
                f"Requested {needed} augmented samples but limiting to {max_augmented} "
                f"({max_augment_ratio}x original {original_count})"
            )
            needed = max_augmented
        
        result = list(texts)
        
        # Create augmented samples by cycling through originals
        augment_idx = 0
        for i in range(needed):
            source_text = texts[augment_idx % original_count]
            augmented = self.augment(source_text, technique)
            result.append(augmented)
            augment_idx += 1
        
        return result[:target_count]


class SimpleOversampler:
    """
    Simple oversampling without augmentation.
    Used when augmentation models are not available or when
    classes have enough original samples.
    """
    
    @staticmethod
    def oversample(texts: List[str], target_count: int) -> List[str]:
        """
        Oversample by random duplication.
        
        Args:
            texts: Original texts
            target_count: Desired count
        
        Returns:
            Oversampled texts
        """
        if len(texts) >= target_count:
            return list(texts)
        
        result = list(texts)
        
        # Duplicate randomly until reaching target
        while len(result) < target_count:
            result.append(random.choice(texts))
        
        return result[:target_count]


# For backward compatibility
def create_augmenter(use_augmentation: bool = True, use_gpu: bool = False) -> TextAugmenter:
    """
    Factory function to create a text augmenter.
    
    Args:
        use_augmentation: If False, returns a minimal augmenter
        use_gpu: Whether to use GPU for translation
    
    Returns:
        TextAugmenter instance
    """
    return TextAugmenter(
        use_gpu=use_gpu,
        back_translation=use_augmentation and TRANSFORMERS_AVAILABLE
    )
