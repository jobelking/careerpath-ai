"""
Named Entity Recognition (NER) Location Remover
Removes geographic entities (GPE/LOC) to prevent geographic leakage in ML models.
"""

import re
import warnings
warnings.filterwarnings('ignore')


class NERLocationRemover:
    """
    Removes geographic entities from text using spaCy NER.
    Prevents location-based bias in career path prediction models.
    """
    
    def __init__(self, placeholder: str = "<LOCATION>"):
        """
        Initialize the NER location remover.
        
        Args:
            placeholder: String to replace detected locations with
        """
        self.placeholder = placeholder
        self.nlp = None
        self._load_spacy_model()
        
        # Comprehensive list of location keywords
        # These supplement NER to catch edge cases
        self.location_keywords = self._build_location_keywords()
        
        # Common location patterns that might slip through NER
        self.location_patterns = [
            r'\b(city|cities|state|country|region|county|province|district)\b',
            r'\b(downtown|uptown|suburb|metropolitan|metro)\b',
            r'\b(north|south|east|west|northeast|northwest|southeast|southwest)ern?\b',
            r'\b(usa|u\.s\.a\.|united states|uk|u\.k\.)\b',
        ]
    
    def _build_location_keywords(self) -> set:
        """
        Build comprehensive set of location keywords to supplement NER.
        Returns set of lowercase location terms.
        """
        keywords = set()
        
        # Major US cities
        us_cities = [
            'seattle', 'portland', 'boston', 'miami', 'denver', 'dallas', 
            'houston', 'austin', 'phoenix', 'philadelphia', 'detroit',
            'minneapolis', 'atlanta', 'cleveland', 'cincinnati', 'pittsburgh',
            'baltimore', 'milwaukee', 'vegas', 'sacramento', 'oakland',
            'raleigh', 'charlotte', 'nashville', 'memphis', 'louisville',
            'indianapolis', 'columbus', 'jacksonville', 'tampa', 'orlando',
            'francisco', 'diego', 'jose', 'antonio', 'angeles', 'los',  # Partial city names
            'chicago', 'omaha', 'tulsa', 'wichita', 'albuquerque', 'tucson',
            'fresno', 'bakersfield', 'kansas',
            # Multi-word cities (these need explicit listing for keyword matching)
            'los angeles', 'san francisco', 'san diego', 'san jose', 'san antonio',
            'new york', 'new orleans', 'las vegas', 'kansas city', 'fort worth',
            'silicon valley'
        ]
        
        # Major international cities
        international_cities = [
            'london', 'paris', 'berlin', 'tokyo', 'beijing', 'shanghai',
            'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai',
            'sydney', 'melbourne', 'toronto', 'vancouver', 'montreal',
            'singapore', 'dubai', 'amsterdam', 'stockholm', 'oslo',
            'copenhagen', 'zurich', 'geneva', 'milan', 'rome', 'madrid',
            'barcelona', 'lisbon', 'moscow', 'prague', 'vienna', 'brussels',
            'dublin', 'edinburgh', 'manchester', 'glasgow', 'birmingham'
        ]
        
        # US states
        us_states = [
            'alabama', 'alaska', 'arizona', 'arkansas', 'california',
            'colorado', 'connecticut', 'delaware', 'florida', 'georgia',
            'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas',
            'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts',
            'michigan', 'minnesota', 'mississippi', 'missouri', 'montana',
            'nebraska', 'nevada', 'hampshire', 'jersey', 'mexico', 'york',
            'carolina', 'dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
            'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
            'wisconsin', 'wyoming'
        ]
        
        # Universities and colleges (often location indicators)
        universities = [
            'stanford', 'harvard', 'mit', 'yale', 'princeton', 'cornell',
            'columbia', 'duke', 'northwestern', 'vanderbilt', 'georgetown',
            'berkeley', 'ucla', 'usc', 'caltech'
        ]
        
        # Countries
        countries = [
            'usa', 'america', 'canada', 'mexico', 'brazil', 'argentina',
            'england', 'britain', 'france', 'germany', 'spain', 'italy',
            'portugal', 'netherlands', 'belgium', 'switzerland', 'austria',
            'sweden', 'norway', 'denmark', 'finland', 'poland', 'russia',
            'ukraine', 'turkey', 'israel', 'egypt', 'nigeria', 'kenya',
            'india', 'pakistan', 'china', 'japan', 'korea', 'vietnam',
            'thailand', 'indonesia', 'malaysia', 'philippines', 'australia',
            'zealand', 'bangladesh', 'colombia', 'peru', 'chile', 'venezuela',
            'ecuador', 'bolivia', 'paraguay', 'uruguay', 'iran', 'iraq',
            'afghanistan', 'nepal', 'srilanka', 'myanmar', 'cambodia', 'laos'
        ]
        
        keywords.update([k.lower() for k in us_cities])
        keywords.update([k.lower() for k in international_cities])
        keywords.update([k.lower() for k in us_states])
        keywords.update([k.lower() for k in universities])
        keywords.update([k.lower() for k in countries])
        
        return keywords
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling."""
        try:
            import spacy
            try:
                # Try to load the model
                self.nlp = spacy.load("en_core_web_sm")
                print("✓ spaCy model 'en_core_web_sm' loaded successfully")
            except OSError:
                print("⚠ spaCy model 'en_core_web_sm' not found. Downloading...")
                import subprocess
                import sys
                
                # Download the model
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ])
                
                # Load the model
                self.nlp = spacy.load("en_core_web_sm")
                print("✓ spaCy model 'en_core_web_sm' downloaded and loaded successfully")
                
        except ImportError:
            raise ImportError(
                "spaCy is not installed. Please install it using: pip install spacy>=3.7.0"
            )
    
    def remove_locations(self, text: str) -> str:
        """
        Remove all geographic entities from text using NER and keyword matching.
        
        Args:
            text: Input text containing potential location entities
            
        Returns:
            Text with all GPE/LOC entities replaced with placeholder
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Process with spaCy NER
        doc = self.nlp(text)
        
        # Build list of entities to remove (GPE = geopolitical entity, LOC = location)
        # Sort in reverse order by position to avoid index shifting during replacement
        entities_to_remove = []
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                entities_to_remove.append((ent.start_char, ent.end_char, ent.text))
        
        # Sort by start position in reverse order
        entities_to_remove.sort(key=lambda x: x[0], reverse=True)
        
        # Replace entities with placeholder
        result = text
        for start, end, entity_text in entities_to_remove:
            result = result[:start] + self.placeholder + result[end:]
        
        # Multi-word phrase matching (for cities like "los angeles", "san francisco")
        # This needs to happen BEFORE single-word matching
        multi_word_locations = [kw for kw in self.location_keywords if ' ' in kw]
        for phrase in sorted(multi_word_locations, key=len, reverse=True):  # Longest first
            pattern = r'\b' + re.escape(phrase) + r'\b'
            result = re.sub(pattern, self.placeholder, result, flags=re.IGNORECASE)
        
        # Additional single-word keyword-based cleanup
        # This catches locations that NER might miss
        words = result.split()
        cleaned_words = []
        for word in words:
            # Remove punctuation for comparison
            word_clean = word.lower().strip('.,;:!?()')
            if word_clean in self.location_keywords:
                cleaned_words.append(self.placeholder)
            else:
                cleaned_words.append(word)
        result = ' '.join(cleaned_words)
        
        # Pattern-based cleanup for common location terms
        for pattern in self.location_patterns:
            result = re.sub(pattern, self.placeholder, result, flags=re.IGNORECASE)
        
        # Clean up multiple consecutive placeholders
        while f"{self.placeholder} {self.placeholder}" in result:
            result = result.replace(f"{self.placeholder} {self.placeholder}", self.placeholder)
        
        return result
    
    def get_detected_locations(self, text: str) -> list:
        """
        Get list of all detected location entities without removing them.
        Useful for debugging and validation.
        
        Args:
            text: Input text
            
        Returns:
            List of tuples (entity_text, entity_label)
        """
        if not text or not isinstance(text, str):
            return []
        
        doc = self.nlp(text)
        locations = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        return locations


class LocationLeakageValidator:
    """
    Validates that no geographic terms appear in model vocabulary or features.
    Uses a conservative list to avoid false positives from legitimate terms.
    """
    
    def __init__(self):
        """Initialize the validator with focused location term lists."""
        # Focused list of unambiguous location terms that should never appear
        # Avoids terms that could be used in professional contexts
        self.forbidden_terms = self._build_forbidden_terms()
    
    def _build_forbidden_terms(self) -> set:
        """
        Build focused set of unambiguous location terms.
        Only includes terms that are CLEARLY geographic and have no professional use.
        Excludes ambiguous terms like 'area', 'american', 'central', etc.
        """
        terms = set()
        
        # Major US cities (unambiguous city names only)
        us_cities = [
            'san francisco', 'los angeles', 'new york', 'chicago', 'houston',
            'phoenix', 'philadelphia', 'san antonio', 'san diego', 'dallas',
            'san jose', 'austin', 'jacksonville', 'fort worth', 'charlotte',
            'seattle', 'denver', 'boston', 'nashville', 'detroit', 'portland',
            'las vegas', 'memphis', 'louisville', 'baltimore', 'milwaukee',
            'albuquerque', 'tucson', 'fresno', 'sacramento', 'atlanta',
            'kansas city', 'miami', 'raleigh', 'omaha', 'minneapolis', 'tulsa',
            'cleveland', 'tampa', 'new orleans', 'bakersfield', 'brooklyn',
            'silicon valley'
        ]
        
        # Major international cities (unambiguous)
        international_cities = [
            'london', 'paris', 'berlin', 'tokyo', 'beijing', 'shanghai', 'mumbai',
            'delhi', 'bangalore', 'hyderabad', 'chennai', 'sydney', 'melbourne',
            'toronto', 'vancouver', 'montreal', 'singapore', 'dubai', 'amsterdam',
            'stockholm', 'oslo', 'copenhagen', 'zurich', 'geneva', 'milan', 'rome',
            'madrid', 'barcelona', 'lisbon', 'moscow', 'prague', 'vienna', 'brussels',
            'dublin', 'edinburgh', 'manchester', 'glasgow', 'birmingham', 'leeds'
        ]
        
        # US states (full names only, no abbreviations to avoid false positives)
        us_states_full = [
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
            'connecticut', 'delaware', 'florida', 'hawaii', 'idaho',
            'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
            'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 
            'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 
            'new hampshire', 'new jersey', 'new mexico', 'new york',
            'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
            'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
            'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
            'west virginia', 'wisconsin', 'wyoming'
        ]
        
        # Countries (but NOT 'america'/'american' as they appear in org names)
        countries = [
            'afghanistan', 'argentina', 'australia', 'bangladesh', 'brazil', 
            'cambodia', 'chile', 'colombia', 'egypt', 'ethiopia', 'france', 
            'germany', 'greece', 'indonesia', 'iran', 'iraq', 'ireland',
            'israel', 'italy', 'japan', 'kenya', 'korea', 'malaysia', 'mexico',
            'morocco', 'myanmar', 'nepal', 'netherlands', 'new zealand', 'nigeria',
            'norway', 'pakistan', 'peru', 'philippines', 'poland', 'portugal',
            'russia', 'saudi arabia', 'south africa', 'spain', 'sweden',
            'switzerland', 'taiwan', 'thailand', 'turkey', 'ukraine', 
            'united arab emirates', 'united kingdom', 'venezuela', 'vietnam'
        ]
        
        # Add all terms (lowercase)
        terms.update([term.lower() for term in us_cities])
        terms.update([term.lower() for term in international_cities])
        terms.update([term.lower() for term in us_states_full])
        terms.update([term.lower() for term in countries])
        
        return terms
    
    def validate_vocabulary(self, vocabulary: dict) -> tuple:
        """
        Check if any forbidden location terms appear in the model vocabulary.
        Uses EXACT match only to avoid false positives.
        
        Args:
            vocabulary: Dictionary mapping terms to indices (from vectorizer.vocabulary_)
            
        Returns:
            Tuple of (is_valid: bool, leaked_terms: list)
        """
        leaked_terms = []
        
        for term in vocabulary.keys():
            term_lower = term.lower()
            # Only check EXACT matches to avoid false positives
            if term_lower in self.forbidden_terms:
                leaked_terms.append(term)
        
        is_valid = len(leaked_terms) == 0
        return is_valid, list(set(leaked_terms))  # Remove duplicates
    
    def validate_features(self, feature_names: list) -> tuple:
        """
        Check if any forbidden location terms appear in feature names.
        Uses EXACT match only.
        
        Args:
            feature_names: List of feature names from the model
            
        Returns:
            Tuple of (is_valid: bool, leaked_terms: list)
        """
        leaked_terms = []
        
        for feature in feature_names:
            feature_lower = feature.lower()
            # Only check EXACT matches
            if feature_lower in self.forbidden_terms:
                leaked_terms.append(feature)
        
        is_valid = len(leaked_terms) == 0
        return is_valid, list(set(leaked_terms))  # Remove duplicates
    
    def assert_no_leakage(self, vocabulary: dict, fail_on_leakage: bool = True):
        """
        Assert that no location leakage exists in vocabulary.
        Raises an exception if leakage is detected.
        
        Args:
            vocabulary: Dictionary mapping terms to indices
            fail_on_leakage: If True, raise exception on detected leakage
            
        Raises:
            AssertionError: If location leakage is detected and fail_on_leakage is True
        """
        is_valid, leaked_terms = self.validate_vocabulary(vocabulary)
        
        if not is_valid:
            error_msg = (
                f"\n{'='*80}\n"
                f"LOCATION LEAKAGE DETECTED!\n"
                f"{'='*80}\n"
                f"Found {len(leaked_terms)} location-related terms in vocabulary:\n"
                f"{', '.join(sorted(leaked_terms)[:20])}"  # Show first 20
            )
            if len(leaked_terms) > 20:
                error_msg += f"\n... and {len(leaked_terms) - 20} more"
            error_msg += f"\n{'='*80}\n"
            
            print(error_msg)
            
            if fail_on_leakage:
                raise AssertionError(
                    f"Training aborted: {len(leaked_terms)} location terms found in vocabulary. "
                    f"Geographic leakage must be eliminated before deployment."
                )
        else:
            print(f"✓ Location leakage validation PASSED - No geographic terms detected in vocabulary")
