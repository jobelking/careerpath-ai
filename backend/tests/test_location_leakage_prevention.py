"""
Unit test for NER-based geographic leakage prevention.
Tests that location entities are properly removed from resume text.
"""

import sys
import os

# Add parent directory to path
backend_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_dir)

from app.preprocessing.ner_location_remover import NERLocationRemover, LocationLeakageValidator


def test_ner_location_removal():
    """Test that NER successfully removes geographic entities."""
    print("\n" + "="*80)
    print("TEST 1: NER Location Removal")
    print("="*80)
    
    # Initialize remover
    remover = NERLocationRemover(placeholder="<LOCATION>")
    
    # Test cases with known locations
    test_cases = [
        {
            "input": "Software engineer with 5 years experience in San Francisco, California",
            "description": "US city and state"
        },
        {
            "input": "Data scientist working in New York City for a Fortune 500 company",
            "description": "Major US city"
        },
        {
            "input": "Worked at Google London office and Amazon Seattle headquarters",
            "description": "Multiple international locations"
        },
        {
            "input": "Experience in Tokyo, Berlin, and Singapore with machine learning projects",
            "description": "Multiple international cities"
        },
        {
            "input": "Led teams across USA, Canada, and United Kingdom",
            "description": "Country names"
        },
        {
            "input": "Python developer with no location mentioned, just skills",
            "description": "No locations (should remain unchanged)"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case["input"]
        description = test_case["description"]
        
        print(f"\nTest case {i}: {description}")
        print(f"  Input:  {input_text}")
        
        # Get detected locations
        detected = remover.get_detected_locations(input_text)
        print(f"  Detected locations: {detected}")
        
        # Remove locations
        output = remover.remove_locations(input_text)
        print(f"  Output: {output}")
        
        # Validate
        if detected:
            # If locations were detected, output should contain placeholder
            if "<LOCATION>" in output:
                print(f"  ✓ PASS: Locations replaced with placeholder")
            else:
                print(f"  ✗ FAIL: Locations detected but not replaced!")
                all_passed = False
        else:
            # No locations detected, output should be similar to input (lowercased)
            if "<LOCATION>" not in output:
                print(f"  ✓ PASS: No locations detected, text preserved")
            else:
                print(f"  ⚠ WARNING: Placeholder added but no locations detected")
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ TEST 1 PASSED: All location removal tests successful")
    else:
        print("✗ TEST 1 FAILED: Some location removal tests failed")
    print("="*80)
    
    return all_passed


def test_leakage_validator():
    """Test that the leakage validator correctly identifies location terms."""
    print("\n" + "="*80)
    print("TEST 2: Location Leakage Validator")
    print("="*80)
    
    validator = LocationLeakageValidator()
    
    # Test case 1: Clean vocabulary (should pass)
    print("\nTest case 1: Clean vocabulary")
    clean_vocab = {
        'python': 0, 'machine': 1, 'learning': 2, 'data': 3, 'science': 4,
        'software': 5, 'engineer': 6, 'developer': 7, 'programming': 8
    }
    is_valid, leaked = validator.validate_vocabulary(clean_vocab)
    print(f"  Vocabulary: {list(clean_vocab.keys())}")
    print(f"  Valid: {is_valid}, Leaked terms: {leaked}")
    
    if is_valid and len(leaked) == 0:
        print("  ✓ PASS: Clean vocabulary correctly validated")
        test1_passed = True
    else:
        print(f"  ✗ FAIL: False positive - clean vocabulary flagged as leaky")
        test1_passed = False
    
    # Test case 2: Vocabulary with locations (should fail)
    print("\nTest case 2: Vocabulary with location terms")
    leaky_vocab = {
        'python': 0, 'california': 1, 'learning': 2, 'newyork': 3, 
        'science': 4, 'london': 5, 'engineer': 6
    }
    is_valid, leaked = validator.validate_vocabulary(leaky_vocab)
    print(f"  Vocabulary: {list(leaky_vocab.keys())}")
    print(f"  Valid: {is_valid}, Leaked terms: {leaked}")
    
    if not is_valid and len(leaked) > 0:
        print(f"  ✓ PASS: Location leakage correctly detected ({len(leaked)} terms)")
        test2_passed = True
    else:
        print(f"  ✗ FAIL: Location terms not detected!")
        test2_passed = False
    
    # Test case 3: Feature names validation
    print("\nTest case 3: Feature names with bigrams")
    clean_features = ['python programming', 'machine learning', 'data science']
    is_valid, leaked = validator.validate_features(clean_features)
    print(f"  Features: {clean_features}")
    print(f"  Valid: {is_valid}, Leaked terms: {leaked}")
    
    if is_valid and len(leaked) == 0:
        print("  ✓ PASS: Clean features correctly validated")
        test3_passed = True
    else:
        print(f"  ✗ FAIL: False positive - clean features flagged")
        test3_passed = False
    
    # Test case 4: Feature names with locations
    print("\nTest case 4: Feature names containing locations")
    leaky_features = ['python programming', 'san francisco', 'data science', 'new york']
    is_valid, leaked = validator.validate_features(leaky_features)
    print(f"  Features: {leaky_features}")
    print(f"  Valid: {is_valid}, Leaked terms: {leaked}")
    
    if not is_valid and len(leaked) > 0:
        print(f"  ✓ PASS: Location leakage in features detected ({len(leaked)} terms)")
        test4_passed = True
    else:
        print(f"  ✗ FAIL: Location terms in features not detected!")
        test4_passed = False
    
    print("\n" + "="*80)
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    if all_passed:
        print("✓ TEST 2 PASSED: All leakage validation tests successful")
    else:
        print("✗ TEST 2 FAILED: Some leakage validation tests failed")
    print("="*80)
    
    return all_passed


def test_integration():
    """Test integration of location removal in preprocessing pipeline."""
    print("\n" + "="*80)
    print("TEST 3: Integration Test")
    print("="*80)
    
    remover = NERLocationRemover(placeholder="<LOCATION>")
    
    # Simulate a resume text with location information
    resume_text = """
    Senior Software Engineer with 8 years of experience in San Francisco Bay Area.
    Worked at Google headquarters in Mountain View, California and Amazon Seattle office.
    Expertise in Python, machine learning, cloud computing (AWS, GCP).
    Managed distributed teams across New York, London, and Singapore.
    Led projects in data science, artificial intelligence, and backend development.
    Education: MS in Computer Science from Stanford University, Palo Alto.
    """
    
    print("\nOriginal resume text:")
    print(resume_text)
    
    # Remove locations
    cleaned_text = remover.remove_locations(resume_text)
    
    print("\nCleaned resume text:")
    print(cleaned_text)
    
    # Detect any remaining location terms
    detected_after = remover.get_detected_locations(cleaned_text)
    
    print(f"\nLocations detected after cleaning: {detected_after}")
    
    # Check that <LOCATION> placeholder is present
    has_placeholder = "<LOCATION>" in cleaned_text
    
    # Check that original location names are NOT in cleaned text
    forbidden_terms = ['francisco', 'mountain view', 'california', 'seattle', 
                       'new york', 'london', 'singapore', 'palo alto', 'stanford']
    remaining_locations = [term for term in forbidden_terms if term.lower() in cleaned_text.lower()]
    
    print(f"\nValidation:")
    print(f"  Placeholder present: {has_placeholder}")
    print(f"  Remaining location terms: {remaining_locations if remaining_locations else 'None'}")
    
    if has_placeholder and len(remaining_locations) == 0 and len(detected_after) == 0:
        print("\n✓ TEST 3 PASSED: Integration test successful")
        print("  - Locations removed")
        print("  - Placeholder inserted")
        print("  - No location terms remain")
        return True
    else:
        print("\n✗ TEST 3 FAILED: Integration test failed")
        if not has_placeholder:
            print("  - Placeholder not found")
        if remaining_locations:
            print(f"  - Location terms still present: {remaining_locations}")
        if detected_after:
            print(f"  - NER still detects locations: {detected_after}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("GEOGRAPHIC LEAKAGE PREVENTION - UNIT TESTS")
    print("="*80)
    
    results = []
    
    try:
        # Test 1: NER location removal
        results.append(("NER Location Removal", test_ner_location_removal()))
    except Exception as e:
        print(f"\n❌ TEST 1 ERROR: {str(e)}")
        results.append(("NER Location Removal", False))
    
    try:
        # Test 2: Leakage validator
        results.append(("Location Leakage Validator", test_leakage_validator()))
    except Exception as e:
        print(f"\n❌ TEST 2 ERROR: {str(e)}")
        results.append(("Location Leakage Validator", False))
    
    try:
        # Test 3: Integration test
        results.append(("Integration Test", test_integration()))
    except Exception as e:
        print(f"\n❌ TEST 3 ERROR: {str(e)}")
        results.append(("Integration Test", False))
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("Geographic leakage prevention is working correctly!")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("Please review and fix the failures before using in production.")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
