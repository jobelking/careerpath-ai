"""
Script to merge overlapping career classes in the preprocessed dataset.

Merges:
1. finance, accountant, banking → Finance & Accounting Careers
2. arts, designer, apparel → Design & Creative Careers

This reduces semantic confusion and improves recall for problem classes.
"""

import pandas as pd
import os

def merge_overlapping_classes(input_path: str, output_path: str):
    """
    Merge semantically overlapping career classes.
    
    Args:
        input_path: Path to original preprocessed CSV
        output_path: Path to save merged CSV
    """
    print("="*60)
    print("MERGING OVERLAPPING CAREER CLASSES")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Original samples: {len(df)}")
    print(f"Original classes: {df['path'].nunique()}")
    
    # Define merge mappings
    merge_map = {
        # Financial services cluster
        'finance': 'finance-accounting',
        'accountant': 'finance-accounting',
        'banking': 'finance-accounting',
        
        # Creative/design cluster  
        'arts': 'design-creative',
        'designer': 'design-creative',
        'apparel': 'design-creative',
        
        # Additional merges to boost existing classes (Feb 2026)
        'java developer': 'software engineer',      # 84 samples → software engineer
        'testing': 'quality assurance',             # 70 samples → quality assurance
        'devops engineer': 'devops',                # 55 samples → devops
    }
    
    # Count before merge
    print("\n" + "-"*60)
    print("BEFORE MERGE:")
    print("-"*60)
    for old_class in merge_map.keys():
        count = len(df[df['path'] == old_class])
        print(f"  {old_class}: {count} samples")
    
    # Apply merges
    print("\n" + "-"*60)
    print("APPLYING MERGES:")
    print("-"*60)
    
    merged_counts = {}
    for old_class, new_class in merge_map.items():
        mask = df['path'] == old_class
        count = mask.sum()
        if count > 0:
            df.loc[mask, 'path'] = new_class
            merged_counts[new_class] = merged_counts.get(new_class, 0) + count
            print(f"  {old_class} ({count}) → {new_class}")
    
    print("\n" + "-"*60)
    print("AFTER MERGE:")
    print("-"*60)
    for new_class, count in merged_counts.items():
        print(f"  {new_class}: {count} samples (combined)")
    
    # Final stats
    print("\n" + "-"*60)
    print("FINAL DATASET STATS:")
    print("-"*60)
    print(f"  Total samples: {len(df)}")
    print(f"  Total classes: {df['path'].nunique()} (was 30)")
    
    # Show all class counts
    print("\n" + "-"*60)
    print("ALL CLASS COUNTS:")
    print("-"*60)
    for path, count in df['path'].value_counts().sort_values(ascending=False).items():
        print(f"  {path}: {count}")
    
    # Save merged dataset
    df.to_csv(output_path, index=False)
    print(f"\n✓ Merged dataset saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'datasets', 'orig_dataset_careerpath-ai_preprocessed.csv')
    output_path = os.path.join(base_dir, 'data', 'datasets', 'merged_dataset_careerpath-ai_preprocessed.csv')
    
    # Run merge
    merge_overlapping_classes(input_path, output_path)
