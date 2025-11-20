"""
Extract disease regions from annotated images to use as reference images.
"""

import os
import sys
from utils import extract_disease_regions

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_references.py <data_dir> <output_dir>")
        print("Example: python extract_references.py data reference_diseases")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist")
        sys.exit(1)
    
    print(f"Extracting disease regions from {data_dir} to {output_dir}...")
    extracted = extract_disease_regions(data_dir, output_dir)
    print(f"\nExtracted {len(extracted)} disease region images")
    print(f"These can now be used as reference images for synthetic data generation")

