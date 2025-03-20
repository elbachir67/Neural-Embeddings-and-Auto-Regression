#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import traceback

print("=== Starting simplified experiment script ===")

# Try to import your modules
try:
    from modelset_loader import ModelSetLoader
    from token_pair_adapter import TokenPairAdapter
    print("Successfully imported custom modules")
except Exception as e:
    print(f"ERROR importing modules: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Check command line arguments
if len(sys.argv) < 3:
    print("Usage: python simple_test.py <modelset_path> <output_dir>")
    sys.exit(1)

modelset_path = sys.argv[1]
output_dir = sys.argv[2]

print(f"ModelSet path: {modelset_path}")
print(f"Output directory: {output_dir}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Try to load the ModelSet
try:
    print("Initializing ModelSetLoader...")
    loader = ModelSetLoader(modelset_path)
    print("ModelSetLoader initialized successfully")
    
    # Get some transformation pairs
    print("Getting transformation pairs...")
    pairs = loader.get_transformation_pairs(limit=2)
    
    if not pairs:
        print("No transformation pairs found")
    else:
        print(f"Found {len(pairs)} transformation pairs:")
        for i, pair in enumerate(pairs):
            print(f"  Pair {i+1}: {pair['name']} - {pair['type']}")
            print(f"    Source: {pair['source']['id']} ({pair['source']['type']})")
            print(f"    Target: {pair['target']['id']} ({pair['target']['type']})")
    
    # Save a simple output file to confirm writing works
    output_file = os.path.join(output_dir, "test_output.txt")
    with open(output_file, "w") as f:
        f.write("Test completed successfully\n")
    print(f"Test output written to {output_file}")
    
except Exception as e:
    print(f"ERROR during execution: {str(e)}")
    traceback.print_exc()

print("=== Script completed ===")