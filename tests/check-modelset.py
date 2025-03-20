#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def check_modelset_structure(modelset_path):
    """Check if the modelset directory has the expected structure"""
    modelset_path = Path(modelset_path)
    
    print(f"Checking ModelSet directory: {modelset_path}")
    print(f"Absolute path: {modelset_path.absolute()}")
    
    # Check if directory exists
    if not modelset_path.exists():
        print("ERROR: ModelSet directory does not exist")
        return False
    
    if not modelset_path.is_dir():
        print("ERROR: ModelSet path is not a directory")
        return False
    
    # List contents
    contents = list(modelset_path.iterdir())
    print(f"Directory contains {len(contents)} items:")
    for item in contents:
        print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
    
    # Check for expected subdirectories
    expected_dirs = ['data', 'txt']
    found_dirs = []
    
    for expected_dir in expected_dirs:
        dir_path = modelset_path / expected_dir
        if dir_path.exists() and dir_path.is_dir():
            found_dirs.append(expected_dir)
            print(f"Found expected directory: {expected_dir}")
            
            # Check contents of subdirectory
            subdir_contents = list(dir_path.iterdir())
            print(f"  Contains {len(subdir_contents)} items")
            if len(subdir_contents) > 0:
                print(f"  First few items: {[item.name for item in subdir_contents[:5]]}")
        else:
            print(f"WARNING: Expected directory not found: {expected_dir}")
    
    # Check for required data files
    data_dir = modelset_path / 'data'
    if data_dir.exists():
        required_files = ['categories_uml.csv', 'categories_ecore.csv']
        for req_file in required_files:
            file_path = data_dir / req_file
            if file_path.exists():
                print(f"Found required file: {req_file}")
                # Check file size and first few lines
                print(f"  Size: {file_path.stat().st_size} bytes")
                try:
                    with open(file_path, 'r') as f:
                        first_lines = [next(f) for _ in range(3)]
                    print(f"  First few lines: {first_lines}")
                except Exception as e:
                    print(f"  Error reading file: {str(e)}")
            else:
                print(f"WARNING: Required file not found: {req_file}")
    
    # Summarize
    if all(expected_dir in found_dirs for expected_dir in expected_dirs):
        print("\nStructure check: PASSED - All expected directories found")
        return True
    else:
        print("\nStructure check: FAILED - Some expected directories are missing")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_modelset.py <path_to_modelset>")
        sys.exit(1)
    
    modelset_path = sys.argv[1]
    check_modelset_structure(modelset_path)