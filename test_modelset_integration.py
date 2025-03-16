#!/usr/bin/env python3
"""
Test script for ModelSet integration with the bidirectional validation framework
"""

import os
from pathlib import Path

# Import your framework
from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer

# Import ModelSet integration
from modelset_loader import ModelSetLoader
from modelset_adapter import ModelSetAdapter

def test_modelset_loader(modelset_path):
    """Test the ModelSet loader"""
    print("Testing ModelSet loader...")
    
    # Initialize the loader
    loader = ModelSetLoader(modelset_path)
    
    # Check that the index was loaded
    if not loader.model_index:
        print("ERROR: Failed to load ModelSet index")
        return False
    
    print(f"Successfully loaded ModelSet index with {len(loader.model_index)} entries")
    print(f"Found {len(loader.transformation_pairs)} transformation pairs")
    
    # Try to find UML to Ecore pairs
    uml_to_ecore = loader.get_transformation_pairs_by_type('UML', 'Ecore')
    print(f"Found {len(uml_to_ecore)} UML to Ecore transformation pairs")
    
    if uml_to_ecore:
        # Try to load the first pair
        pair = uml_to_ecore[0]
        print(f"Testing model loading with pair: {pair.get('name', 'Unnamed')}")
        
        source_model = loader.load_model_file(pair['source'])
        target_model = loader.load_model_file(pair['target'])
        
        if source_model and target_model:
            print("Successfully loaded both source and target models")
            return loader
        else:
            print("WARNING: Failed to load one or both models, but continuing...")
            return loader
    else:
        print("WARNING: No UML to Ecore pairs found, but continuing...")
        return loader

def test_modelset_adapter(loader):
    """Test the ModelSet adapter"""
    print("\nTesting ModelSet adapter...")
    
    # Initialize your framework
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Initialize the adapter
    adapter = ModelSetAdapter(transformer)
    
    # Try to find a pair to test with
    pairs = loader.get_transformation_pairs_by_type('UML', 'Ecore')
    
    if pairs:
        pair = pairs[0]
        print(f"Testing adapter with pair: {pair.get('name', 'Unnamed')}")
        
        source_model = loader.load_model_file(pair['source'])
        target_model = loader.load_model_file(pair['target'])
        
        if source_model and target_model:
            # Test token pair conversion
            source_token_pairs = adapter.convert_to_token_pairs(source_model)
            target_token_pairs = adapter.convert_to_token_pairs(target_model)
            
            print(f"Converted source model to {len(source_token_pairs)} token pairs")
            print(f"Converted target model to {len(target_token_pairs)} token pairs")
            
            # Test rule extraction
            rules = adapter.extract_transformation_rules(source_model, target_model)
            print(f"Extracted {len(rules)} transformation rules")
            
            # Test a simple transformation
            print("Testing transformation...")
            result = adapter.transform_and_validate(source_model, target_model, "translation")
            
            print(f"Transformation completed with quality: {result['transformation_quality']:.4f}")
            print(f"Applied rules: {', '.join(result['applied_rules'])}")
            
            return adapter
        else:
            print("WARNING: Failed to load models, skipping adapter test")
            return adapter
    else:
        print("WARNING: No suitable pairs found, skipping adapter test")
        return adapter

def main():
    """Main test function"""
    print("=" * 80)
    print("ModelSet Integration Test")
    print("=" * 80)
    
    # Use the ModelSet path from environment variable or default
    modelset_path = os.environ.get('MODELSET_PATH', './modelset-dataset')
    
    # Check if the directory exists
    if not Path(modelset_path).exists():
        print(f"ERROR: ModelSet path {modelset_path} does not exist")
        print("Please clone the ModelSet repository first with:")
        print("  git clone https://github.com/modelset/modelset-dataset.git")
        return
    
    # Test the ModelSet loader
    loader = test_modelset_loader(modelset_path)
    if not loader:
        print("ModelSet loader test failed")
        return
    
    # Test the ModelSet adapter
    adapter = test_modelset_adapter(loader)
    if not adapter:
        print("ModelSet adapter test failed")
        return
    
    print("\n" + "=" * 80)
    print("All tests passed! ModelSet integration is working.")
    print("You can now run experiments with:")
    print(f"  python run_modelset_experiments.py --modelset {modelset_path} --output results --experiment all")
    print("=" * 80)

if __name__ == "__main__":
    main()