#!/usr/bin/env python3
"""
Simple test script for ModelSet integration
"""

import sys
from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer
from modelset_loader import ModelSetLoader
from token_pair_adapter import TokenPairAdapter

def main():
    """Main test function"""
    # Path to ModelSet repository
    modelset_path = "./modelset-dataset"
    
    # Initialize ModelSet loader
    print("Initializing ModelSet loader...")
    loader = ModelSetLoader(modelset_path)
    
    # Initialize adapter
    adapter = TokenPairAdapter()
    
    # Get transformation pairs
    translation_pairs = loader.get_transformation_pairs("translation", limit=1)
    
    if not translation_pairs:
        print("No translation pairs found in ModelSet")
        return
    
    pair = translation_pairs[0]
    print(f"Testing with pair: {pair['name']}")
    
    # Load source and target models
    source_model = loader.load_model(pair['source'])
    target_model = loader.load_model(pair['target'])
    
    print("\nSource Model:")
    print(source_model.to_text())
    
    print("\nTarget Model:")
    print(target_model.to_text())
    
    # Convert to token pairs
    source_token_pairs = adapter.convert_to_token_pairs(source_model)
    target_token_pairs = adapter.convert_to_token_pairs(target_model)
    
    print(f"\nConverted source model to {len(source_token_pairs)} token pairs")
    print(f"Converted target model to {len(target_token_pairs)} token pairs")
    
    # Create transformation rules
    rules = adapter.create_transformation_rules(source_model, target_model, "translation")
    
    print(f"\nCreated {len(rules)} transformation rules:")
    for rule in rules:
        print(f"- {rule.id}")
    
    # Initialize framework components
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Add rules to transformer
    for rule in rules:
        transformer.add_rule(rule)
    
    # Transform with validation
    print("\nPerforming transformation with bidirectional validation...")
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        source_model, intent="translation", max_rules=len(rules)
    )
    
    print("\nTransformed Model:")
    print(transformed_model.to_text())
    
    print("\nApplied Rules:")
    for rule in applied_rules:
        print(f"- {rule.id}")
    
    print("\nValidation Scores:")
    for i, scores in enumerate(validation_scores):
        print(f"Step {i+1}:")
        print(f"  Forward Validation: {scores['forward_validation_score']:.4f}")
        print(f"  Backward Validation: {scores['backward_validation_score']:.4f}")
        print(f"  Transformation Quality: {scores['transformation_quality']:.4f}")
    
    print("\nModelSet integration test completed successfully!")

if __name__ == "__main__":
    main()