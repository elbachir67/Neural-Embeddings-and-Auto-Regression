"""
Test script for direct ModelSet integration
"""

from bidirectional_validator import TransformationRule, ContextEncoder, BidirectionalValidator, IntentAwareTransformer
from modelset_direct import SimpleModelSetIntegration

def run_test():
    """Run a simple test with synthetic models"""
    print("=" * 80)
    print("Testing Direct ModelSet Integration")
    print("=" * 80)
    
    # Initialize the direct integration
    integration = SimpleModelSetIntegration('./modelset-dataset')
    
    # Get a transformation pair
    print("\nGetting transformation pair...")
    pair = integration.get_transformation_pair(0, 'translation')
    
    print(f"Pair: {pair['name']} ({pair['type']})")
    
    source_model = pair['source_model']
    target_model = pair['target_model']
    
    print("\nSource Model:")
    print(source_model.to_text())
    
    print("\nTarget Model:")
    print(target_model.to_text())
    
    # Initialize your framework
    print("\nInitializing framework...")
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Create appropriate transformation rules
    rules = []
    if source_model.type.lower() == 'uml' and target_model.type.lower() == 'ecore':
        # UML to Ecore rules
        rules.append(TransformationRule(
            "StateToEClass",
            "State",
            "EClass",
            "translation",
            ["UML States must be transformed to Ecore EClasses"]
        ))
        
        rules.append(TransformationRule(
            "PropertyToEAttribute",
            "Property",
            "EAttribute",
            "translation",
            ["UML Properties must be transformed to Ecore EAttributes"]
        ))
        
        rules.append(TransformationRule(
            "ClassToEClass",
            "Class",
            "EClass",
            "translation",
            ["UML Classes must be transformed to Ecore EClasses"]
        ))
        
        rules.append(TransformationRule(
            "PackageToEPackage",
            "Package",
            "EPackage",
            "translation",
            ["UML Packages must be transformed to Ecore EPackages"]
        ))
    else:
        # Generic rules
        for node_type in set(data['type'] for _, data in source_model.graph.nodes(data=True)):
            target_type = f"E{node_type}" if target_model.type.lower() == 'ecore' else node_type
            rule_id = f"{node_type}To{target_type}"
            
            rules.append(TransformationRule(
                rule_id,
                node_type,
                target_type,
                "translation",
                [f"{source_model.type} {node_type} must be transformed to {target_model.type} {target_type}"]
            ))
    
    # Add rules to transformer
    for rule in rules:
        transformer.add_rule(rule)
        print(f"Added rule: {rule.id}")
    
    # Transform the model
    print("\nTransforming model...")
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        source_model, intent="translation", max_rules=5
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
    
    print("\nTest completed successfully!")
    
    return transformer, integration

if __name__ == "__main__":
    run_test()