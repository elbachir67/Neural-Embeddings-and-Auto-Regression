"""
Run experiments with direct ModelSet integration
"""

import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from bidirectional_validator import TransformationRule, ContextEncoder, BidirectionalValidator, IntentAwareTransformer
from modelset_direct import SimpleModelSetIntegration

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run experiments with ModelSet dataset')
    
    parser.add_argument('--modelset', type=str, default='./modelset-dataset',
                        help='Path to the ModelSet dataset repository')
    
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save experiment results')
    
    parser.add_argument('--experiment', type=str, choices=['basic', 'auto-regression', 'intent', 'all'],
                        default='basic', help='Which experiment to run')
    
    return parser.parse_args()

def initialize_framework():
    """Initialize your bidirectional validation framework"""
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    return transformer

def run_basic_experiment(modelset_path, output_dir):
    """Run basic bidirectional validation experiment"""
    print("\n" + "="*80)
    print("Running Basic Experiment...")
    print("="*80 + "\n")
    
    # Initialize framework and integration
    transformer = initialize_framework()
    integration = SimpleModelSetIntegration(modelset_path)
    
    # Get transformation pairs
    results = []
    
    for i in range(3):  # Run with 3 different pairs
        print(f"\nTransformation pair {i+1}:")
        
        # Get a pair, alternating between translation and revision
        pair_type = "translation" if i % 2 == 0 else "revision"
        pair = integration.get_transformation_pair(i, pair_type)
        
        print(f"  Pair: {pair['name']} ({pair['type']})")
        
        source_model = pair['source_model']
        target_model = pair['target_model']
        
        # Create appropriate transformation rules
        rules = create_rules_for_pair(source_model, target_model, pair['type'])
        
        # Add rules to transformer
        for rule in rules:
            transformer.add_rule(rule)
        
        # Transform the model
        print("  Transforming model...")
        transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
            source_model, intent=pair['type'], max_rules=5
        )
        
        # Get the final scores
        final_scores = validation_scores[-1] if validation_scores else {
            'forward_validation_score': 0,
            'backward_validation_score': 0,
            'transformation_quality': 0
        }
        
        print(f"  Forward Validation: {final_scores['forward_validation_score']:.4f}")
        print(f"  Backward Validation: {final_scores['backward_validation_score']:.4f}")
        print(f"  Transformation Quality: {final_scores['transformation_quality']:.4f}")
        
        # Store results
        results.append({
            'pair_name': pair['name'],
            'pair_type': pair['type'],
            'forward_validation': final_scores['forward_validation_score'],
            'backward_validation': final_scores['backward_validation_score'],
            'transformation_quality': final_scores['transformation_quality'],
            'applied_rules': [rule.id for rule in applied_rules]
        })
        
        # Clear rules for the next pair
        transformer.rules_library = []
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'basic_experiment.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    visualize_basic_results(results, output_dir)
    
    return results

def create_rules_for_pair(source_model, target_model, intent):
    """Create appropriate transformation rules for a model pair"""
    rules = []
    
    # Define rules based on model types
    if source_model.type.lower() == 'uml' and target_model.type.lower() == 'ecore':
        # UML to Ecore rules
        rules.append(TransformationRule(
            "StateToEClass",
            "State",
            "EClass",
            intent,
            ["UML States must be transformed to Ecore EClasses"]
        ))
        
        rules.append(TransformationRule(
            "PropertyToEAttribute",
            "Property",
            "EAttribute",
            intent,
            ["UML Properties must be transformed to Ecore EAttributes"]
        ))
        
        rules.append(TransformationRule(
            "ClassToEClass",
            "Class",
            "EClass",
            intent,
            ["UML Classes must be transformed to Ecore EClasses"]
        ))
        
        rules.append(TransformationRule(
            "PackageToEPackage",
            "Package",
            "EPackage",
            intent,
            ["UML Packages must be transformed to Ecore EPackages"]
        ))
        
        rules.append(TransformationRule(
            "AssociationToEReference",
            "Association",
            "EReference",
            intent,
            ["UML Associations must be transformed to Ecore EReferences"]
        ))
    elif source_model.type == target_model.type:
        # Revision rules within the same metamodel
        if source_model.type.lower() == 'uml':
            rules.append(TransformationRule(
                "AddGuardCondition",
                "Transition",
                "Transition",
                intent,
                ["Add guard condition to transitions"]
            ))
            
            rules.append(TransformationRule(
                "AddEventHandler",
                "State",
                "State",
                intent,
                ["Add event handlers to states"]
            ))
        elif source_model.type.lower() == 'ecore':
            rules.append(TransformationRule(
                "AddEOperation",
                "EClass",
                "EClass",
                intent,
                ["Add operations to EClasses"]
            ))
            
            rules.append(TransformationRule(
                "MakeEReferencesComposition",
                "EReference",
                "EReference",
                intent,
                ["Make EReferences composition relationships"]
            ))
    else:
        # Generic rules for other metamodel combinations
        for node_type in set(data['type'] for _, data in source_model.graph.nodes(data=True)):
            target_type = f"E{node_type}" if target_model.type.lower() == 'ecore' else node_type
            rule_id = f"{node_type}To{target_type}"
            
            rules.append(TransformationRule(
                rule_id,
                node_type,
                target_type,
                intent,
                [f"{source_model.type} {node_type} must be transformed to {target_model.type} {target_type}"]
            ))
    
    return rules

def visualize_basic_results(results, output_dir):
    """Create visualization of basic experiment results"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    names = [r['pair_name'][:15] + '...' if len(r['pair_name']) > 15 else r['pair_name'] for r in results]
    fwd_scores = [r['forward_validation'] for r in results]
    bwd_scores = [r['backward_validation'] for r in results]
    quality = [r['transformation_quality'] for r in results]
    
    x = np.arange(len(names))
    width = 0.25
    
    # Create bars
    plt.bar(x - width, fwd_scores, width, label='Forward Validation')
    plt.bar(x, bwd_scores, width, label='Backward Validation')
    plt.bar(x + width, quality, width, label='Transformation Quality')
    
    # Add labels and legend
    plt.xlabel('Model Pairs')
    plt.ylabel('Score')
    plt.title('Bidirectional Validation Scores for Model Pairs')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'basic_experiment.png'))
    plt.close()

def run_auto_regression_experiment(modelset_path, output_dir):
    """Run auto-regression experiment"""
    print("\n" + "="*80)
    print("Running Auto-Regression Experiment...")
    print("="*80 + "\n")
    
    # Initialize integration
    integration = SimpleModelSetIntegration(modelset_path)
    
    # Get model sequences
    results = []
    
    # Process multiple sequences
    for seq_index in range(2):
        print(f"\nSequence {seq_index+1}:")
        
        # Get a sequence
        sequence = integration.get_model_sequence(seq_index)
        print(f"  Sequence: {sequence['name']}")
        print(f"  Contains {len(sequence['models'])} models")
        
        if len(sequence['models']) < 2:
            print("  Not enough models in sequence, skipping...")
            continue
        
        # Use the first n-1 models as history and the last one as current
        history_models = sequence['models'][:-1]
        current_model = sequence['models'][-1]
        
        # Run WITH auto-regression
        print("  Running WITH auto-regression...")
        with_result = run_with_auto_regression(history_models, current_model)
        
        # Run WITHOUT auto-regression
        print("  Running WITHOUT auto-regression...")
        without_result = run_without_auto_regression(current_model)
        
        # Calculate improvement
        improvement = with_result['quality'] - without_result['quality']
        
        print(f"  WITH Auto-Regression:")
        print(f"    Forward Validation: {with_result['forward']:.4f}")
        print(f"    Backward Validation: {with_result['backward']:.4f}")
        print(f"    Transformation Quality: {with_result['quality']:.4f}")
        
        print(f"  WITHOUT Auto-Regression:")
        print(f"    Forward Validation: {without_result['forward']:.4f}")
        print(f"    Backward Validation: {without_result['backward']:.4f}")
        print(f"    Transformation Quality: {without_result['quality']:.4f}")
        
        print(f"  Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
        
        # Store results
        results.append({
            'sequence_name': sequence['name'],
            'with_auto_regression': {
                'forward_validation': with_result['forward'],
                'backward_validation': with_result['backward'],
                'transformation_quality': with_result['quality']
            },
            'without_auto_regression': {
                'forward_validation': without_result['forward'],
                'backward_validation': without_result['backward'],
                'transformation_quality': without_result['quality']
            },
            'improvement': improvement
        })
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'auto_regression_experiment.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    visualize_auto_regression_results(results, output_dir)
    
    return results

def run_with_auto_regression(history_models, current_model):
    """Run transformation with auto-regression"""
    # Initialize framework
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Create rules based on historical models
    history_rules = []
    for i in range(len(history_models) - 1):
        source = history_models[i]
        target = history_models[i+1]
        intent = "revision" if source.type == target.type else "translation"
        
        rules = create_rules_for_pair(source, target, intent)
        for rule in rules:
            transformer.add_rule(rule)
            history_rules.append(rule)
    
    # Transform with history
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        current_model, 
        intent="translation", 
        max_rules=5,
        history_models=history_models,
        history_rules=history_rules
    )
    
    # Get final scores
    final_scores = validation_scores[-1] if validation_scores else {
        'forward_validation_score': 0,
        'backward_validation_score': 0,
        'transformation_quality': 0
    }
    
    return {
        'forward': final_scores['forward_validation_score'],
        'backward': final_scores['backward_validation_score'],
        'quality': final_scores['transformation_quality']
    }

def run_without_auto_regression(current_model):
    """Run transformation without auto-regression"""
    # Initialize framework
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Create generic rules
    rules = []
    for node_type in set(data['type'] for _, data in current_model.graph.nodes(data=True)):
        target_type = node_type  # For simplicity, use same type in target
        rule_id = f"{node_type}ToTarget"
        
        rules.append(TransformationRule(
            rule_id,
            node_type,
            target_type,
            "translation",
            [f"Convert {node_type} to target model"]
        ))
    
    # Add rules to transformer
    for rule in rules:
        transformer.add_rule(rule)
    
    # Transform without history
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        current_model, 
        intent="translation", 
        max_rules=5
    )
    
    # Get final scores
    final_scores = validation_scores[-1] if validation_scores else {
        'forward_validation_score': 0,
        'backward_validation_score': 0,
        'transformation_quality': 0
    }
    
    return {
        'forward': final_scores['forward_validation_score'],
        'backward': final_scores['backward_validation_score'],
        'quality': final_scores['transformation_quality']
    }

def visualize_auto_regression_results(results, output_dir):
    """Create visualization of auto-regression experiment results"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    names = [r['sequence_name'][:15] + '...' if len(r['sequence_name']) > 15 else r['sequence_name'] 
            for r in results]
    with_quality = [r['with_auto_regression']['transformation_quality'] for r in results]
    without_quality = [r['without_auto_regression']['transformation_quality'] for r in results]
    improvements = [r['improvement'] * 100 for r in results]  # To percentage
    
    x = np.arange(len(names))
    width = 0.35
    
    # Create bars for quality scores
    ax1 = plt.subplot(111)
    ax1.bar(x - width/2, with_quality, width, label='With Auto-Regression')
    ax1.bar(x + width/2, without_quality, width, label='Without Auto-Regression')
    
    ax1.set_xlabel('Model Sequences')
    ax1.set_ylabel('Transformation Quality')
    ax1.set_title('Impact of Auto-Regression on Transformation Quality')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1.0)
    
    # Add secondary y-axis for improvement percentage
    ax2 = ax1.twinx()
    ax2.plot(x, improvements, 'r-o', label='Improvement (%)')
    ax2.set_ylabel('Improvement (%)')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'auto_regression_experiment.png'))
    plt.close()

def run_intent_experiment(modelset_path, output_dir):
    """Run intent-aware experiment"""
    print("\n" + "="*80)
    print("Running Intent-Aware Experiment...")
    print("="*80 + "\n")
    
    # Initialize integration
    integration = SimpleModelSetIntegration(modelset_path)
    
    # Get pairs for translation and revision
    translation_results = []
    revision_results = []
    
    # Process translation pairs
    for i in range(2):
        print(f"\nTranslation pair {i+1}:")
        
        # Get a translation pair
        pair = integration.get_transformation_pair(i, 'translation')
        print(f"  Pair: {pair['name']} ({pair['type']})")
        
        source_model = pair['source_model']
        target_model = pair['target_model']
        
        # Run with translation intent
        result = run_with_intent(source_model, target_model, "translation")
        
        print(f"  TRANSLATION Intent:")
        print(f"    Forward Validation: {result['forward']:.4f}")
        print(f"    Backward Validation: {result['backward']:.4f}")
        print(f"    Transformation Quality: {result['quality']:.4f}")
        
        # Store results
        translation_results.append({
            'pair_name': pair['name'],
            'forward_validation': result['forward'],
            'backward_validation': result['backward'],
            'transformation_quality': result['quality']
        })
    
    # Process revision pairs
    for i in range(2):
        print(f"\nRevision pair {i+1}:")
        
        # Get a revision pair
        pair = integration.get_transformation_pair(i, 'revision')
        print(f"  Pair: {pair['name']} ({pair['type']})")
        
        source_model = pair['source_model']
        target_model = pair['target_model']
        
        # Run with revision intent
        result = run_with_intent(source_model, target_model, "revision")
        
        print(f"  REVISION Intent:")
        print(f"    Forward Validation: {result['forward']:.4f}")
        print(f"    Backward Validation: {result['backward']:.4f}")
        print(f"    Transformation Quality: {result['quality']:.4f}")
        
        # Store results
        revision_results.append({
            'pair_name': pair['name'],
            'forward_validation': result['forward'],
            'backward_validation': result['backward'],
            'transformation_quality': result['quality']
        })
    
    # Combine results
    results = {
        'translation': translation_results,
        'revision': revision_results
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'intent_experiment.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    visualize_intent_results(results, output_dir)
    
    return results

def run_with_intent(source_model, target_model, intent):
    """Run transformation with specific intent"""
    # Initialize framework
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Create rules based on intent
    rules = create_rules_for_pair(source_model, target_model, intent)
    
    # Add rules to transformer
    for rule in rules:
        transformer.add_rule(rule)
    
    # Transform with intent
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        source_model, 
        intent=intent, 
        max_rules=5
    )
    
    # Get final scores
    final_scores = validation_scores[-1] if validation_scores else {
        'forward_validation_score': 0,
        'backward_validation_score': 0,
        'transformation_quality': 0
    }
    
    return {
        'forward': final_scores['forward_validation_score'],
        'backward': final_scores['backward_validation_score'],
        'quality': final_scores['transformation_quality']
    }

def visualize_intent_results(results, output_dir):
    """Create visualization of intent experiment results"""
    plt.figure(figsize=(10, 6))
    
    # Calculate averages
    translation_fwd = np.mean([r['forward_validation'] for r in results['translation']]) if results['translation'] else 0
    translation_bwd = np.mean([r['backward_validation'] for r in results['translation']]) if results['translation'] else 0
    translation_quality = np.mean([r['transformation_quality'] for r in results['translation']]) if results['translation'] else 0
    
    revision_fwd = np.mean([r['forward_validation'] for r in results['revision']]) if results['revision'] else 0
    revision_bwd = np.mean([r['backward_validation'] for r in results['revision']]) if results['revision'] else 0
    revision_quality = np.mean([r['transformation_quality'] for r in results['revision']]) if results['revision'] else 0
    
    # Set up chart
    labels = ['Forward Validation', 'Backward Validation', 'Transformation Quality']
    translation_scores = [translation_fwd, translation_bwd, translation_quality]
    revision_scores = [revision_fwd, revision_bwd, revision_quality]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, translation_scores, width, label='Translation Intent')
    plt.bar(x + width/2, revision_scores, width, label='Revision Intent')
    
    plt.xlabel('Validation Metric')
    plt.ylabel('Score')
    plt.title('Comparison of Transformation Intents')
    plt.xticks(x, labels)
    plt.legend()
    plt.ylim(0, 1.0)
    
    # Add value labels above bars
    for i, v in enumerate(translation_scores):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(revision_scores):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'intent_experiment.png'))
    plt.close()

def main():
    """Main function to run experiments"""
    args = parse_args()
    
    print("=" * 80)
    print("Token Pair Bidirectional Validation Framework")
    print("ModelSet Dataset Integration")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Run the selected experiment
    if args.experiment in ['basic', 'all']:
        run_basic_experiment(args.modelset, args.output)
    
    if args.experiment in ['auto-regression', 'all']:
        run_auto_regression_experiment(args.modelset, args.output)
    
    if args.experiment in ['intent', 'all']:
        run_intent_experiment(args.modelset, args.output)
    
    print("\n" + "=" * 80)
    print("Experiments complete! Results saved in", args.output)
    print("=" * 80)

if __name__ == "__main__":
    main()