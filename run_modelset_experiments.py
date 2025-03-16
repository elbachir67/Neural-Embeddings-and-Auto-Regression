#!/usr/bin/env python3
"""
Run experiments with token pair-based bidirectional validation using ModelSet
"""

import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer, TransformationRule
from modelset_loader import ModelSetLoader
from token_pair_adapter import TokenPairAdapter

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run experiments with ModelSet dataset')
    
    parser.add_argument('--modelset', type=str, required=True,
                        help='Path to the ModelSet dataset')
    
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save experiment results')
    
    parser.add_argument('--experiment', type=str, choices=['basic', 'auto-regression', 'intent', 'all'],
                        default='basic', help='Which experiment to run')
    
    parser.add_argument('--domain', type=str, default=None,
                        help='Filter by domain (e.g., statemachine, class)')
    
    return parser.parse_args()

def run_basic_experiment(modelset_path, output_dir, domain=None):
    """Run basic bidirectional validation experiment"""
    print("\n" + "="*80)
    print("Running Basic Experiment with ModelSet...")
    print("="*80 + "\n")
    
    # Initialize components
    loader = ModelSetLoader(modelset_path)
    adapter = TokenPairAdapter()
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Get transformation pairs
    pairs = loader.get_transformation_pairs("translation", limit=5)
    
    if not pairs:
        print("No suitable transformation pairs found")
        return
    
    print(f"Found {len(pairs)} transformation pairs")
    
    results = []
    
    # Process each pair
    for pair_index, pair in enumerate(pairs):
        print(f"\nProcessing pair {pair_index+1}: {pair['name']}")
        
        # Load source and target models
        source_model = loader.load_model(pair['source'])
        target_model = loader.load_model(pair['target'])
        
        # Convert models to token pairs
        source_token_pairs = adapter.convert_to_token_pairs(source_model)
        target_token_pairs = adapter.convert_to_token_pairs(target_model)
        
        print(f"Converted source model to {len(source_token_pairs)} token pairs")
        print(f"Converted target model to {len(target_token_pairs)} token pairs")
        
        # Create transformation rules
        rules = adapter.create_transformation_rules(source_model, target_model, "translation")
        
        # Add rules to transformer
        for rule in rules:
            transformer.add_rule(rule)
        
        print(f"Created {len(rules)} transformation rules")
        
        # Transform with validation
        transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
            source_model, intent="translation", max_rules=len(rules)
        )
        
        # Get validation scores
        final_scores = validation_scores[-1] if validation_scores else {
            'forward_validation_score': 0,
            'backward_validation_score': 0,
            'transformation_quality': 0
        }
        
        print(f"Forward Validation: {final_scores['forward_validation_score']:.4f}")
        print(f"Backward Validation: {final_scores['backward_validation_score']:.4f}")
        print(f"Transformation Quality: {final_scores['transformation_quality']:.4f}")
        
        # Store results
        results.append({
            'pair_name': pair['name'],
            'source_id': pair['source']['id'],
            'target_id': pair['target']['id'],
            'forward_validation': final_scores['forward_validation_score'],
            'backward_validation': final_scores['backward_validation_score'],
            'transformation_quality': final_scores['transformation_quality'],
            'applied_rules': [rule.id for rule in applied_rules]
        })
        
        # Clear rules for next pair
        transformer.rules_library = []
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'basic_experiment.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    visualize_basic_results(results, output_dir)
    
    return results

def visualize_basic_results(results, output_dir):
    """Create visualization of basic experiment results"""
    if not results:
        return
    
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

def run_auto_regression_experiment(modelset_path, output_dir, domain=None):
    """Run auto-regression experiment"""
    print("\n" + "="*80)
    print("Running Auto-Regression Experiment with ModelSet...")
    print("="*80 + "\n")
    
    # Initialize components
    loader = ModelSetLoader(modelset_path)
    adapter = TokenPairAdapter()
    
    # Get model sequences for UML and Ecore
    uml_sequence = loader.get_model_sequence(domain, "UML", limit=3)
    ecore_sequence = loader.get_model_sequence(domain, "Ecore", limit=3)
    
    if len(uml_sequence) < 2 and len(ecore_sequence) < 2:
        print("Not enough models found for sequences")
        return
    
    # Choose the longer sequence
    sequence = uml_sequence if len(uml_sequence) >= len(ecore_sequence) else ecore_sequence
    
    if len(sequence) < 2:
        print("Not enough models in sequence for auto-regression experiment")
        return
    
    print(f"Using sequence of {len(sequence)} {sequence[0].type} models")
    
    results = []
    
    # Run with and without auto-regression
    print("\nComparing transformation with and without auto-regression:")
    
    # Split sequence: use first n-1 models for history, last model as current
    history_models = sequence[:-1]
    current_model = sequence[-1]
    
    # With auto-regression
    print("\nRunning WITH auto-regression:")
    with_result = run_with_auto_regression(history_models, current_model, adapter)
    
    # Without auto-regression
    print("\nRunning WITHOUT auto-regression:")
    without_result = run_without_auto_regression(current_model, adapter)
    
    # Calculate improvement
    improvement = with_result['quality'] - without_result['quality']
    
    print(f"\nResults comparison:")
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
        'sequence_type': sequence[0].type,
        'sequence_length': len(sequence),
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

def run_with_auto_regression(history_models, current_model, adapter):
    """Run transformation with auto-regression"""
    # Initialize components
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Create transformation rules based on history
    history_rules = []
    for i in range(len(history_models) - 1):
        source = history_models[i]
        target = history_models[i+1]
        
        # Create rules for this history step
        rules = adapter.create_transformation_rules(source, target)
        
        # Add rules to transformer and history
        for rule in rules:
            transformer.add_rule(rule)
            history_rules.append(rule)
    
    print(f"Created {len(history_rules)} rules from transformation history")
    
    # Run transformation with history
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        current_model, 
        intent="revision" if current_model.type == history_models[0].type else "translation", 
        max_rules=len(transformer.rules_library),
        history_models=history_models[:-1],
        history_rules=history_rules
    )
    
    # Get validation scores
    final_scores = validation_scores[-1] if validation_scores else {
        'forward_validation_score': 0,
        'backward_validation_score': 0,
        'transformation_quality': 0
    }
    
    return {
        'forward': final_scores['forward_validation_score'],
        'backward': final_scores['backward_validation_score'],
        'quality': final_scores['transformation_quality'],
        'applied_rules': [rule.id for rule in applied_rules],
        'transformed_model': transformed_model
    }

def run_without_auto_regression(current_model, adapter):
    """Run transformation without auto-regression"""
    # Initialize components
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Create basic rules for the model type
    if current_model.type.lower() == 'uml':
        target_type = 'ecore'
        rules = [
            ("ClassToEClass", "Class", "EClass", "translation", 
             ["UML Classes must be transformed to Ecore EClasses"]),
            ("PropertyToEAttribute", "Property", "EAttribute", "translation", 
             ["UML Properties must be transformed to Ecore EAttributes"]),
            ("AssociationToEReference", "Association", "EReference", "translation", 
             ["UML Associations must be transformed to Ecore EReferences"])
        ]
    else:
        target_type = 'uml'
        rules = [
            ("EClassToClass", "EClass", "Class", "translation", 
             ["Ecore EClasses must be transformed to UML Classes"]),
            ("EAttributeToProperty", "EAttribute", "Property", "translation", 
             ["Ecore EAttributes must be transformed to UML Properties"]),
            ("EReferenceToAssociation", "EReference", "Association", "translation", 
             ["Ecore EReferences must be transformed to UML Associations"])
        ]
    
    # Add rules to transformer
    for rule_id, source_pattern, target_pattern, intent, constraints in rules:
        rule = TransformationRule(rule_id, source_pattern, target_pattern, intent, constraints)
        transformer.add_rule(rule)
    
    print(f"Using {len(rules)} standard rules without history")
    
    # Run transformation without history
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        current_model, 
        intent="translation", 
        max_rules=len(transformer.rules_library)
    )
    
    # Get validation scores
    final_scores = validation_scores[-1] if validation_scores else {
        'forward_validation_score': 0,
        'backward_validation_score': 0,
        'transformation_quality': 0
    }
    
    return {
        'forward': final_scores['forward_validation_score'],
        'backward': final_scores['backward_validation_score'],
        'quality': final_scores['transformation_quality'],
        'applied_rules': [rule.id for rule in applied_rules],
        'transformed_model': transformed_model
    }

def visualize_auto_regression_results(results, output_dir):
    """Create visualization of auto-regression experiment results"""
    if not results:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    seq_types = [f"{r['sequence_type']} (n={r['sequence_length']})" for r in results]
    with_scores = [r['with_auto_regression']['transformation_quality'] for r in results]
    without_scores = [r['without_auto_regression']['transformation_quality'] for r in results]
    improvements = [r['improvement'] * 100 for r in results]  # Convert to percentage
    
    x = np.arange(len(seq_types))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart for scores
    ax1.bar(x - width/2, with_scores, width, label='With Auto-Regression')
    ax1.bar(x + width/2, without_scores, width, label='Without Auto-Regression')
    
    ax1.set_xlabel('Model Sequence')
    ax1.set_ylabel('Transformation Quality')
    ax1.set_title('Impact of Auto-Regression on Transformation Quality')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_types)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1.0)
    
    # Line chart for improvement percentage
    ax2 = ax1.twinx()
    ax2.plot(x, improvements, 'r-o', label='Improvement (%)')
    ax2.set_ylabel('Improvement (%)')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'auto_regression_experiment.png'))
    plt.close()

def run_intent_experiment(modelset_path, output_dir, domain=None):
    """Run intent-aware experiment"""
    print("\n" + "="*80)
    print("Running Intent-Aware Experiment with ModelSet...")
    print("="*80 + "\n")
    
    # Initialize components
    loader = ModelSetLoader(modelset_path)
    adapter = TokenPairAdapter()
    
    # Get pairs for translation and revision
    translation_pairs = loader.get_transformation_pairs("translation", limit=3)
    revision_pairs = loader.get_transformation_pairs("revision", limit=3)
    
    if not translation_pairs and not revision_pairs:
        print("No suitable transformation pairs found")
        return
    
    print(f"Found {len(translation_pairs)} translation pairs and {len(revision_pairs)} revision pairs")
    
    translation_results = []
    revision_results = []
    
    # Process translation pairs
    for pair_index, pair in enumerate(translation_pairs):
        print(f"\nProcessing translation pair {pair_index+1}: {pair['name']}")
        
        # Load source and target models
        source_model = loader.load_model(pair['source'])
        target_model = loader.load_model(pair['target'])
        
        # Run with translation intent
        result = run_with_intent(source_model, target_model, adapter, "translation")
        
        print(f"  TRANSLATION Intent:")
        print(f"    Forward Validation: {result['forward']:.4f}")
        print(f"    Backward Validation: {result['backward']:.4f}")
        print(f"    Transformation Quality: {result['quality']:.4f}")
        
        # Store results
        translation_results.append({
            'pair_name': pair['name'],
            'source_id': pair['source']['id'],
            'target_id': pair['target']['id'],
            'forward_validation': result['forward'],
            'backward_validation': result['backward'],
            'transformation_quality': result['quality']
        })
    
    # Process revision pairs
    for pair_index, pair in enumerate(revision_pairs):
        print(f"\nProcessing revision pair {pair_index+1}: {pair['name']}")
        
        # Load source and target models
        source_model = loader.load_model(pair['source'])
        target_model = loader.load_model(pair['target'])
        
        # Run with revision intent
        result = run_with_intent(source_model, target_model, adapter, "revision")
        
        print(f"  REVISION Intent:")
        print(f"    Forward Validation: {result['forward']:.4f}")
        print(f"    Backward Validation: {result['backward']:.4f}")
        print(f"    Transformation Quality: {result['quality']:.4f}")
        
        # Store results
        revision_results.append({
            'pair_name': pair['name'],
            'source_id': pair['source']['id'],
            'target_id': pair['target']['id'],
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

def run_with_intent(source_model, target_model, adapter, intent):
    """Run transformation with specific intent"""
    # Initialize components
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Create transformation rules with specified intent
    rules = adapter.create_transformation_rules(source_model, target_model, intent)
    
    # Add rules to transformer
    for rule in rules:
        transformer.add_rule(rule)
    
    # Run transformation with intent
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        source_model, 
        intent=intent, 
        max_rules=len(rules)
    )
    
    # Get validation scores
    final_scores = validation_scores[-1] if validation_scores else {
        'forward_validation_score': 0,
        'backward_validation_score': 0,
        'transformation_quality': 0
    }
    
    return {
        'forward': final_scores['forward_validation_score'],
        'backward': final_scores['backward_validation_score'],
        'quality': final_scores['transformation_quality'],
        'applied_rules': [rule.id for rule in applied_rules],
        'transformed_model': transformed_model
    }

def visualize_intent_results(results, output_dir):
    """Create visualization of intent experiment results"""
    if not results or (not results['translation'] and not results['revision']):
        return
    
    plt.figure(figsize=(10, 6))
    
    # Calculate average scores for each intent
    translation_fwd = np.mean([r['forward_validation'] for r in results['translation']]) if results['translation'] else 0
    translation_bwd = np.mean([r['backward_validation'] for r in results['translation']]) if results['translation'] else 0
    translation_qual = np.mean([r['transformation_quality'] for r in results['translation']]) if results['translation'] else 0
    
    revision_fwd = np.mean([r['forward_validation'] for r in results['revision']]) if results['revision'] else 0
    revision_bwd = np.mean([r['backward_validation'] for r in results['revision']]) if results['revision'] else 0
    revision_qual = np.mean([r['transformation_quality'] for r in results['revision']]) if results['revision'] else 0
    
    # Create grouped bar chart
    labels = ['Forward Validation', 'Backward Validation', 'Transformation Quality']
    translation_scores = [translation_fwd, translation_bwd, translation_qual]
    revision_scores = [revision_fwd, revision_bwd, revision_qual]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, translation_scores, width, label='Translation Intent')
    plt.bar(x + width/2, revision_scores, width, label='Revision Intent')
    
    plt.xlabel('Validation Metric')
    plt.ylabel('Average Score')
    plt.title('Comparison of Transformation Intents')
    plt.xticks(x, labels)
    plt.legend()
    plt.ylim(0, 1.0)
    
    # Add value labels
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
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run experiments
    if args.experiment in ['basic', 'all']:
        run_basic_experiment(args.modelset, args.output, args.domain)
    
    if args.experiment in ['auto-regression', 'all']:
        run_auto_regression_experiment(args.modelset, args.output, args.domain)
    
    if args.experiment in ['intent', 'all']:
        run_intent_experiment(args.modelset, args.output, args.domain)
    
    print("\n" + "=" * 80)
    print("Experiments complete! Results saved in", args.output)
    print("=" * 80)

if __name__ == "__main__":
    main()