#!/usr/bin/env python3
"""
Demonstrate the autoregressive mechanism from the paper
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer
from modelset_loader import ModelSetLoader
from token_pair_adapter import TokenPairAdapter

def run_autoregression_demo(modelset_path, output_dir, domain=None):
    """Run an autoregression demonstration with visualization"""
    print("\nRunning Auto-Regression Demonstration...")
    
    # Initialize components
    loader = ModelSetLoader(modelset_path)
    adapter = TokenPairAdapter()
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    
    # Get a sequence of models from the same domain (preferably 'travel' or another with good data)
    domain = domain or "travel"
    model_sequence = loader.get_model_sequence(domain, "UML", limit=4)
    
    if len(model_sequence) < 3:
        print(f"Not enough models found in domain '{domain}'. Need at least 3 models.")
        return None
    
    print(f"Using sequence of {len(model_sequence)} {model_sequence[0].type} models from '{domain}' domain")
    
    results = []
    
    # Run transformations with varying history lengths
    for history_length in range(3):
        result = run_with_history_length(model_sequence, history_length, encoder, validator, adapter)
        results.append(result)
    
    # Visualize results
    visualize_autoregression_results(results, output_dir, domain)
    
    return results

def run_with_history_length(model_sequence, history_length, encoder, validator, adapter):
    """Run transformation with specified history length"""
    # Clear the last model for transformation
    current_model = model_sequence[-1]
    
    # Use specified history length (0 = no history)
    if history_length == 0:
        history_models = []
        print("\nRunning WITHOUT auto-regression (no history)")
    else:
        history_models = model_sequence[-history_length-1:-1]
        print(f"\nRunning with {history_length} models in history")
    
    # Create transformer
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Add rules from history if available
    history_rules = []
    if history_models:
        for i in range(len(history_models) - 1):
            source = history_models[i]
            target = history_models[i+1]
            
            # Create rules for this history step
            rules = adapter.create_transformation_rules(source, target, "revision")
            
            # Add rules to transformer and history
            for rule in rules:
                transformer.add_rule(rule)
                history_rules.append(rule)
        
        print(f"  Created {len(history_rules)} rules from transformation history")
    else:
        # Add basic rules for revision transformation
        rules = adapter.create_transformation_rules(current_model, current_model, "revision")
        for rule in rules:
            transformer.add_rule(rule)
        print(f"  Created {len(rules)} basic rules (no history)")
    
    # Run transformation
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        current_model, 
        intent="revision", 
        max_rules=len(transformer.rules_library),
        history_models=history_models,
        history_rules=history_rules
    )
    
    # Get validation scores
    final_scores = validation_scores[-1] if validation_scores else {
        'forward_validation_score': 0,
        'backward_validation_score': 0,
        'transformation_quality': 0
    }
    
    print(f"  Forward Validation: {final_scores['forward_validation_score']:.4f}")
    print(f"  Backward Validation: {final_scores['backward_validation_score']:.4f}")
    print(f"  Transformation Quality: {final_scores['transformation_quality']:.4f}")
    
    return {
        'history_length': history_length,
        'forward_validation': final_scores['forward_validation_score'],
        'backward_validation': final_scores['backward_validation_score'],
        'transformation_quality': final_scores['transformation_quality'],
        'applied_rules': len(applied_rules)
    }

def visualize_autoregression_results(results, output_dir, domain=None):
    """Create visualization of auto-regression results"""
    if not results:
        return
    
    # Extract data
    history_lengths = [r['history_length'] for r in results]
    quality_scores = [r['transformation_quality'] for r in results]
    forward_scores = [r['forward_validation'] for r in results]
    backward_scores = [r['backward_validation'] for r in results]
    
    # Calculate improvements
    improvements = []
    for i in range(1, len(quality_scores)):
        improvements.append((quality_scores[i] - quality_scores[0]) * 100)  # As percentage
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First subplot: Validation scores by history length
    x = np.arange(len(history_lengths))
    width = 0.25
    
    ax1.bar(x - width, forward_scores, width, label='Forward Validation')
    ax1.bar(x, backward_scores, width, label='Backward Validation')
    ax1.bar(x + width, quality_scores, width, label='Transformation Quality')
    
    ax1.set_xlabel('History Length (# of models)')
    ax1.set_ylabel('Validation Score')
    ax1.set_title('Impact of History Length on Validation Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(history_lengths)
    ax1.legend()
    ax1.set_ylim(0.9, 1.0)  # Adjusted to see differences better
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, score in enumerate(quality_scores):
        ax1.text(i + width, score + 0.002, f'{score:.4f}', ha='center', fontsize=9)
    
    # Second subplot: Improvement from auto-regression
    if len(improvements) > 0:
        bars = ax2.bar(history_lengths[1:], improvements, color='green')
        ax2.set_xlabel('History Length (# of models)')
        ax2.set_ylabel('Improvement in Quality (%)')
        ax2.set_title('Improvement from Auto-Regression')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}%',
                    ha='center', fontsize=10)
    
    plt.suptitle(f'Auto-Regression Mechanism Evaluation ({domain.capitalize()} Domain)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'autoregression_demo.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'autoregression_demo.pdf'))
    plt.close()
    
    print(f"Auto-regression demonstration chart saved to {output_dir}")

def main():
    """Main function to run the demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run auto-regression demonstration')
    parser.add_argument('--modelset', type=str, required=True,
                        help='Path to the ModelSet dataset')
    parser.add_argument('--domain', type=str, default='travel',
                        help='Domain to use for demonstration (default: travel)')
    parser.add_argument('--output', type=str, default='paper_figures',
                        help='Directory to save visualization figures')
    
    args = parser.parse_args()
    
    # Run demonstration
    run_autoregression_demo(args.modelset, args.output, args.domain)

if __name__ == "__main__":
    main()