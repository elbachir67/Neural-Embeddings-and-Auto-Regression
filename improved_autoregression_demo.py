#!/usr/bin/env python3
"""
Improved Auto-regression Demonstration
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer, TransformationRule
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
    
    # Get a sequence of models from the same domain
    domain = domain or "travel"
    model_sequence = loader.get_model_sequence(domain, "UML", limit=4)
    
    if len(model_sequence) < 3:
        print(f"Not enough models found in domain '{domain}'. Need at least 3 models.")
        print("Try another domain with more models.")
        return None
    
    print(f"Using sequence of {len(model_sequence)} {model_sequence[0].type} models from '{domain}' domain")
    
    # We'll use the last model as our target for transformation
    current_model = model_sequence[-1]
    
    # Generate basic rules for this model type
    basic_rules = adapter.create_transformation_rules(current_model, current_model, "revision")
    
    # Results to store transformation quality with different history lengths
    results = []
    
    # Run with no history (baseline)
    print("\nRunning WITHOUT auto-regression (baseline)")
    baseline_result = transform_without_history(current_model, basic_rules, encoder, validator)
    results.append({
        'history_length': 0,
        'forward_validation': baseline_result['forward_validation'],
        'backward_validation': baseline_result['backward_validation'],
        'transformation_quality': baseline_result['transformation_quality'],
        'applied_rules': baseline_result['applied_rules']
    })
    
    # Run with history - try different sizes of history
    max_history = min(3, len(model_sequence) - 1)
    
    for i in range(1, max_history + 1):
        # Use i models from history
        history_models = model_sequence[-(i+1):-1]
        print(f"\nRunning with {i} model(s) in history")
        
        history_result = transform_with_history(current_model, history_models, 
                                              basic_rules, encoder, validator, adapter)
        
        results.append({
            'history_length': i,
            'forward_validation': history_result['forward_validation'],
            'backward_validation': history_result['backward_validation'],
            'transformation_quality': history_result['transformation_quality'],
            'applied_rules': history_result['applied_rules']
        })
    
    # Visualize results
    visualize_autoregression_results(results, output_dir, domain)
    
    return results

def transform_without_history(model, basic_rules, encoder, validator):
    """Transform model without using history"""
    
    # Create transformer without history
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Add basic rules
    for rule in basic_rules:
        transformer.add_rule(rule)
    
    print(f"  Using {len(basic_rules)} basic rules")
    
    # Transform the model
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        model, intent="revision", max_rules=len(basic_rules)
    )
    
    # Get validation scores
    if validation_scores:
        final_scores = validation_scores[-1]
        print(f"  Forward Validation: {final_scores['forward_validation_score']:.4f}")
        print(f"  Backward Validation: {final_scores['backward_validation_score']:.4f}")
        print(f"  Transformation Quality: {final_scores['transformation_quality']:.4f}")
        print(f"  Applied {len(applied_rules)} rules")
        
        return {
            'forward_validation': final_scores['forward_validation_score'],
            'backward_validation': final_scores['backward_validation_score'],
            'transformation_quality': final_scores['transformation_quality'],
            'applied_rules': len(applied_rules)
        }
    else:
        print("  No validation scores available")
        return {
            'forward_validation': 0,
            'backward_validation': 0,
            'transformation_quality': 0,
            'applied_rules': 0
        }

def transform_with_history(model, history_models, basic_rules, encoder, validator, adapter):
    """Transform model using history"""
    
    # Create transformer
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Add derived rules from history
    history_rules = []
    
    if len(history_models) >= 2:
        # Create rules from pairs of models in history
        for i in range(len(history_models) - 1):
            source = history_models[i]
            target = history_models[i+1]
            
            # Create rules for this history step
            pair_rules = adapter.create_transformation_rules(source, target, "revision")
            
            # Add rules to transformer and history
            for rule in pair_rules:
                transformer.add_rule(rule)
                history_rules.append(rule)
        
        print(f"  Created {len(history_rules)} rules from transformation history")
    elif len(history_models) == 1:
        # For a single history model, derive rules by comparing with current model
        history_model = history_models[0]
        pair_rules = adapter.create_transformation_rules(history_model, model, "revision")
        
        for rule in pair_rules:
            transformer.add_rule(rule)
            history_rules.append(rule)
            
        print(f"  Created {len(history_rules)} rules by analyzing current and history model")
    
    # If no history rules could be created, use basic rules
    if not history_rules:
        for rule in basic_rules:
            transformer.add_rule(rule)
        print(f"  Falling back to {len(basic_rules)} basic rules")
    
    # Transform the model
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        model, 
        intent="revision", 
        max_rules=len(transformer.rules_library),
        history_models=history_models,
        history_rules=history_rules
    )
    
    # Get validation scores
    if validation_scores:
        final_scores = validation_scores[-1]
        print(f"  Forward Validation: {final_scores['forward_validation_score']:.4f}")
        print(f"  Backward Validation: {final_scores['backward_validation_score']:.4f}")
        print(f"  Transformation Quality: {final_scores['transformation_quality']:.4f}")
        print(f"  Applied {len(applied_rules)} rules")
        
        return {
            'forward_validation': final_scores['forward_validation_score'],
            'backward_validation': final_scores['backward_validation_score'],
            'transformation_quality': final_scores['transformation_quality'],
            'applied_rules': len(applied_rules)
        }
    else:
        print("  No validation scores available")
        return {
            'forward_validation': 0,
            'backward_validation': 0,
            'transformation_quality': 0,
            'applied_rules': 0
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
    applied_rules = [r['applied_rules'] for r in results]
    
    # Calculate improvements relative to baseline (history_length=0)
    baseline_quality = quality_scores[0] if quality_scores else 0
    improvements = []
    for score in quality_scores[1:]:
        improvements.append((score - baseline_quality) * 100)  # As percentage
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First subplot: Validation scores by history length
    x = np.arange(len(history_lengths))
    width = 0.25
    
    bars1 = ax1.bar(x - width, forward_scores, width, label='Forward Validation')
    bars2 = ax1.bar(x, backward_scores, width, label='Backward Validation')
    bars3 = ax1.bar(x + width, quality_scores, width, label='Transformation Quality')
    
    ax1.set_xlabel('History Length (# of models)')
    ax1.set_ylabel('Validation Score')
    ax1.set_title('Impact of History Length on Validation Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(history_lengths)
    ax1.legend()
    
    # Set y-axis limits to better show differences
    min_score = min([min(forward_scores), min(backward_scores), min(quality_scores)])
    ax1.set_ylim(max(0.9, min_score - 0.02), 1.0)
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
            pos_y = height + 0.05 if height >= 0 else height - 0.15
            ax2.text(bar.get_x() + bar.get_width()/2., pos_y,
                    f'{height:.2f}%',
                    ha='center', fontsize=10)
    
    plt.suptitle(f'Auto-Regression Mechanism Evaluation ({domain.capitalize()} Domain)', fontsize=16)
    
    # Adjust layout to accommodate the title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'autoregression_demo_{domain}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'autoregression_demo_{domain}.pdf'))
    plt.close()
    
    print(f"Auto-regression demonstration chart saved to {output_dir}")

def main():
    """Main function to run the demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run auto-regression demonstration')
    parser.add_argument('--modelset', type=str, required=True,
                        help='Path to the ModelSet dataset')
    parser.add_argument('--domain', type=str, default=None,
                        help='Domain to use for demonstration (default: try several)')
    parser.add_argument('--output', type=str, default='paper_figures',
                        help='Directory to save visualization figures')
    
    args = parser.parse_args()
    
    if args.domain:
        # Run with specific domain
        run_autoregression_demo(args.modelset, args.output, args.domain)
    else:
        # Try several domains to find good examples
        domains = ["travel", "library", "statemachine", "class"]
        for domain in domains:
            print(f"\n\n===== Testing {domain.upper()} domain =====")
            run_autoregression_demo(args.modelset, args.output, domain)

if __name__ == "__main__":
    main()