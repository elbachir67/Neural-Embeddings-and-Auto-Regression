#!/usr/bin/env python3
"""
Enhanced Auto-regression Demonstration using Embeddings
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer, TransformationRule
from modelset_loader import ModelSetLoader
from token_pair_adapter import TokenPairAdapter
from embedding_generator import EmbeddingGenerator

def run_embedding_autoregression_demo(modelset_path, output_dir, domain=None):
    """Run an embedding-enhanced autoregression demonstration"""
    print("\nRunning Embedding-Enhanced Auto-Regression Demonstration...")
    
    # Initialize components
    loader = ModelSetLoader(modelset_path)
    adapter = TokenPairAdapter()
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    embedding_generator = EmbeddingGenerator()
    
    # Get a sequence of models from the same domain
    domain = domain or "travel"
    model_sequence = loader.get_model_sequence(domain, "UML", limit=4)
    
    if len(model_sequence) < 3:
        print(f"Not enough models found in domain '{domain}'. Need at least 3 models.")
        print("Try another domain with more models.")
        return None
    
    print(f"Using sequence of {len(model_sequence)} {model_sequence[0].type} models from '{domain}' domain")
    
    # Results to store transformation quality with different approaches
    results = []
    
    # We'll use the last model as our target for transformation
    current_model = model_sequence[-1]
    
    # Get text representation for embedding generation
    current_model_text = get_model_text(current_model)
    current_embedding = embedding_generator.generate_embedding(current_model_text)
    
    # Run baseline (no history, no embeddings)
    print("\n1. Running BASELINE (no history, no embeddings)")
    baseline_result = run_baseline(current_model, encoder, validator, adapter)
    results.append({
        'approach': 'Baseline',
        'forward_validation': baseline_result['forward_validation'],
        'backward_validation': baseline_result['backward_validation'],
        'transformation_quality': baseline_result['transformation_quality']
    })
    
    # Run with autoregression only (no embeddings)
    print("\n2. Running with AUTO-REGRESSION only (no embeddings)")
    history_models = model_sequence[:-1]
    autoregression_result = run_with_autoregression(current_model, history_models, 
                                                 encoder, validator, adapter)
    results.append({
        'approach': 'Auto-Regression',
        'forward_validation': autoregression_result['forward_validation'],
        'backward_validation': autoregression_result['backward_validation'],
        'transformation_quality': autoregression_result['transformation_quality']
    })
    
    # Run with embeddings only (no autoregression)
    print("\n3. Running with EMBEDDINGS only (no autoregression)")
    embedding_result = run_with_embeddings(current_model, current_embedding, 
                                        encoder, validator, adapter, embedding_generator)
    results.append({
        'approach': 'Embeddings',
        'forward_validation': embedding_result['forward_validation'],
        'backward_validation': embedding_result['backward_validation'],
        'transformation_quality': embedding_result['transformation_quality']
    })
    
    # Run with both autoregression and embeddings
    print("\n4. Running with COMBINED auto-regression + embeddings")
    history_embeddings = []
    for model in history_models:
        model_text = get_model_text(model)
        embedding = embedding_generator.generate_embedding(model_text)
        history_embeddings.append(embedding)
    
    combined_result = run_with_combined(current_model, current_embedding, 
                                      history_models, history_embeddings,
                                      encoder, validator, adapter, embedding_generator)
    results.append({
        'approach': 'Combined',
        'forward_validation': combined_result['forward_validation'],
        'backward_validation': combined_result['backward_validation'],
        'transformation_quality': combined_result['transformation_quality']
    })
    
    # Visualize results
    visualize_embedding_autoregression_results(results, output_dir, domain)
    
    return results

def get_model_text(model):
    """Get text representation of a model"""
    return model.to_text()

def run_baseline(model, encoder, validator, adapter):
    """Run transformation with no history and no embeddings"""
    # Create transformer
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Generate basic rules
    rules = adapter.create_transformation_rules(model, model, "revision")
    
    # Add rules to transformer
    for rule in rules:
        transformer.add_rule(rule)
    
    print(f"  Using {len(rules)} basic rules")
    
    # Transform the model
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        model, intent="revision", max_rules=len(rules)
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
            'rules': len(rules),
            'applied_rules': len(applied_rules)
        }
    else:
        print("  No validation scores available")
        return {
            'forward_validation': 0,
            'backward_validation': 0,
            'transformation_quality': 0,
            'rules': len(rules),
            'applied_rules': 0
        }

def run_with_autoregression(model, history_models, encoder, validator, adapter):
    """Run transformation with autoregression but no embeddings"""
    # Create transformer
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Generate rules from history
    history_rules = []
    for i in range(len(history_models) - 1):
        source = history_models[i]
        target = history_models[i+1]
        
        # Create rules for this history step
        rules = adapter.create_transformation_rules(source, target, "revision")
        
        # Add rules to transformer and history
        for rule in rules:
            transformer.add_rule(rule)
            history_rules.append(rule)
    
    print(f"  Created {len(history_rules)} rules from history")
    
    # If no history rules, create default rules
    if not history_rules:
        default_rules = adapter.create_transformation_rules(model, model, "revision")
        for rule in default_rules:
            transformer.add_rule(rule)
        print(f"  No history rules, using {len(default_rules)} default rules")
    
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
            'rules': len(transformer.rules_library),
            'applied_rules': len(applied_rules)
        }
    else:
        print("  No validation scores available")
        return {
            'forward_validation': 0,
            'backward_validation': 0,
            'transformation_quality': 0,
            'rules': len(transformer.rules_library),
            'applied_rules': 0
        }

def run_with_embeddings(model, model_embedding, encoder, validator, adapter, embedding_generator):
    """Run transformation with embeddings but no autoregression"""
    # Create transformer
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Generate basic rules
    rules = adapter.create_transformation_rules(model, model, "revision")
    
    # Add rules to transformer
    for rule in rules:
        transformer.add_rule(rule)
    
    print(f"  Using {len(rules)} basic rules")
    
    # Transform the model
    transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
        model, intent="revision", max_rules=len(rules)
    )
    
    # Apply embedding enhancement to backward validation
    transformed_text = get_model_text(transformed_model)
    transformed_embedding = embedding_generator.generate_embedding(transformed_text)
    
    # Get validation scores and enhance them
    if validation_scores and len(validation_scores) > 0:
        standard_scores = validation_scores[-1]
        
        # Calculate enhanced backward validation using embeddings
        embedding_similarity = embedding_generator.compute_similarity(
            get_model_text(model), get_model_text(transformed_model)
        )
        
        # Blend standard and embedding-based validation (adjust beta as needed)
        beta = 0.7  # Weight for token-pair based score
        enhanced_backward = beta * standard_scores['backward_validation_score'] + (1 - beta) * embedding_similarity
        
        # Calculate enhanced overall quality
        alpha = 0.75  # For revision transformations
        enhanced_quality = alpha * standard_scores['forward_validation_score'] + (1 - alpha) * enhanced_backward
        
        print(f"  Forward Validation: {standard_scores['forward_validation_score']:.4f}")
        print(f"  Standard Backward Validation: {standard_scores['backward_validation_score']:.4f}")
        print(f"  Enhanced Backward Validation: {enhanced_backward:.4f}")
        print(f"  Enhanced Transformation Quality: {enhanced_quality:.4f}")
        print(f"  Embedding Similarity: {embedding_similarity:.4f}")
        print(f"  Applied {len(applied_rules)} rules")
        
        return {
            'forward_validation': standard_scores['forward_validation_score'],
            'backward_validation': enhanced_backward,
            'transformation_quality': enhanced_quality,
            'embedding_similarity': embedding_similarity,
            'standard_backward': standard_scores['backward_validation_score'],
            'rules': len(rules),
            'applied_rules': len(applied_rules)
        }
    else:
        print("  No validation scores available")
        return {
            'forward_validation': 0,
            'backward_validation': 0,
            'transformation_quality': 0,
            'embedding_similarity': 0,
            'standard_backward': 0,
            'rules': len(rules),
            'applied_rules': 0
        }

def run_with_combined(model, model_embedding, history_models, history_embeddings, 
                    encoder, validator, adapter, embedding_generator):
    """Run transformation with both autoregression and embeddings"""
    # Create transformer
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Generate rules from history
    history_rules = []
    for i in range(len(history_models) - 1):
        source = history_models[i]
        target = history_models[i+1]
        
        # Create rules for this history step
        rules = adapter.create_transformation_rules(source, target, "revision")
        
        # Add rules to transformer and history
        for rule in rules:
            transformer.add_rule(rule)
            history_rules.append(rule)
    
    print(f"  Created {len(history_rules)} rules from history")
    
    # If no history rules, create default rules
    if not history_rules:
        default_rules = adapter.create_transformation_rules(model, model, "revision")
        for rule in default_rules:
            transformer.add_rule(rule)
        print(f"  No history rules, using {len(default_rules)} default rules")
    
    # Use embedding similarity to weight rule selection
    if history_embeddings:
        print(f"  Using {len(history_embeddings)} historical embeddings to weight rule selection")
        
        # Transform the model with auto-regression and embedding enhancement
        transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
            model, 
            intent="revision", 
            max_rules=len(transformer.rules_library),
            history_models=history_models,
            history_rules=history_rules
        )
        
        # Apply embedding enhancement to backward validation
        transformed_text = get_model_text(transformed_model)
        transformed_embedding = embedding_generator.generate_embedding(transformed_text)
        
        # Get validation scores and enhance them
        if validation_scores and len(validation_scores) > 0:
            standard_scores = validation_scores[-1]
            
            # Calculate enhanced backward validation using embeddings
            embedding_similarity = embedding_generator.compute_similarity(
                get_model_text(model), get_model_text(transformed_model)
            )
            
            # Blend standard and embedding-based validation (adjust beta as needed)
            beta = 0.7  # Weight for token-pair based score
            enhanced_backward = beta * standard_scores['backward_validation_score'] + (1 - beta) * embedding_similarity
            
            # Calculate enhanced overall quality
            alpha = 0.75  # For revision transformations
            enhanced_quality = alpha * standard_scores['forward_validation_score'] + (1 - alpha) * enhanced_backward
            
            print(f"  Forward Validation: {standard_scores['forward_validation_score']:.4f}")
            print(f"  Standard Backward Validation: {standard_scores['backward_validation_score']:.4f}")
            print(f"  Enhanced Backward Validation: {enhanced_backward:.4f}")
            print(f"  Enhanced Transformation Quality: {enhanced_quality:.4f}")
            print(f"  Embedding Similarity: {embedding_similarity:.4f}")
            print(f"  Applied {len(applied_rules)} rules")
            
            return {
                'forward_validation': standard_scores['forward_validation_score'],
                'backward_validation': enhanced_backward,
                'transformation_quality': enhanced_quality,
                'embedding_similarity': embedding_similarity,
                'standard_backward': standard_scores['backward_validation_score'],
                'rules': len(transformer.rules_library),
                'applied_rules': len(applied_rules)
            }
        else:
            print("  No validation scores available")
            return {
                'forward_validation': 0,
                'backward_validation': 0,
                'transformation_quality': 0,
                'embedding_similarity': 0,
                'standard_backward': 0,
                'rules': len(transformer.rules_library),
                'applied_rules': 0
            }
    else:
        return run_with_autoregression(model, history_models, encoder, validator, adapter)

def visualize_embedding_autoregression_results(results, output_dir, domain=None):
    """Create visualization of embedding-enhanced autoregression results"""
    if not results:
        return
    
    # Extract data
    approaches = [r['approach'] for r in results]
    quality_scores = [r['transformation_quality'] for r in results]
    forward_scores = [r['forward_validation'] for r in results]
    backward_scores = [r['backward_validation'] for r in results]
    
    # Calculate improvements relative to baseline
    baseline_quality = quality_scores[0] if quality_scores else 0
    improvements = []
    for score in quality_scores[1:]:
        improvements.append((score - baseline_quality) * 100)  # As percentage
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First subplot: Scores by approach
    x = np.arange(len(approaches))
    width = 0.25
    
    bars1 = ax1.bar(x - width, forward_scores, width, label='Forward Validation')
    bars2 = ax1.bar(x, backward_scores, width, label='Backward Validation')
    bars3 = ax1.bar(x + width, quality_scores, width, label='Transformation Quality')
    
    ax1.set_xlabel('Approach')
    ax1.set_ylabel('Validation Score')
    ax1.set_title('Validation Scores by Approach')
    ax1.set_xticks(x)
    ax1.set_xticklabels(approaches)
    ax1.legend()
    
    # Set y-axis limits to better show differences
    min_score = min([min(forward_scores), min(backward_scores), min(quality_scores)])
    ax1.set_ylim(max(0.9, min_score - 0.02), 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, score in enumerate(quality_scores):
        ax1.text(i + width, score + 0.002, f'{score:.4f}', ha='center', fontsize=9)
    
    # Second subplot: Improvement from each approach compared to baseline
    if len(improvements) > 0:
        bars = ax2.bar(approaches[1:], improvements, color=['blue', 'green', 'purple'])
        ax2.set_xlabel('Approach')
        ax2.set_ylabel('Improvement over Baseline (%)')
        ax2.set_title('Improvement from Advanced Approaches')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            pos_y = height + 0.05 if height >= 0 else height - 0.15
            ax2.text(bar.get_x() + bar.get_width()/2., pos_y,
                    f'{height:.2f}%',
                    ha='center', fontsize=10)
    
    plt.suptitle(f'Embedding-Enhanced Auto-Regression Evaluation ({domain.capitalize()} Domain)', fontsize=16)
    
    # Adjust layout to accommodate the title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'embedding_autoregression_{domain}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'embedding_autoregression_{domain}.pdf'))
    plt.close()
    
    print(f"Embedding-enhanced auto-regression chart saved to {output_dir}")

def main():
    """Main function to run the demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run embedding-enhanced auto-regression demonstration')
    parser.add_argument('--modelset', type=str, required=True,
                        help='Path to the ModelSet dataset')
    parser.add_argument('--domain', type=str, default=None,
                        help='Domain to use for demonstration (default: try several)')
    parser.add_argument('--output', type=str, default='paper_figures',
                        help='Directory to save visualization figures')
    
    args = parser.parse_args()
    
    if args.domain:
        # Run with specific domain
        run_embedding_autoregression_demo(args.modelset, args.output, args.domain)
    else:
        # Try several domains to find good examples
        domains = ["travel", "library", "statemachine", "class"]
        for domain in domains:
            print(f"\n\n===== Testing {domain.upper()} domain =====")
            run_embedding_autoregression_demo(args.modelset, args.output, domain)

if __name__ == "__main__":
    main()