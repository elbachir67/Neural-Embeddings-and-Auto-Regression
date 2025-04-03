#!/usr/bin/env python3
"""
Simplified Parameter Optimization Script for Alpha and Beta
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import your existing components
from bidirectional_validator import ContextEncoder, BidirectionalValidator, TransformationRule
from modelset_loader import ModelSetLoader
from embedding_generator import EmbeddingGenerator
from token_pair_adapter import TokenPairAdapter

def main():
    """Main function to run parameter optimization"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Optimize alpha and beta parameters')
    parser.add_argument('--modelset', type=str, default='./modelset-dataset', 
                        help='Path to the ModelSet dataset')
    parser.add_argument('--output', type=str, default='results/param_opt',
                        help='Directory to save optimization results')
    args = parser.parse_args()
    
    print("Starting parameter optimization...")
    print(f"ModelSet path: {args.modelset}")
    print(f"Output directory: {args.output}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    figures_dir = os.path.join(args.output, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    loader = ModelSetLoader(args.modelset)
    adapter = TokenPairAdapter()
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    embedding_generator = EmbeddingGenerator()
    
    # Set alpha and beta ranges to test
    alpha_values = np.arange(0.4, 0.8, 0.1)
    beta_values = np.arange(0.5, 0.9, 0.1)
    
    print(f"Alpha values to test: {alpha_values}")
    print(f"Beta values to test: {beta_values}")
    
    # Load a small number of model pairs for testing
    print("Loading model pairs...")
    try:
        translation_pairs = loader.get_transformation_pairs("translation", limit=3)
        revision_pairs = loader.get_transformation_pairs("revision", limit=3)
        
        print(f"Loaded {len(translation_pairs)} translation pairs and {len(revision_pairs)} revision pairs")
        
        # Create a list to store loaded model pairs
        loaded_pairs = []
        
        # Load source and target models for each pair
        for pair_type, pairs in [("translation", translation_pairs), ("revision", revision_pairs)]:
            for pair in pairs:
                print(f"Loading pair: {pair['name']}")
                source_model = loader.load_model(pair['source'])
                target_model = loader.load_model(pair['target'])
                
                # Get model text representations
                source_text = source_model.to_text()
                target_text = target_model.to_text()
                
                loaded_pairs.append({
                    'name': pair['name'],
                    'source_model': source_model,
                    'target_model': target_model,
                    'source_text': source_text,
                    'target_text': target_text,
                    'intent': pair_type
                })
        
        print(f"Successfully loaded {len(loaded_pairs)} model pairs")
    except Exception as e:
        print(f"Error loading model pairs: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Store results for each parameter combination
    results = []
    
    # Process each alpha-beta combination
    print("\nRunning parameter optimization...")
    for alpha in alpha_values:
        for beta in beta_values:
            print(f"\nTesting parameters: α={alpha:.1f}, β={beta:.1f}")
            
            # Collect results for this parameter combination
            pair_results = []
            
            # Process each pair
            for pair in loaded_pairs:
                try:
                    # Print detailed information about the current pair
                    print(f"  Processing pair: {pair['name']} ({pair['intent']})")
                    print(f"    Source model: {pair['source_model'].id} with {len(pair['source_model'].graph.nodes())} nodes")
                    print(f"    Target model: {pair['target_model'].id} with {len(pair['target_model'].graph.nodes())} nodes")
                    
                    # Generate embeddings
                    source_embedding = embedding_generator.generate_embedding(pair['source_text'])
                    target_embedding = embedding_generator.generate_embedding(pair['target_text'])
                    
                    print(f"    Generated embeddings")
                    
                    # Create rules based on models
                    rules = adapter.create_transformation_rules(
                        pair['source_model'], pair['target_model'], pair['intent'])
                    
                    print(f"    Created {len(rules)} transformation rules")
                    
                    # Compute forward validation score
                    fvs = validator.compute_forward_validation_score(pair['target_model'], rules)
                    print(f"    Forward validation score: {fvs:.4f}")
                    
                    # Compute standard backward validation score
                    bvs = validator.compute_backward_validation_score(pair['source_model'], pair['target_model'])
                    print(f"    Backward validation score: {bvs:.4f}")
                    
                    # Compute embedding similarity
                    embedding_similarity = embedding_generator.compute_similarity(
                        pair['source_text'], pair['target_text'])
                    print(f"    Embedding similarity: {embedding_similarity:.4f}")
                    
                    # Compute enhanced backward validation with current beta
                    enhanced_bvs = beta * bvs + (1 - beta) * embedding_similarity
                    print(f"    Enhanced backward validation: {enhanced_bvs:.4f}")
                    
                    # Compute transformation quality with current alpha
                    standard_tq = alpha * fvs + (1 - alpha) * bvs
                    enhanced_tq = alpha * fvs + (1 - alpha) * enhanced_bvs
                    
                    print(f"    Standard transformation quality: {standard_tq:.4f}")
                    print(f"    Enhanced transformation quality: {enhanced_tq:.4f}")
                    print(f"    Improvement: {(enhanced_tq - standard_tq)*100:.2f}%")
                    
                    # Store results for this pair
                    pair_results.append({
                        'pair_name': pair['name'],
                        'intent': pair['intent'],
                        'scores': {
                            'forward_validation': float(fvs),
                            'backward_validation': float(bvs),
                            'embedding_similarity': float(embedding_similarity),
                            'enhanced_backward_validation': float(enhanced_bvs),
                            'standard_quality': float(standard_tq),
                            'enhanced_quality': float(enhanced_tq),
                            'improvement': float(enhanced_tq - standard_tq)
                        }
                    })
                except Exception as e:
                    print(f"    Error processing pair: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate average scores for this parameter combination
            if pair_results:
                # Overall averages
                all_improvements = [r['scores']['improvement'] for r in pair_results]
                all_standard_quality = [r['scores']['standard_quality'] for r in pair_results]
                all_enhanced_quality = [r['scores']['enhanced_quality'] for r in pair_results]
                
                avg_improvement = np.mean(all_improvements)
                avg_standard_quality = np.mean(all_standard_quality)
                avg_enhanced_quality = np.mean(all_enhanced_quality)
                
                # Intent-specific averages
                translation_results = [r for r in pair_results if r['intent'] == 'translation']
                revision_results = [r for r in pair_results if r['intent'] == 'revision']
                
                trans_improvement = np.mean([r['scores']['improvement'] for r in translation_results]) if translation_results else 0
                rev_improvement = np.mean([r['scores']['improvement'] for r in revision_results]) if revision_results else 0
                
                print(f"\n  Average results for α={alpha:.1f}, β={beta:.1f}:")
                print(f"    Overall improvement: {avg_improvement*100:.2f}%")
                print(f"    Translation improvement: {trans_improvement*100:.2f}%")
                print(f"    Revision improvement: {rev_improvement*100:.2f}%")
                
                # Store results for this parameter combination
                results.append({
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'overall': {
                        'improvement': float(avg_improvement),
                        'standard_quality': float(avg_standard_quality),
                        'enhanced_quality': float(avg_enhanced_quality)
                    },
                    'translation': {
                        'improvement': float(trans_improvement)
                    },
                    'revision': {
                        'improvement': float(rev_improvement)
                    },
                    'pair_results': pair_results
                })
    
    # Save full results
    print("\nSaving results...")
    with open(os.path.join(args.output, 'parameter_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find optimal parameters
    overall_improvements = np.zeros((len(beta_values), len(alpha_values)))
    trans_improvements = np.zeros((len(beta_values), len(alpha_values)))
    rev_improvements = np.zeros((len(beta_values), len(alpha_values)))
    
    for result in results:
        alpha_idx = np.where(alpha_values == result['alpha'])[0][0]
        beta_idx = np.where(beta_values == result['beta'])[0][0]
        
        overall_improvements[beta_idx, alpha_idx] = result['overall']['improvement'] * 100  # Convert to percentage
        trans_improvements[beta_idx, alpha_idx] = result['translation']['improvement'] * 100
        rev_improvements[beta_idx, alpha_idx] = result['revision']['improvement'] * 100
    
    # Find optimal parameters
    overall_opt_idx = np.unravel_index(np.argmax(overall_improvements), overall_improvements.shape)
    trans_opt_idx = np.unravel_index(np.argmax(trans_improvements), trans_improvements.shape)
    rev_opt_idx = np.unravel_index(np.argmax(rev_improvements), rev_improvements.shape)
    
    overall_opt_alpha = alpha_values[overall_opt_idx[1]]
    overall_opt_beta = beta_values[overall_opt_idx[0]]
    trans_opt_alpha = alpha_values[trans_opt_idx[1]]
    trans_opt_beta = beta_values[trans_opt_idx[0]]
    rev_opt_alpha = alpha_values[rev_opt_idx[1]]
    rev_opt_beta = beta_values[rev_opt_idx[0]]
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Overall improvement heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(overall_improvements, cmap='viridis', 
               extent=[min(alpha_values)-0.05, max(alpha_values)+0.05, 
                        min(beta_values)-0.05, max(beta_values)+0.05])
    plt.colorbar(label='Improvement (%)')
    plt.xlabel('Alpha (α)')
    plt.ylabel('Beta (β)')
    plt.title('Overall Parameter Optimization')
    plt.xticks(alpha_values)
    plt.yticks(beta_values)
    
    # Mark optimal point
    plt.plot(overall_opt_alpha, overall_opt_beta, 'r*', markersize=15)
    plt.annotate(f'Optimal: α={overall_opt_alpha:.1f}, β={overall_opt_beta:.1f}\nImprovement: {overall_improvements[overall_opt_idx]:.2f}%',
                 xy=(overall_opt_alpha, overall_opt_beta),
                 xytext=(overall_opt_alpha-0.1, overall_opt_beta-0.1),
                 arrowprops=dict(arrowstyle="->", color='white'),
                 color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'overall_optimization.png'), dpi=300)
    plt.close()
    
    # Create separate heatmaps for translation and revision
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Translation heatmap
    im1 = ax1.imshow(trans_improvements, cmap='Blues', 
                    extent=[min(alpha_values)-0.05, max(alpha_values)+0.05, 
                            min(beta_values)-0.05, max(beta_values)+0.05])
    fig.colorbar(im1, ax=ax1, label='Improvement (%)')
    ax1.set_xlabel('Alpha (α)')
    ax1.set_ylabel('Beta (β)')
    ax1.set_title('Translation Transformation Optimization')
    ax1.set_xticks(alpha_values)
    ax1.set_yticks(beta_values)
    
    # Mark translation optimal point
    ax1.plot(trans_opt_alpha, trans_opt_beta, 'r*', markersize=15)
    ax1.annotate(f'Optimal: α={trans_opt_alpha:.1f}, β={trans_opt_beta:.1f}\nImprovement: {trans_improvements[trans_opt_idx]:.2f}%',
                 xy=(trans_opt_alpha, trans_opt_beta),
                 xytext=(trans_opt_alpha-0.1, trans_opt_beta-0.1),
                 arrowprops=dict(arrowstyle="->", color='white'),
                 color='white')
    
    # Revision heatmap
    im2 = ax2.imshow(rev_improvements, cmap='Greens', 
                    extent=[min(alpha_values)-0.05, max(alpha_values)+0.05, 
                            min(beta_values)-0.05, max(beta_values)+0.05])
    fig.colorbar(im2, ax=ax2, label='Improvement (%)')
    ax2.set_xlabel('Alpha (α)')
    ax2.set_ylabel('Beta (β)')
    ax2.set_title('Revision Transformation Optimization')
    ax2.set_xticks(alpha_values)
    ax2.set_yticks(beta_values)
    
    # Mark revision optimal point
    ax2.plot(rev_opt_alpha, rev_opt_beta, 'r*', markersize=15)
    ax2.annotate(f'Optimal: α={rev_opt_alpha:.1f}, β={rev_opt_beta:.1f}\nImprovement: {rev_improvements[rev_opt_idx]:.2f}%',
                 xy=(rev_opt_alpha, rev_opt_beta),
                 xytext=(rev_opt_alpha-0.1, rev_opt_beta-0.1),
                 arrowprops=dict(arrowstyle="->", color='white'),
                 color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'intent_specific_optimization.png'), dpi=300)
    plt.close()
    
    # Save optimal parameters
    optimal_params = {
        'overall': {
            'alpha': float(overall_opt_alpha),
            'beta': float(overall_opt_beta),
            'improvement': float(overall_improvements[overall_opt_idx])
        },
        'translation': {
            'alpha': float(trans_opt_alpha),
            'beta': float(trans_opt_beta),
            'improvement': float(trans_improvements[trans_opt_idx])
        },
        'revision': {
            'alpha': float(rev_opt_alpha),
            'beta': float(rev_opt_beta),
            'improvement': float(rev_improvements[rev_opt_idx])
        }
    }
    
    with open(os.path.join(args.output, 'optimal_parameters.json'), 'w') as f:
        json.dump(optimal_params, f, indent=2)
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("Parameter Optimization Results")
    print("=" * 80)
    
    print("\nOverall Recommended Parameters:")
    print(f"  α = {overall_opt_alpha:.1f}, β = {overall_opt_beta:.1f}")
    print(f"  Average improvement: {overall_improvements[overall_opt_idx]:.2f}%")
    
    print("\nTranslation Transformation Recommended Parameters:")
    print(f"  α = {trans_opt_alpha:.1f}, β = {trans_opt_beta:.1f}")
    print(f"  Average improvement: {trans_improvements[trans_opt_idx]:.2f}%")
    
    print("\nRevision Transformation Recommended Parameters:")
    print(f"  α = {rev_opt_alpha:.1f}, β = {rev_opt_beta:.1f}")
    print(f"  Average improvement: {rev_improvements[rev_opt_idx]:.2f}%")
    
    print("\n" + "=" * 80)
    print(f"Optimization complete! Results saved in {args.output}")
    print("=" * 80)

if __name__ == "__main__":
    main()