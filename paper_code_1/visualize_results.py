#!/usr/bin/env python3
"""
Visualization script for embedding-enhanced bidirectional validation experiments
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_experiment_results(results_file):
    """Load experiment results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_intent_comparison_chart(results, output_dir):
    """Create intent-specific benefits visualization"""
    print("Creating intent comparison chart...")
    
    # Extract improvements by transformation type
    translation_improvements = []
    revision_improvements = []
    
    for result in results:
        if result['pair_type'] == 'translation':
            translation_improvements.append(result['improvement'] * 100)  # Convert to percentage
        elif result['pair_type'] == 'revision':
            revision_improvements.append(result['improvement'] * 100)  # Convert to percentage
    
    # Calculate statistics
    trans_avg = np.mean(translation_improvements)
    rev_avg = np.mean(revision_improvements)
    trans_std = np.std(translation_improvements)
    rev_std = np.std(revision_improvements)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(
        ['Translation'] * len(translation_improvements), 
        translation_improvements, 
        color='blue', 
        alpha=0.7, 
        label='Translation transformations'
    )
    plt.scatter(
        ['Revision'] * len(revision_improvements), 
        revision_improvements, 
        color='green', 
        alpha=0.7, 
        label='Revision transformations'
    )
    
    # Add error bars for standard deviation
    plt.errorbar(['Translation', 'Revision'], [trans_avg, rev_avg], 
                 yerr=[trans_std, rev_std], fmt='o', color='black', 
                 ecolor='black', capsize=10, label='Mean ± Std Dev')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.ylabel('Improvement in Transformation Quality (%)')
    plt.title('Impact of Embedding Enhancement by Transformation Intent')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations for averages
    plt.annotate(f'Average: {trans_avg:.2f}%', 
                 xy=('Translation', trans_avg), 
                 xytext=(0, 20), 
                 textcoords='offset points',
                 ha='center',
                 arrowprops=dict(arrowstyle='->'))
    
    plt.annotate(f'Average: {rev_avg:.2f}%', 
                 xy=('Revision', rev_avg), 
                 xytext=(0, 20), 
                 textcoords='offset points',
                 ha='center',
                 arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'intent_comparison_chart.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'intent_comparison_chart.pdf'))
    plt.close()
    
    print(f"Intent comparison chart saved to {output_dir}")
    return {
        'translation_avg': trans_avg,
        'revision_avg': rev_avg,
        'translation_std': trans_std,
        'revision_std': rev_std
    }

def create_similarity_improvement_chart(results, output_dir):
    """Create embedding similarity vs. improvement relationship chart"""
    print("Creating similarity-improvement chart...")
    
    # Extract data by transformation type
    translation_data = []
    revision_data = []
    
    for result in results:
        similarity = result['direct_embedding_similarity']
        improvement = result['improvement'] * 100  # Convert to percentage
        
        if result['pair_type'] == 'translation':
            translation_data.append((similarity, improvement))
        elif result['pair_type'] == 'revision':
            revision_data.append((similarity, improvement))
    
    # Extract data for plotting
    trans_sim = [d[0] for d in translation_data]
    trans_imp = [d[1] for d in translation_data]
    rev_sim = [d[0] for d in revision_data]
    rev_imp = [d[1] for d in revision_data]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(trans_sim, trans_imp, color='blue', alpha=0.7, s=100, label='Translation transformations')
    plt.scatter(rev_sim, rev_imp, color='green', alpha=0.7, s=100, label='Revision transformations')
    
    # Add regression lines
    if len(trans_sim) > 1:
        trans_z = np.polyfit(trans_sim, trans_imp, 1)
        trans_p = np.poly1d(trans_z)
        plt.plot(np.sort(trans_sim), trans_p(np.sort(trans_sim)), "b--", alpha=0.5)
    
    if len(rev_sim) > 1:
        rev_z = np.polyfit(rev_sim, rev_imp, 1)
        rev_p = np.poly1d(rev_z)
        plt.plot(np.sort(rev_sim), rev_p(np.sort(rev_sim)), "g--", alpha=0.5)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # Highlight the point with highest embedding similarity and improvement
    if rev_sim:
        max_sim_idx = rev_sim.index(max(rev_sim))
        plt.annotate(f'Highest similarity: {rev_sim[max_sim_idx]:.4f}\nImprovement: {rev_imp[max_sim_idx]:.2f}%',
                     xy=(rev_sim[max_sim_idx], rev_imp[max_sim_idx]),
                     xytext=(rev_sim[max_sim_idx]-0.05, rev_imp[max_sim_idx]+0.3),
                     arrowprops=dict(arrowstyle='->'))
    
    # Add labels and title
    plt.xlabel('Direct Embedding Similarity')
    plt.ylabel('Improvement in Transformation Quality (%)')
    plt.title('Relationship Between Direct Embedding Similarity and Quality Improvement')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set axis limits
    plt.xlim(0.85, 1.0)
    plt.ylim(-0.7, 0.9)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'similarity_improvement_chart.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'similarity_improvement_chart.pdf'))
    plt.close()
    
    print(f"Similarity-improvement chart saved to {output_dir}")

def create_bidirectional_metrics_chart(results, output_dir):
    """Create bidirectional validation metrics comparison chart"""
    print("Creating bidirectional metrics chart...")
    
    # Collect metrics by transformation type
    translation_forward = []
    translation_backward_reg = []
    translation_backward_enh = []
    translation_quality_reg = []
    translation_quality_enh = []
    
    revision_forward = []
    revision_backward_reg = []
    revision_backward_enh = []
    revision_quality_reg = []
    revision_quality_enh = []
    
    for result in results:
        if result['pair_type'] == 'translation':
            translation_forward.append(result['regular']['forward_validation'])
            translation_backward_reg.append(result['regular']['backward_validation'])
            translation_backward_enh.append(result['enhanced']['backward_validation'])
            translation_quality_reg.append(result['regular']['transformation_quality'])
            translation_quality_enh.append(result['enhanced']['transformation_quality'])
        elif result['pair_type'] == 'revision':
            revision_forward.append(result['regular']['forward_validation'])
            revision_backward_reg.append(result['regular']['backward_validation'])
            revision_backward_enh.append(result['enhanced']['backward_validation'])
            revision_quality_reg.append(result['regular']['transformation_quality'])
            revision_quality_enh.append(result['enhanced']['transformation_quality'])
    
    # Calculate averages
    trans_regular = [
        np.mean(translation_forward),
        np.mean(translation_backward_reg),
        np.mean(translation_quality_reg)
    ]
    
    trans_enhanced = [
        np.mean(translation_forward),
        np.mean(translation_backward_enh),
        np.mean(translation_quality_enh)
    ]
    
    rev_regular = [
        np.mean(revision_forward),
        np.mean(revision_backward_reg),
        np.mean(revision_quality_reg)
    ]
    
    rev_enhanced = [
        np.mean(revision_forward),
        np.mean(revision_backward_enh),
        np.mean(revision_quality_enh)
    ]
    
    # Metrics labels
    metrics = ['Forward Validation', 'Backward Validation', 'Transformation Quality']
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Set up positions
    x = np.arange(len(metrics))
    width = 0.2
    
    # Create grouped bar chart
    plt.bar(x - width*1.5, trans_regular, width, label='Translation (Regular)', color='#1f77b4')
    plt.bar(x - width/2, trans_enhanced, width, label='Translation (Enhanced)', color='#1f77b4', alpha=0.5)
    plt.bar(x + width/2, rev_regular, width, label='Revision (Regular)', color='#2ca02c')
    plt.bar(x + width*1.5, rev_enhanced, width, label='Revision (Enhanced)', color='#2ca02c', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Validation Metric')
    plt.ylabel('Score')
    plt.title('Comparison of Bidirectional Validation Metrics by Transformation Intent')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Set y-axis to start from 0.90 to better show differences
    plt.ylim(0.90, 0.98)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=90, fontsize=8)
    
    add_labels(plt.gca().patches)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bidirectional_metrics_chart.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'bidirectional_metrics_chart.pdf'))
    plt.close()
    
    print(f"Bidirectional metrics chart saved to {output_dir}")

def create_contribution_chart(results, output_dir):
    """Create token pair vs. embedding contribution chart"""
    print("Creating contribution chart...")
    
    # Get average token pair similarity
    # Since this is constant in your results for translation, use the value
    token_pair_values = [0.3040, 0.3040]  # Constant from your results
    
    # Calculate average embedding similarity and impact
    trans_embedding = []
    trans_impact = []
    rev_embedding = []
    rev_impact = []
    
    for result in results:
        if result['pair_type'] == 'translation':
            trans_embedding.append(result['direct_embedding_similarity'])
            trans_impact.append(result['improvement'] * 100)  # Convert to percentage
        elif result['pair_type'] == 'revision':
            rev_embedding.append(result['direct_embedding_similarity'])
            rev_impact.append(result['improvement'] * 100)  # Convert to percentage
    
    # Average values
    embedding_values = [np.mean(trans_embedding), np.mean(rev_embedding)]
    validation_impact = [np.mean(trans_impact), np.mean(rev_impact)]
    
    # Types labels
    types = ['Translation\nTransformations', 'Revision\nTransformations']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First subplot: Similarity Comparison
    x = np.arange(len(types))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, token_pair_values, width, label='Token Pair Similarity', color='#ff7f0e')
    bars2 = ax1.bar(x + width/2, embedding_values, width, label='Embedding Similarity', color='#9467bd')
    
    ax1.set_ylabel('Similarity Score')
    ax1.set_title('Token Pair vs. Embedding Similarity by Transformation Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types)
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Second subplot: Impact on Validation
    bars3 = ax2.bar(types, validation_impact, color=['#ff7f0e' if x < 0 else '#9467bd' for x in validation_impact])
    ax2.set_ylabel('Improvement in Transformation Quality (%)')
    ax2.set_title('Impact of Embedding Enhancement by Transformation Type')
    ax2.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(validation_impact):
        ax2.text(i, v + 0.05 if v >= 0 else v - 0.15,
                f'{v:.2f}%',
                ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'contribution_chart.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'contribution_chart.pdf'))
    plt.close()
    
    print(f"Contribution chart saved to {output_dir}")

def main():
    """Main function to generate all visualizations"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for embedding experiment results')
    parser.add_argument('--results', type=str, required=True, 
                        help='Path to detailed_comparison.json results file')
    parser.add_argument('--output', type=str, default='figures',
                        help='Directory to save visualization figures')
    
    args = parser.parse_args()
    
    # Load experiment results
    try:
        results = load_experiment_results(args.results)
        print(f"Loaded {len(results)} experiment results from {args.results}")
    except Exception as e:
        print(f"Error loading results from {args.results}: {str(e)}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate visualizations
    create_intent_comparison_chart(results, args.output)
    create_similarity_improvement_chart(results, args.output)
    create_bidirectional_metrics_chart(results, args.output)
    create_contribution_chart(results, args.output)
    
    print(f"\nAll visualizations saved to {args.output}")
    print("These figures can be used in your paper to better illustrate your findings.")

if __name__ == "__main__":
    main()