#!/usr/bin/env python3
"""
Statistical Analysis for Neural Embeddings in Model Transformation Validation

This script performs statistical analysis on experiment results to verify:
1. The differential impact of embedding enhancement on different transformation types
2. The statistical significance of improvements/degradations
3. The synergistic effects of combined approaches
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_experiment_data(results_file):
    """
    Load and parse the experimental results from JSON file
    
    Args:
        results_file: Path to the JSON results file
        
    Returns:
        Pandas DataFrame containing the experimental results
    """
    print(f"Loading experiment results from {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Extract nested values for easier access
    df['regular_quality'] = df['regular'].apply(lambda x: x['transformation_quality'])
    df['enhanced_quality'] = df['enhanced'].apply(lambda x: x['transformation_quality'])
    
    # Calculate raw and percentage differences
    df['diff'] = df['enhanced_quality'] - df['regular_quality']
    df['diff_percent'] = (df['diff'] / df['regular_quality']) * 100
    
    print(f"Loaded {len(df)} experiment results")
    return df

def analyze_by_transformation_type(df, output_dir=None):
    """
    Analyze results separated by transformation type (translation vs. revision)
    
    Args:
        df: DataFrame containing experimental results
        output_dir: Optional directory to save output files
        
    Returns:
        Dictionary containing statistical results
    """
    print("\nAnalyzing results by transformation type...")
    
    # Group by transformation type
    translation_df = df[df['pair_type'] == 'translation']
    revision_df = df[df['pair_type'] == 'revision']
    
    print(f"Translation transformations: {len(translation_df)}")
    print(f"Revision transformations: {len(revision_df)}")
    
    results = {}
    
    # Analyze translation transformations
    if len(translation_df) > 1:
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(
            translation_df['enhanced_quality'], 
            translation_df['regular_quality']
        )
        
        # Calculate mean difference
        mean_diff = translation_df['diff'].mean()
        mean_diff_percent = translation_df['diff_percent'].mean()
        
        # Store results
        results['translation'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'mean_difference': float(mean_diff),
            'mean_difference_percent': float(mean_diff_percent),
            'significant': p_val < 0.05,
            'sample_size': len(translation_df),
            'std_dev': float(translation_df['diff_percent'].std())
        }
        
        print(f"\nTranslation Transformations:")
        print(f"  Mean improvement: {mean_diff_percent:.2f}%")
        print(f"  t-statistic: {t_stat:.2f}")
        print(f"  p-value: {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
    
    # Analyze revision transformations
    if len(revision_df) > 1:
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(
            revision_df['enhanced_quality'], 
            revision_df['regular_quality']
        )
        
        # Calculate mean difference
        mean_diff = revision_df['diff'].mean()
        mean_diff_percent = revision_df['diff_percent'].mean()
        
        # Store results
        results['revision'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'mean_difference': float(mean_diff),
            'mean_difference_percent': float(mean_diff_percent),
            'significant': p_val < 0.05,
            'sample_size': len(revision_df),
            'std_dev': float(revision_df['diff_percent'].std())
        }
        
        print(f"\nRevision Transformations:")
        print(f"  Mean improvement: {mean_diff_percent:.2f}%")
        print(f"  t-statistic: {t_stat:.2f}")
        print(f"  p-value: {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
    
    # Generate visualizations if output directory is provided
    if output_dir:
        create_statistical_visualizations(translation_df, revision_df, results, output_dir)
    
    return results

def analyze_autoregression_results(autoregression_file, output_dir=None):
    """
    Analyze autoregression experiment results
    
    Args:
        autoregression_file: Path to autoregression results file
        output_dir: Optional directory to save output files
        
    Returns:
        Dictionary containing statistical results
    """
    print(f"\nAnalyzing auto-regression results from {autoregression_file}")
    
    with open(autoregression_file, 'r') as f:
        data = json.load(f)
    
    # Extract configuration results
    baseline = next(r for r in data if r['config'] == 'Baseline')
    autoregression = next(r for r in data if r['config'] == 'Auto-Regression Only')
    embeddings = next(r for r in data if r['config'] == 'Embeddings Only')
    combined = next(r for r in data if r['config'] == 'Combined (Optimized)')
    
    # Calculate improvements
    ar_improvement = ((autoregression['quality'] - baseline['quality']) / baseline['quality']) * 100
    emb_improvement = ((embeddings['quality'] - baseline['quality']) / baseline['quality']) * 100
    combined_improvement = ((combined['quality'] - baseline['quality']) / baseline['quality']) * 100
    
    # Calculate synergy
    expected_combined = ar_improvement + emb_improvement
    synergy = combined_improvement - expected_combined
    
    results = {
        'auto_regression_improvement': float(ar_improvement),
        'embeddings_improvement': float(emb_improvement),
        'combined_improvement': float(combined_improvement),
        'expected_combined': float(expected_combined),
        'synergy': float(synergy)
    }
    
    print(f"Auto-Regression Improvement: {ar_improvement:.2f}%")
    print(f"Embeddings Improvement: {emb_improvement:.2f}%")
    print(f"Combined Improvement: {combined_improvement:.2f}%")
    print(f"Expected Combined (Sum): {expected_combined:.2f}%")
    print(f"Synergy: {synergy:.2f}%")
    
    # Generate visualization if output directory is provided
    if output_dir:
        create_autoregression_visualization(results, output_dir)
    
    return results

def create_statistical_visualizations(translation_df, revision_df, results, output_dir):
    """
    Create statistical visualizations for the analysis
    
    Args:
        translation_df: DataFrame containing translation transformation results
        revision_df: DataFrame containing revision transformation results
        results: Dictionary containing statistical results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create box plots of improvements by transformation type
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    translation_diffs = translation_df['diff_percent'] if not translation_df.empty else []
    revision_diffs = revision_df['diff_percent'] if not revision_df.empty else []
    
    data = [translation_diffs, revision_diffs]
    labels = ['Translation', 'Revision']
    
    # Create box plot
    box = plt.boxplot(data, patch_artist=True, labels=labels)
    
    # Set colors
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add statistical annotations
    if 'translation' in results:
        plt.text(1, min(translation_diffs) - 0.2, 
                f"t = {results['translation']['t_statistic']:.2f}\np = {results['translation']['p_value']:.3f}",
                ha='center')
    
    if 'revision' in results:
        plt.text(2, max(revision_diffs) + 0.2, 
                f"t = {results['revision']['t_statistic']:.2f}\np = {results['revision']['p_value']:.3f}",
                ha='center')
    
    plt.ylabel('Improvement (%)')
    plt.title('Embedding Enhancement Effect by Transformation Type')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transformation_type_boxplot.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'transformation_type_boxplot.pdf'))
    plt.close()
    
    # Create bar chart of mean improvements
    plt.figure(figsize=(8, 6))
    
    means = [
        results.get('translation', {}).get('mean_difference_percent', 0),
        results.get('revision', {}).get('mean_difference_percent', 0)
    ]
    
    # Create bars with different colors based on significance
    bars = plt.bar(labels, means, color=['lightblue', 'lightgreen'])
    
    # Add significance markers
    for i, (label, result_key) in enumerate(zip(labels, ['translation', 'revision'])):
        if result_key in results and results[result_key]['significant']:
            plt.text(i, means[i] + 0.05, '*', ha='center', fontsize=20)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.ylabel('Mean Improvement (%)')
    plt.title('Average Effect of Embedding Enhancement')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_improvements.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'mean_improvements.pdf'))
    plt.close()

def create_autoregression_visualization(results, output_dir):
    """
    Create visualization for auto-regression analysis
    
    Args:
        results: Dictionary containing auto-regression analysis results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create bar chart of improvements
    plt.figure(figsize=(10, 6))
    
    # Data
    labels = ['Auto-Regression', 'Embeddings', 'Combined', 'Expected Combined']
    values = [
        results['auto_regression_improvement'],
        results['embeddings_improvement'],
        results['combined_improvement'],
        results['expected_combined']
    ]
    
    # Create bars
    bars = plt.bar(labels, values, color=['lightblue', 'lightgreen', 'purple', 'lightgray'])
    
    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom')
    
    # Add synergy annotation
    plt.annotate(f"Synergy: {results['synergy']:.2f}%",
                xy=(2, values[2]),
                xytext=(3, values[2] + 1),
                arrowprops=dict(arrowstyle="->", color='red'))
    
    plt.ylabel('Improvement over Baseline (%)')
    plt.title('Synergistic Effects of Combined Approaches')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'autoregression_synergy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'autoregression_synergy.pdf'))
    plt.close()

def convert_numpy_types(obj):
    """Convert NumPy types to standard Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def main():
    """Main function to run statistical analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform statistical analysis on experiment results')
    
    parser.add_argument('--results', type=str, required=True,
                        help='Path to the detailed comparison results JSON file')
    
    parser.add_argument('--autoregression', type=str, default=None,
                        help='Path to the autoregression results JSON file (optional)')
    
    parser.add_argument('--output', type=str, default='statistical_analysis',
                        help='Directory to save analysis results and visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and analyze experimental data
    df = load_experiment_data(args.results)
    type_results = analyze_by_transformation_type(df, args.output)
    
    # Analyze autoregression results if provided
    ar_results = None
    if args.autoregression and os.path.exists(args.autoregression):
        ar_results = analyze_autoregression_results(args.autoregression, args.output)
    
    # Combine all results
    all_results = {
        'transformation_type_analysis': type_results,
        'autoregression_analysis': ar_results,
        'experiment_summary': {
            'total_pairs': len(df),
            'translation_pairs': len(df[df['pair_type'] == 'translation']),
            'revision_pairs': len(df[df['pair_type'] == 'revision']),
            'overall_mean_improvement': float(df['diff_percent'].mean()),
            'overall_significant': stats.ttest_rel(df['enhanced_quality'], df['regular_quality']).pvalue < 0.05
        }
    }
    
        # Before JSON serialization, convert NumPy types
    all_results = convert_numpy_types(all_results)

    # Save combined results
    with open(os.path.join(args.output, 'statistical_analysis_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nStatistical analysis complete. Results saved to {args.output}")
    
    # Print final summary
    print("\nFinal Statistical Summary:")
    print(f"  Total model pairs analyzed: {len(df)}")
    print(f"  Translation transformations: {len(df[df['pair_type'] == 'translation'])}")
    print(f"  Revision transformations: {len(df[df['pair_type'] == 'revision'])}")
    
    if 'translation' in type_results:
        print(f"\n  Translation effect: {type_results['translation']['mean_difference_percent']:.2f}%")
        print(f"    t = {type_results['translation']['t_statistic']:.2f}, p = {type_results['translation']['p_value']:.4f}")
        print(f"    {'Statistically significant' if type_results['translation']['significant'] else 'Not significant'}")
    
    if 'revision' in type_results:
        print(f"\n  Revision effect: {type_results['revision']['mean_difference_percent']:.2f}%")
        print(f"    t = {type_results['revision']['t_statistic']:.2f}, p = {type_results['revision']['p_value']:.4f}")
        print(f"    {'Statistically significant' if type_results['revision']['significant'] else 'Not significant'}")
    
    if ar_results:
        print(f"\n  Auto-regression improvement: {ar_results['auto_regression_improvement']:.2f}%")
        print(f"  Embeddings improvement: {ar_results['embeddings_improvement']:.2f}%")
        print(f"  Combined improvement: {ar_results['combined_improvement']:.2f}%")
        print(f"  Synergistic effect: {ar_results['synergy']:.2f}%")

if __name__ == "__main__":
    main()