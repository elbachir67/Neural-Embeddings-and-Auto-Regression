#!/usr/bin/env python3
import os
import sys
import argparse
import json
import traceback

# Add debug prints at the very beginning
print("=== DEBUG: Script starting ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    print("Attempting to import packages...")
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    print("Successfully imported base packages")
    
    print("Attempting to import custom modules...")
    from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer, TransformationRule
    from modelset_loader import ModelSetLoader
    from token_pair_adapter import TokenPairAdapter
    from embedding_generator import EmbeddingGenerator
    print("Successfully imported custom modules")
except Exception as e:
    print(f"ERROR during imports: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run embedding-enhanced experiments')
    
    parser.add_argument('--modelset', type=str, required=True,
                        help='Path to the ModelSet dataset')
    
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save experiment results')
    
    parser.add_argument('--experiment', type=str, choices=['basic', 'compare', 'all'],
                        default='all', help='Which experiment to run')
    
    parser.add_argument('--domain', type=str, default=None,
                        help='Filter by domain (e.g., statemachine, class)')
                        
    parser.add_argument('--limit', type=int, default=5,
                        help='Limit number of transformation pairs to process')
    
    return parser.parse_args()

def run_basic_experiment_with_embeddings(modelset_path, output_dir, domain=None, limit=5):
    """Run basic experiment with embedding enhancement"""
    print("\n" + "="*80)
    print("Running Embedding-Enhanced Basic Experiment...")
    print("="*80 + "\n")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if modelset path exists
    modelset_path = Path(modelset_path)
    if not modelset_path.exists():
        print(f"ERROR: ModelSet path does not exist: {modelset_path}")
        print(f"Absolute path: {modelset_path.absolute()}")
        return

    print(f"ModelSet path exists: {modelset_path}")
    print(f"Contents of ModelSet directory: {os.listdir(modelset_path)}")
    
    try:
        print("Initializing components...")
        # Initialize components
        loader = ModelSetLoader(modelset_path)
        print("ModelSetLoader initialized")
        
        adapter = TokenPairAdapter()
        print("TokenPairAdapter initialized")
        
        encoder = ContextEncoder()
        print("ContextEncoder initialized")
        
        validator = BidirectionalValidator(encoder)
        print("BidirectionalValidator initialized")
        
        transformer = IntentAwareTransformer(encoder, validator)
        print("IntentAwareTransformer initialized")
        
        embedding_generator = EmbeddingGenerator()
        print("EmbeddingGenerator initialized")
        
        # Get transformation pairs
        print("Getting transformation pairs...")
        pairs = loader.get_transformation_pairs("translation", limit=limit)
        
        if not pairs:
            print("No suitable transformation pairs found")
            return
        
        print(f"Found {len(pairs)} transformation pairs")
        
        # Rest of your function...
        # (We'll just return early for debugging)
        print("Basic experiment initialization complete.")
        
    except Exception as e:
        print(f"ERROR in run_basic_experiment_with_embeddings: {str(e)}")
        traceback.print_exc()
        return

# The rest of your functions (run_comparison_experiment, visualize_embedding_results, etc.)
# ...

def main():
    """Main function to run experiments"""
    try:
        print("Parsing arguments...")
        args = parse_args()
        
        print("=" * 80)
        print("Embedding-Enhanced Token Pair Bidirectional Validation")
        print("=" * 80)
        print(f"Arguments: {args}")
        
        # Create output directory
        print(f"Creating output directory: {args.output}")
        os.makedirs(args.output, exist_ok=True)
        
        # Create figures directory
        figures_dir = os.path.join(args.output, 'figures')
        print(f"Creating figures directory: {figures_dir}")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Run experiments
        if args.experiment in ['basic', 'all']:
            print(f"Running basic experiment with modelset={args.modelset}")
            run_basic_experiment_with_embeddings(args.modelset, args.output, args.domain, args.limit)
        
        # For debugging, we'll skip other experiments
        # if args.experiment in ['compare', 'all']:
        #     run_comparison_experiment(args.modelset, args.output, args.domain, args.limit)
        
        print("\n" + "=" * 80)
        print("Experiments complete! Results saved in", args.output)
        print("=" * 80)
        print(f"Figures saved in {figures_dir}")
        print("=" * 80)
    
    except Exception as e:
        print(f"ERROR in main function: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Script called from command line")
    main()
    print("Script execution completed")