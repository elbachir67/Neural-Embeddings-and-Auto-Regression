#!/usr/bin/env python3
"""
Run experiments with optimized parameters from parameter optimization
"""

import os
import sys
import argparse
import logging
import datetime
import json

# Configure logging
log_filename = f"optimized_experiments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also output to console
    ]
)
logging.info("Starting optimized experiments script")

# Import your custom modules
try:
    from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer
    from modelset_loader import ModelSetLoader
    from token_pair_adapter import TokenPairAdapter
    from embedding_generator import EmbeddingGenerator
    logging.info("Successfully imported custom modules")
except Exception as e:
    logging.error(f"Error importing custom modules: {str(e)}", exc_info=True)
    sys.exit(1)

def run_optimized_comparison(modelset_path, output_dir, limit=5):
    """Run a detailed comparison between regular and embedding-enhanced approaches with optimized parameters"""
    logging.info("\nRunning Detailed Comparison Experiment with Optimized Parameters...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize components
        logging.info("Initializing components with optimized parameters...")
        loader = ModelSetLoader(modelset_path)
        adapter = TokenPairAdapter()
        encoder = ContextEncoder()
        validator = BidirectionalValidator(encoder)
        embedding_generator = EmbeddingGenerator()
        logging.info("Components initialized")
    except Exception as e:
        logging.error(f"Error initializing components: {str(e)}", exc_info=True)
        return
    
    try:
        # Get pairs of both types
        logging.info("Getting transformation pairs...")
        translation_pairs = loader.get_transformation_pairs("translation", limit=limit)
        revision_pairs = loader.get_transformation_pairs("revision", limit=limit)
        
        all_pairs = translation_pairs + revision_pairs
        
        if not all_pairs:
            logging.warning("No suitable transformation pairs found")
            return
        
        logging.info(f"Found {len(all_pairs)} transformation pairs ({len(translation_pairs)} translation, {len(revision_pairs)} revision)")
    except Exception as e:
        logging.error(f"Error getting transformation pairs: {str(e)}", exc_info=True)
        return
    
    results = []
    
    # Process each pair with optimized parameters
    for pair_index, pair in enumerate(all_pairs):
        logging.info(f"\nProcessing pair {pair_index+1}: {pair['name']} ({pair['type']})")
        
        try:
            # Load source and target models with text
            source_model, source_text = loader.load_model_with_text(pair['source'])
            target_model, target_text = loader.load_model_with_text(pair['target'])
            
            if not source_text or not target_text:
                logging.warning("  Skipping pair due to missing text representations")
                continue
                
            logging.info(f"  Loaded source model: {source_model.id} ({source_model.type})")
            logging.info(f"  Loaded target model: {target_model.id} ({target_model.type})")
            
            # Generate embeddings
            logging.info("  Generating embeddings for models...")
            source_embedding = embedding_generator.generate_embedding(source_text)
            target_embedding = embedding_generator.generate_embedding(target_text)
            
            # Direct embedding similarity
            embedding_similarity = embedding_generator.compute_similarity(source_text, target_text)
            logging.info(f"  Direct embedding similarity: {embedding_similarity:.4f}")
            
            # Create transformation rules
            rules = adapter.create_transformation_rules(source_model, target_model, pair['type'])
            
            # Compute forward validation score
            fvs = validator.compute_forward_validation_score(target_model, rules)
            
            # Compute standard backward validation score
            bvs = validator.compute_backward_validation_score(source_model, target_model)
            
            # Use optimized beta value (0.7)
            beta = 0.7
            enhanced_bvs = beta * bvs + (1 - beta) * embedding_similarity
            
            # Use intent-specific alpha values
            if pair['type'] == "translation":
                alpha = 0.5  # Optimized value for translation
            else:  # revision
                alpha = 0.7  # Optimized value for revision
            
            # Compute transformation quality scores
            standard_tq = alpha * fvs + (1 - alpha) * bvs
            enhanced_tq = alpha * fvs + (1 - alpha) * enhanced_bvs
            improvement = enhanced_tq - standard_tq
            
            logging.info(f"  Intent: {pair['type'].upper()}")
            logging.info(f"  Parameters: alpha={alpha:.1f}, beta={beta:.1f}")
            logging.info(f"  Forward Validation: {fvs:.4f}")
            logging.info(f"  Standard Backward Validation: {bvs:.4f}")
            logging.info(f"  Enhanced Backward Validation: {enhanced_bvs:.4f}")
            logging.info(f"  Standard Quality: {standard_tq:.4f}")
            logging.info(f"  Enhanced Quality: {enhanced_tq:.4f}")
            logging.info(f"  Improvement: {improvement*100:.2f}%")
            
            # Store result with pair info
            pair_result = {
                'pair_name': pair['name'],
                'pair_type': pair['type'],
                'source_id': pair['source']['id'],
                'target_id': pair['target']['id'],
                'direct_embedding_similarity': float(embedding_similarity),
                'alpha': float(alpha),
                'beta': float(beta),
                'regular': {
                    'forward_validation': float(fvs),
                    'backward_validation': float(bvs),
                    'transformation_quality': float(standard_tq)
                },
                'enhanced': {
                    'forward_validation': float(fvs),
                    'backward_validation': float(enhanced_bvs),
                    'transformation_quality': float(enhanced_tq)
                },
                'improvement': float(improvement)
            }
            
            results.append(pair_result)
            
        except Exception as e:
            logging.error(f"  Error processing pair: {str(e)}", exc_info=True)
            continue
    
    # Save results
    try:
        if results:
            # Save JSON results
            with open(os.path.join(output_dir, 'detailed_comparison_optimized.json'), 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {os.path.join(output_dir, 'detailed_comparison_optimized.json')}")
        else:
            logging.warning("No results to save or visualize")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}", exc_info=True)
    
    return results

def main():
    """Main function to run experiments with optimized parameters"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run experiments with optimized parameters')
    
    parser.add_argument('--modelset', type=str, required=True,
                        help='Path to the ModelSet dataset')
    
    parser.add_argument('--output', type=str, default='results_optimized',
                        help='Directory to save experiment results')
    
    parser.add_argument('--pairs', type=int, default=10,
                        help='Number of transformation pairs to process')
    
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("Running Experiments with Optimized Parameters")
    logging.info("=" * 80)
    logging.info(f"ModelSet path: {args.modelset}")
    logging.info(f"Output directory: {args.output}")
    logging.info(f"Number of pairs: {args.pairs}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run intent-aware comparison with optimized parameters
    logging.info("\nRunning intent-aware comparison with optimized parameters...")
    results = run_optimized_comparison(args.modelset, args.output, limit=args.pairs)
    
    logging.info("\n" + "=" * 80)
    logging.info(f"Experiments complete! Results saved in {args.output}")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()