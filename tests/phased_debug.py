import os
import sys
import argparse
import json
import logging
import datetime
import traceback
import time

# Configure detailed logging
log_filename = f"embedding_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also output to console
    ]
)

# Log system info
logging.info(f"Python version: {sys.version}")
logging.info(f"Current working directory: {os.getcwd()}")
logging.info("Starting embedding experiments script with phased debugging")

# Phase 1: Import packages with timing
logging.info("=== PHASE 1: Importing packages ===")
phase_start = time.time()

try:
    logging.debug("Importing base packages...")
    import matplotlib
    logging.debug(f"Matplotlib version: {matplotlib.__version__}")
    matplotlib.use('Agg')  # Use non-interactive backend
    logging.debug("Set matplotlib to non-interactive backend")
    
    import matplotlib.pyplot as plt
    logging.debug("Imported matplotlib.pyplot")
    
    import numpy as np
    logging.debug("Imported numpy")
    
    from pathlib import Path
    logging.debug("Imported pathlib")
    
    logging.debug("Importing torch...")
    import torch
    logging.debug(f"Imported torch (version: {torch.__version__})")
    
    logging.debug("Importing sklearn components...")
    from sklearn.metrics.pairwise import cosine_similarity
    logging.debug("Imported sklearn components")
    
    logging.info(f"Base packages imported successfully in {time.time() - phase_start:.2f} seconds")
    
    # Phase 2: Import custom modules with timing
    logging.info("=== PHASE 2: Importing custom modules ===")
    phase_start = time.time()
    
    logging.debug("Importing bidirectional_validator...")
    from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer, TransformationRule
    logging.debug("Imported bidirectional_validator")
    
    logging.debug("Importing modelset_loader...")
    from modelset_loader import ModelSetLoader
    logging.debug("Imported modelset_loader")
    
    logging.debug("Importing token_pair_adapter...")
    from token_pair_adapter import TokenPairAdapter
    logging.debug("Imported token_pair_adapter")
    
    logging.debug("Importing embedding_generator...")
    from embedding_generator import EmbeddingGenerator
    logging.debug("Imported embedding_generator")
    
    logging.info(f"Custom modules imported successfully in {time.time() - phase_start:.2f} seconds")
except Exception as e:
    logging.error(f"Error during imports: {str(e)}", exc_info=True)
    sys.exit(1)

def parse_args():
    """Parse command line arguments"""
    logging.debug("Parsing command line arguments")
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
    
    args = parser.parse_args()
    logging.debug(f"Parsed arguments: {vars(args)}")
    return args

def run_basic_experiment_with_embeddings(modelset_path, output_dir, domain=None, limit=5):
    """Run basic experiment with embedding enhancement - with phase debugging"""
    logging.info("=== PHASE 3: Starting basic experiment ===")
    phase_start = time.time()
    
    # Make sure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.debug(f"Created output directory: {output_dir}")
    except Exception as e:
        logging.error(f"Error creating output directory: {str(e)}", exc_info=True)
    
    # Phase 3.1: Initialize components
    logging.info("=== PHASE 3.1: Initializing components ===")
    component_start = time.time()
    
    try:
        logging.debug("Initializing ModelSetLoader...")
        loader = ModelSetLoader(modelset_path)
        logging.debug("ModelSetLoader initialized")
        
        logging.debug("Initializing TokenPairAdapter...")
        adapter = TokenPairAdapter()
        logging.debug("TokenPairAdapter initialized")
        
        logging.debug("Initializing ContextEncoder...")
        encoder = ContextEncoder()
        logging.debug("ContextEncoder initialized")
        
        logging.debug("Initializing BidirectionalValidator...")
        validator = BidirectionalValidator(encoder)
        logging.debug("BidirectionalValidator initialized")
        
        logging.debug("Initializing IntentAwareTransformer...")
        transformer = IntentAwareTransformer(encoder, validator)
        logging.debug("IntentAwareTransformer initialized")
        
        logging.debug("Initializing EmbeddingGenerator...")
        embedding_generator = EmbeddingGenerator()
        logging.debug("EmbeddingGenerator initialized")
        
        logging.info(f"Components initialized in {time.time() - component_start:.2f} seconds")
    except Exception as e:
        logging.error(f"Error initializing components: {str(e)}", exc_info=True)
        return
    
    # Phase 3.2: Get transformation pairs
    logging.info("=== PHASE 3.2: Getting transformation pairs ===")
    pairs_start = time.time()
    
    try:
        logging.debug("Getting transformation pairs...")
        pairs = loader.get_transformation_pairs("translation", limit=limit)
        
        if not pairs:
            logging.warning("No suitable transformation pairs found")
            return
        
        logging.info(f"Found {len(pairs)} transformation pairs in {time.time() - pairs_start:.2f} seconds")
        
        # Log summary of pairs
        for i, pair in enumerate(pairs):
            logging.debug(f"Pair {i+1}: {pair['name']} - {pair['type']}")
            logging.debug(f"  Source: {pair['source']['id']} ({pair['source']['type']})")
            logging.debug(f"  Target: {pair['target']['id']} ({pair['target']['type']})")
    except Exception as e:
        logging.error(f"Error getting transformation pairs: {str(e)}", exc_info=True)
        return
    
    # Phase 3.3: Process pairs
    logging.info("=== PHASE 3.3: Processing transformation pairs ===")
    results = []
    
    for pair_index, pair in enumerate(pairs):
        pair_start = time.time()
        logging.info(f"Processing pair {pair_index+1}/{len(pairs)}: {pair['name']}")
        
        try:
            # Load models
            logging.debug(f"Loading models for pair {pair_index+1}...")
            source_model, source_text = loader.load_model_with_text(pair['source'])
            target_model, target_text = loader.load_model_with_text(pair['target'])
            
            if not source_text or not target_text:
                logging.warning(f"Skipping pair {pair_index+1} due to missing text representations")
                continue
                
            logging.debug(f"Loaded source model: {source_model.id} ({source_model.type})")
            logging.debug(f"Loaded target model: {target_model.id} ({target_model.type})")
            
            # Generate embeddings
            logging.debug(f"Generating embeddings for pair {pair_index+1}...")
            source_embedding = embedding_generator.generate_embedding(source_text)
            target_embedding = embedding_generator.generate_embedding(target_text)
            logging.debug(f"Generated embeddings for pair {pair_index+1}")
            
            # Convert to token pairs
            logging.debug(f"Converting models to token pairs for pair {pair_index+1}...")
            source_token_pairs = adapter.convert_to_token_pairs_with_embeddings(
                source_model, source_text, source_embedding)
            target_token_pairs = adapter.convert_to_token_pairs_with_embeddings(
                target_model, target_text, target_embedding)
            
            logging.debug(f"Converted source model to {len(source_token_pairs)} token pairs")
            logging.debug(f"Converted target model to {len(target_token_pairs)} token pairs")
            
            # Create transformation rules
            logging.debug(f"Creating transformation rules for pair {pair_index+1}...")
            rules = adapter.create_transformation_rules(source_model, target_model, "translation")
            
            # Clear previous rules and add new ones
            transformer.rules_library = []
            for rule in rules:
                transformer.add_rule(rule)
            
            logging.debug(f"Created {len(rules)} transformation rules")
            
            # Transform with validation
            logging.debug(f"Performing transformation with validation for pair {pair_index+1}...")
            transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
                source_model, intent="translation", max_rules=len(rules)
            )
            logging.debug(f"Transformation completed for pair {pair_index+1}")
            
            # Get validation scores
            final_scores = validation_scores[-1] if validation_scores else {
                'forward_validation_score': 0,
                'backward_validation_score': 0,
                'transformation_quality': 0
            }
            
            logging.debug(f"Validating transformation for pair {pair_index+1}...")
            enhanced_backward_validation = validator.compute_backward_validation_score_with_embeddings(
                source_model, transformed_model, source_embedding, target_embedding
            )
            
            # Compute enhanced quality
            alpha = 0.5  # For translation
            enhanced_quality = alpha * final_scores['forward_validation_score'] + (1 - alpha) * enhanced_backward_validation
            
            logging.info(f"Validation scores for pair {pair_index+1}:")
            logging.info(f"  Regular Forward Validation: {final_scores['forward_validation_score']:.4f}")
            logging.info(f"  Regular Backward Validation: {final_scores['backward_validation_score']:.4f}")
            logging.info(f"  Regular Transformation Quality: {final_scores['transformation_quality']:.4f}")
            logging.info(f"  Enhanced Backward Validation: {enhanced_backward_validation:.4f}")
            logging.info(f"  Enhanced Transformation Quality: {enhanced_quality:.4f}")
            
            # Compute direct similarity
            direct_similarity = embedding_generator.compute_similarity(source_text, target_text)
            logging.debug(f"Direct embedding similarity: {direct_similarity:.4f}")
            
            # Store results
            results.append({
                'pair_name': pair['name'],
                'source_id': pair['source']['id'],
                'target_id': pair['target']['id'],
                'regular': {
                    'forward_validation': float(final_scores['forward_validation_score']),
                    'backward_validation': float(final_scores['backward_validation_score']),
                    'transformation_quality': float(final_scores['transformation_quality'])
                },
                'enhanced': {
                    'forward_validation': float(final_scores['forward_validation_score']),
                    'backward_validation': float(enhanced_backward_validation),
                    'transformation_quality': float(enhanced_quality)
                },
                'direct_similarity': float(direct_similarity),
                'applied_rules': [rule.id for rule in applied_rules],
                'improvement': float(enhanced_quality - final_scores['transformation_quality'])
            })
            
            logging.info(f"Completed processing pair {pair_index+1} in {time.time() - pair_start:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error processing pair {pair_index+1}: {str(e)}", exc_info=True)
            continue
    
    # Phase 3.4: Save and visualize results
    if results:
        logging.info("=== PHASE 3.4: Saving and visualizing results ===")
        try:
            # Save JSON results
            logging.debug("Saving results to JSON...")
            results_file = os.path.join(output_dir, 'embedding_enhanced_experiment.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {results_file}")
            
            # Visualize results
            logging.debug("Creating visualizations...")
            visualize_embedding_results(results, output_dir)
            logging.info("Visualizations created successfully")
        except Exception as e:
            logging.error(f"Error saving or visualizing results: {str(e)}", exc_info=True)
    else:
        logging.warning("No results to save or visualize")
    
    logging.info(f"Basic experiment completed in {time.time() - phase_start:.2f} seconds")
    return results

def visualize_embedding_results(results, output_dir):
    """Create visualization comparing regular and enhanced validation - with error handling"""
    logging.debug("Starting visualization creation")
    
    if not results:
        logging.warning("No results to visualize")
        return
    
    try:
        # Create figures directory
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        logging.debug(f"Created figures directory: {figures_dir}")
        
        # Prepare data
        names = [r['pair_name'][:15] + '...' if len(r['pair_name']) > 15 else r['pair_name'] for r in results]
        regular_quality = [r['regular']['transformation_quality'] for r in results]
        enhanced_quality = [r['enhanced']['transformation_quality'] for r in results]
        
        # Create simple bar chart for comparison
        logging.debug("Creating bar chart for quality comparison...")
        plt.figure(figsize=(10, 6))
        x = np.arange(len(names))
        width = 0.35
        
        plt.bar(x - width/2, regular_quality, width, label='Regular')
        plt.bar(x + width/2, enhanced_quality, width, label='Enhanced')
        
        plt.xlabel('Model Pairs')
        plt.ylabel('Transformation Quality')
        plt.title('Regular vs. Enhanced Transformation Quality')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.0)
        plt.tight_layout()
        
        # Save figure
        chart_file = os.path.join(figures_dir, 'quality_comparison.png')
        plt.savefig(chart_file)
        plt.close()
        logging.info(f"Bar chart saved to {chart_file}")
        
    except Exception as e:
        logging.error(f"Error creating visualizations: {str(e)}", exc_info=True)

def main():
    try:
        logging.info("=== PHASE 0: Starting main function ===")
        args = parse_args()
        
        logging.info("==== Embedding-Enhanced Token Pair Bidirectional Validation ====")
        logging.info(f"Arguments: {vars(args)}")
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        logging.debug(f"Ensured output directory exists: {args.output}")
        
        # Run experiment
        if args.experiment in ['basic', 'all']:
            run_basic_experiment_with_embeddings(args.modelset, args.output, args.domain, args.limit)
        
        logging.info("==== Experiments completed ====")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logging.info("Script called from command line")
    main()
    logging.info("Script execution completed")