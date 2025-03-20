#!/usr/bin/env python3
import os
import sys
import argparse
import json
import traceback
import logging
import datetime

# Configure logging
log_filename = f"embedding_experiments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also output to console
    ]
)
logging.info("Starting embedding experiments script")

# First import matplotlib and set backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    from bidirectional_validator import ContextEncoder, BidirectionalValidator, IntentAwareTransformer, TransformationRule
    from modelset_loader import ModelSetLoader
    from token_pair_adapter import TokenPairAdapter
    from embedding_generator import EmbeddingGenerator
    logging.info("Successfully imported custom modules")
except Exception as e:
    logging.error(f"Error importing custom modules: {str(e)}", exc_info=True)
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
    
    args = parser.parse_args()
    return args

def run_basic_experiment_with_embeddings(modelset_path, output_dir, domain=None, limit=5):
    """Run basic experiment with embedding enhancement"""
    logging.info("\nRunning Embedding-Enhanced Basic Experiment...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    try:
        logging.info("Initializing components...")
        loader = ModelSetLoader(modelset_path)
        adapter = TokenPairAdapter()
        encoder = ContextEncoder()
        validator = BidirectionalValidator(encoder)
        transformer = IntentAwareTransformer(encoder, validator)
        embedding_generator = EmbeddingGenerator()
        logging.info("Components initialized")
    except Exception as e:
        logging.error(f"Error initializing components: {str(e)}", exc_info=True)
        return
    
    # Get transformation pairs
    try:
        logging.info("Getting transformation pairs...")
        pairs = loader.get_transformation_pairs("translation", limit=limit)
        
        if not pairs:
            logging.warning("No suitable transformation pairs found")
            return
        
        logging.info(f"Found {len(pairs)} transformation pairs")
    except Exception as e:
        logging.error(f"Error getting transformation pairs: {str(e)}", exc_info=True)
        return
    
    results = []
    
    # Process each pair
    for pair_index, pair in enumerate(pairs):
        logging.info(f"\nProcessing pair {pair_index+1}: {pair['name']}")
        
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
            
            # Convert models to token pairs with embeddings
            source_token_pairs = adapter.convert_to_token_pairs_with_embeddings(
                source_model, source_text, source_embedding)
            target_token_pairs = adapter.convert_to_token_pairs_with_embeddings(
                target_model, target_text, target_embedding)
            
            logging.info(f"  Converted source model to {len(source_token_pairs)} token pairs")
            logging.info(f"  Converted target model to {len(target_token_pairs)} token pairs")
            
            # Create transformation rules
            rules = adapter.create_transformation_rules(source_model, target_model, "translation")
            
            # Add rules to transformer
            transformer.rules_library = []  # Clear previous rules
            for rule in rules:
                transformer.add_rule(rule)
            
            logging.info(f"  Created {len(rules)} transformation rules")
            
            # Transform with validation - regular approach
            transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
                source_model, intent="translation", max_rules=len(rules)
            )
            
            # Get regular validation scores
            final_scores = validation_scores[-1] if validation_scores else {
                'forward_validation_score': 0,
                'backward_validation_score': 0,
                'transformation_quality': 0
            }
            
            # Compute embedding-enhanced backward validation
            enhanced_backward_validation = validator.compute_backward_validation_score_with_embeddings(
                source_model, transformed_model, source_embedding, target_embedding
            )
            
            # Compute enhanced transformation quality
            alpha = 0.75 if pair['type'] == 'revision' else 0.5
            enhanced_quality = alpha * final_scores['forward_validation_score'] + (1 - alpha) * enhanced_backward_validation
            
            logging.info(f"  Regular Forward Validation: {final_scores['forward_validation_score']:.4f}")
            logging.info(f"  Regular Backward Validation: {final_scores['backward_validation_score']:.4f}")
            logging.info(f"  Regular Transformation Quality: {final_scores['transformation_quality']:.4f}")
            logging.info(f"  Enhanced Backward Validation: {enhanced_backward_validation:.4f}")
            logging.info(f"  Enhanced Transformation Quality: {enhanced_quality:.4f}")
            
            # Direct embedding similarity
            direct_similarity = embedding_generator.compute_similarity(source_text, target_text)
            logging.info(f"  Direct embedding similarity: {direct_similarity:.4f}")
            
            # Token pair similarity
            try:
                token_pair_similarity = adapter.compute_token_pair_similarity(source_token_pairs, target_token_pairs)
                logging.info(f"  Token pair similarity: {token_pair_similarity:.4f}")
            except Exception as e:
                token_pair_similarity = 0.0
                logging.warning(f"  Could not compute token pair similarity: {str(e)}")
            
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
                'comparisons': {
                    'direct_embedding_similarity': float(direct_similarity),
                    'token_pair_similarity': float(token_pair_similarity)
                },
                'applied_rules': [rule.id for rule in applied_rules],
                'improvement': float(enhanced_quality - final_scores['transformation_quality'])
            })
            
        except Exception as e:
            logging.error(f"  Error processing pair: {str(e)}", exc_info=True)
            continue
    
    # Save results
    try:
        if results:
            # Save JSON results
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'embedding_enhanced_experiment.json'), 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {os.path.join(output_dir, 'embedding_enhanced_experiment.json')}")
            
            # Visualize results
            visualize_embedding_results(results, output_dir)
        else:
            logging.warning("No results to save or visualize")
    except Exception as e:
        logging.error(f"Error saving or visualizing results: {str(e)}", exc_info=True)
    
    return results

def visualize_embedding_results(results, output_dir):
    """Create visualization comparing regular and enhanced validation"""
    try:
        if not results:
            logging.warning("No results to visualize")
            return
        
        # Create figures directory
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Prepare data
        names = [r['pair_name'][:15] + '...' if len(r['pair_name']) > 15 else r['pair_name'] for r in results]
        regular_quality = [r['regular']['transformation_quality'] for r in results]
        enhanced_quality = [r['enhanced']['transformation_quality'] for r in results]
        improvements = [r['improvement'] for r in results]
        
        x = np.arange(len(names))
        width = 0.35
        
        # Create bar chart for quality comparison
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, regular_quality, width, label='Regular Approach', color='#1f77b4')
        plt.bar(x + width/2, enhanced_quality, width, label='Embedding-Enhanced', color='#ff7f0e')
        
        plt.xlabel('Model Pairs')
        plt.ylabel('Transformation Quality')
        plt.title('Regular vs. Embedding-Enhanced Transformation Quality')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend(loc='upper left')
        plt.ylim(0, 1.0)
        
        # Add value labels
        for i, (reg, enh) in enumerate(zip(regular_quality, enhanced_quality)):
            plt.text(i - width/2, reg + 0.02, f'{reg:.3f}', ha='center')
            plt.text(i + width/2, enh + 0.02, f'{enh:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'quality_comparison.png'))
        plt.close()
        logging.info(f"Quality comparison chart saved to {os.path.join(figures_dir, 'quality_comparison.png')}")
        
    except Exception as e:
        logging.error(f"Error visualizing results: {str(e)}", exc_info=True)

def run_comparison_experiment(modelset_path, output_dir, domain=None, limit=3):
    """Run a detailed comparison between regular and embedding-enhanced approaches"""
    logging.info("\nRunning Detailed Comparison Experiment...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize components
        logging.info("Initializing components...")
        loader = ModelSetLoader(modelset_path)
        adapter = TokenPairAdapter()
        encoder = ContextEncoder()
        validator = BidirectionalValidator(encoder)
        transformer = IntentAwareTransformer(encoder, validator)
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
    
    # Process each pair
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
            direct_similarity = embedding_generator.compute_similarity(source_text, target_text)
            logging.info(f"  Direct embedding similarity: {direct_similarity:.4f}")
            
            # Create transformation rules
            rules = adapter.create_transformation_rules(source_model, target_model, pair['type'])
            
            # Add rules to transformer
            transformer.rules_library = []  # Clear previous rules
            for rule in rules:
                transformer.add_rule(rule)
            
            logging.info(f"  Created {len(rules)} transformation rules")
            
            # Transform with validation - regular approach
            transformed_model, applied_rules, validation_scores = transformer.transform_with_validation(
                source_model, intent=pair['type'], max_rules=len(rules)
            )
            
            # Get regular validation scores
            final_scores = validation_scores[-1] if validation_scores else {
                'forward_validation_score': 0,
                'backward_validation_score': 0,
                'transformation_quality': 0
            }
            
            # Compute embedding-enhanced backward validation
            enhanced_backward_validation = validator.compute_backward_validation_score_with_embeddings(
                source_model, transformed_model, source_embedding, target_embedding
            )
            
            # Compute enhanced transformation quality
            alpha = 0.75 if pair['type'] == 'revision' else 0.5
            enhanced_quality = alpha * final_scores['forward_validation_score'] + (1 - alpha) * enhanced_backward_validation
            
            logging.info(f"  Regular Forward Validation: {final_scores['forward_validation_score']:.4f}")
            logging.info(f"  Regular Backward Validation: {final_scores['backward_validation_score']:.4f}")
            logging.info(f"  Regular Transformation Quality: {final_scores['transformation_quality']:.4f}")
            logging.info(f"  Enhanced Backward Validation: {enhanced_backward_validation:.4f}")
            logging.info(f"  Enhanced Transformation Quality: {enhanced_quality:.4f}")
            logging.info(f"  Improvement: {(enhanced_quality - final_scores['transformation_quality'])*100:.2f}%")
            
            # Store results
            results.append({
                'pair_name': pair['name'],
                'pair_type': pair['type'],
                'source_id': pair['source']['id'],
                'target_id': pair['target']['id'],
                'direct_embedding_similarity': float(direct_similarity),
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
                'improvement': float(enhanced_quality - final_scores['transformation_quality']),
                'applied_rules': [rule.id for rule in applied_rules]
            })
            
        except Exception as e:
            logging.error(f"  Error processing pair: {str(e)}", exc_info=True)
            continue
    
    # Save results
    try:
        if results:
            # Save JSON results
            with open(os.path.join(output_dir, 'detailed_comparison.json'), 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {os.path.join(output_dir, 'detailed_comparison.json')}")
            
            # Create detailed visualization
            visualize_detailed_comparison(results, output_dir)
        else:
            logging.warning("No results to save or visualize")
    except Exception as e:
        logging.error(f"Error saving or visualizing results: {str(e)}", exc_info=True)
    
    return results

def visualize_detailed_comparison(results, output_dir):
    """Create a detailed visualization comparing approaches by transformation type"""
    try:
        if not results:
            logging.warning("No results to visualize")
            return
        
        # Create figures directory
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Split by transformation type
        translation_results = [r for r in results if r['pair_type'] == 'translation']
        revision_results = [r for r in results if r['pair_type'] == 'revision']
        
        # Create simple comparison chart
        plt.figure(figsize=(12, 8))
        
        # Group data by type
        pair_names = []
        regular_quality = []
        enhanced_quality = []
        pair_types = []
        
        for result in results:
            pair_names.append(result['pair_name'][:15] + '...' if len(result['pair_name']) > 15 else result['pair_name'])
            regular_quality.append(result['regular']['transformation_quality'])
            enhanced_quality.append(result['enhanced']['transformation_quality'])
            pair_types.append(result['pair_type'])
        
        x = np.arange(len(pair_names))
        width = 0.35
        
        # Create bars with different colors for each type
        plt.bar(x - width/2, regular_quality, width, label='Regular', 
                color=['blue' if t == 'translation' else 'green' for t in pair_types])
        plt.bar(x + width/2, enhanced_quality, width, label='Enhanced',
                color=['red' if t == 'translation' else 'orange' for t in pair_types])
        
        plt.xlabel('Model Pairs')
        plt.ylabel('Transformation Quality')
        plt.title('Comparison by Transformation Type')
        plt.xticks(x, pair_names, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.0)
        
        # Add type labels
        for i, pair_type in enumerate(pair_types):
            plt.text(i, -0.05, pair_type[0].upper(), color='black', fontweight='bold', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'type_comparison.png'))
        plt.close()
        logging.info(f"Type comparison chart saved to {os.path.join(figures_dir, 'type_comparison.png')}")
        
    except Exception as e:
        logging.error(f"Error creating detailed comparison visualizations: {str(e)}", exc_info=True)

def main():
    """Main function to run experiments"""
    try:
        logging.info("Parsing arguments...")
        args = parse_args()
        
        logging.info("=" * 80)
        logging.info("Embedding-Enhanced Token Pair Bidirectional Validation")
        logging.info("=" * 80)
        logging.info(f"Arguments: {vars(args)}")
        
        # Create output directory
        logging.info(f"Creating output directory: {args.output}")
        os.makedirs(args.output, exist_ok=True)
        
        # Create figures directory
        figures_dir = os.path.join(args.output, 'figures')
        logging.info(f"Creating figures directory: {figures_dir}")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Run experiments
        if args.experiment in ['basic', 'all']:
            logging.info(f"Running basic experiment with modelset={args.modelset}")
            run_basic_experiment_with_embeddings(args.modelset, args.output, args.domain, args.limit)
        
        if args.experiment in ['compare', 'all']:
            logging.info(f"Running comparison experiment with modelset={args.modelset}")
            run_comparison_experiment(args.modelset, args.output, args.domain, args.limit)
        
        logging.info("\n" + "=" * 80)
        logging.info("Experiments complete! Results saved in " + args.output)
        logging.info("=" * 80)
        logging.info(f"Figures saved in {figures_dir}")
        logging.info("=" * 80)
    
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logging.info("Script called from command line")
    main()
    logging.info("Script execution completed")