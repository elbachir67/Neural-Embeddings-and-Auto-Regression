#!/usr/bin/env python3
"""
Embedding-Enhanced Auto-Regression with Optimized Parameters
"""

import os
import sys
import argparse
import logging
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
log_filename = f"autoregression_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logging.info("Starting auto-regression experiment with optimized parameters")

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

def run_optimized_autoregression(modelset_path, output_dir, domain=None, sequence_length=3):
    """Run auto-regression experiment with optimized parameters"""
    logging.info("\nRunning Auto-Regression Experiment with Optimized Parameters...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    loader = ModelSetLoader(modelset_path)
    adapter = TokenPairAdapter()
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    embedding_generator = EmbeddingGenerator()
    
    # Get model sequences for UML and Ecore
    uml_sequence = loader.get_model_sequence(domain, "UML", limit=sequence_length)
    ecore_sequence = loader.get_model_sequence(domain, "Ecore", limit=sequence_length)
    
    if len(uml_sequence) < 2 and len(ecore_sequence) < 2:
        logging.warning("Not enough models found for sequences")
        return
    
    # Choose the longer sequence
    sequence = uml_sequence if len(uml_sequence) >= len(ecore_sequence) else ecore_sequence
    sequence_type = "UML" if sequence == uml_sequence else "Ecore"
    
    if len(sequence) < 2:
        logging.warning("Not enough models in sequence for auto-regression experiment")
        return
    
    logging.info(f"Using sequence of {len(sequence)} {sequence_type} models")
    
    results = []
    
    # Run with different configurations
    configs = [
        {"name": "Baseline", "use_autoregression": False, "use_embeddings": False},
        {"name": "Auto-Regression Only", "use_autoregression": True, "use_embeddings": False},
        {"name": "Embeddings Only", "use_autoregression": False, "use_embeddings": True},
        {"name": "Combined (Optimized)", "use_autoregression": True, "use_embeddings": True}
    ]
    
    # Split sequence: use first n-1 models for history, last model as current
    history_models = sequence[:-1]
    current_model = sequence[-1]
    
    # Process each configuration
    for config in configs:
        logging.info(f"\nRunning with configuration: {config['name']}")
        
        # Determine intent based on sequence type
        intent = "revision" if sequence_type == current_model.type else "translation"
        
        # Generate embeddings if needed
        if config['use_embeddings']:
            # Generate text representations
            history_texts = [model.to_text() for model in history_models]
            current_text = current_model.to_text()
            
            # Generate embeddings
            history_embeddings = [embedding_generator.generate_embedding(text) for text in history_texts]
            current_embedding = embedding_generator.generate_embedding(current_text)
            
            logging.info(f"  Generated embeddings for {len(history_models) + 1} models")
        else:
            history_embeddings = None
            current_embedding = None
        
        # Use optimized alpha and beta values based on intent
        if intent == "translation":
            alpha = 0.5  # Optimized for translation
        else:  # revision
            alpha = 0.7  # Optimized for revision
        beta = 0.7  # Optimized beta value
        
        # Run transformation with current configuration
        result = run_with_config(
            current_model, 
            history_models, 
            adapter, 
            transformer, 
            validator, 
            embedding_generator,
            config['use_autoregression'],
            config['use_embeddings'],
            history_embeddings,
            current_embedding,
            intent,
            alpha,
            beta
        )
        
        # Add configuration info to result
        result['config'] = config['name']
        results.append(result)
        
        logging.info(f"  Config: {config['name']}")
        logging.info(f"  Intent: {intent.upper()}")
        logging.info(f"  Parameters: alpha={alpha:.1f}, beta={beta:.1f}")
        logging.info(f"  Forward Score: {result['forward']:.4f}")
        logging.info(f"  Backward Score: {result['backward']:.4f}")
        logging.info(f"  Quality: {result['quality']:.4f}")
    
    # Calculate improvements relative to baseline
    baseline_quality = next(r['quality'] for r in results if r['config'] == 'Baseline')
    for result in results:
        result['improvement'] = result['quality'] - baseline_quality
        result['improvement_percent'] = (result['improvement'] / baseline_quality) * 100
        logging.info(f"  {result['config']} improvement: {result['improvement_percent']:.2f}%")
    
    # Save results
    with open(os.path.join(output_dir, 'autoregression_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_autoregression_visualization(results, output_dir, intent)
    
    return results

def run_with_config(current_model, history_models, adapter, transformer, validator, 
                   embedding_generator, use_autoregression, use_embeddings,
                   history_embeddings, current_embedding, intent, alpha, beta):
    """Run transformation with specific configuration"""
    
    # Reset transformer rules
    transformer.rules_library = []
    
    # Create rules based on configuration
    if use_autoregression and len(history_models) > 0:
        # Create rules from history
        history_rules = []
        for i in range(len(history_models) - 1):
            source = history_models[i]
            target = history_models[i+1]
            rules = adapter.create_transformation_rules(source, target)
            for rule in rules:
                transformer.add_rule(rule)
                history_rules.append(rule)
        
        logging.info(f"  Created {len(history_rules)} rules from transformation history")
    else:
        # Create basic rules based on model type
        if current_model.type.lower() == 'uml':
            create_default_uml_rules(transformer)
        else:
            create_default_ecore_rules(transformer)
    
    # Run transformation
    transformed_model, applied_rules, _ = transformer.transform_with_validation(
        current_model, 
        intent=intent,
        max_rules=len(transformer.rules_library),
        history_models=history_models if use_autoregression else None,
        history_rules=transformer.rules_library if use_autoregression else None
    )
    
    # Compute validation scores
    fvs = validator.compute_forward_validation_score(transformed_model, applied_rules)
    bvs = validator.compute_backward_validation_score(current_model, transformed_model)
    
    # If using embeddings, enhance backward validation
    if use_embeddings and current_embedding is not None:
        # Generate embedding for transformed model
        transformed_text = transformed_model.to_text()
        transformed_embedding = embedding_generator.generate_embedding(transformed_text)
        
        # Compute enhanced backward validation
        embedding_similarity = embedding_generator.compute_similarity(
            current_model.to_text(), transformed_text)
        enhanced_bvs = beta * bvs + (1 - beta) * embedding_similarity
        
        # Compute quality with enhanced backward validation
        quality = alpha * fvs + (1 - alpha) * enhanced_bvs
    else:
        # Compute quality with standard backward validation
        enhanced_bvs = bvs
        quality = alpha * fvs + (1 - alpha) * bvs
    
    return {
        'forward': float(fvs),
        'backward': float(bvs),
        'enhanced_backward': float(enhanced_bvs),
        'quality': float(quality),
        'applied_rules': [rule.id for rule in applied_rules],
        'intent': intent,
        'alpha': float(alpha),
        'beta': float(beta)
    }

def create_default_uml_rules(transformer):
    """Create default rules for UML models"""
    from bidirectional_validator import TransformationRule
    
    rules = [
        ("ClassToEClass", "Class", "EClass", "translation", 
         ["UML Classes must be transformed to Ecore EClasses"]),
        ("PropertyToEAttribute", "Property", "EAttribute", "translation", 
         ["UML Properties must be transformed to Ecore EAttributes"]),
        ("AssociationToEReference", "Association", "EReference", "translation", 
         ["UML Associations must be transformed to Ecore EReferences"])
    ]
    
    for rule_id, source_pattern, target_pattern, intent, constraints in rules:
        rule = TransformationRule(rule_id, source_pattern, target_pattern, intent, constraints)
        transformer.add_rule(rule)
    
    logging.info(f"  Created {len(rules)} default UML rules")

def create_default_ecore_rules(transformer):
    """Create default rules for Ecore models"""
    from bidirectional_validator import TransformationRule
    
    rules = [
        ("EClassToClass", "EClass", "Class", "translation", 
         ["Ecore EClasses must be transformed to UML Classes"]),
        ("EAttributeToProperty", "EAttribute", "Property", "translation", 
         ["Ecore EAttributes must be transformed to UML Properties"]),
        ("EReferenceToAssociation", "EReference", "Association", "translation", 
         ["Ecore EReferences must be transformed to UML Associations"])
    ]
    
    for rule_id, source_pattern, target_pattern, intent, constraints in rules:
        rule = TransformationRule(rule_id, source_pattern, target_pattern, intent, constraints)
        transformer.add_rule(rule)
    
    logging.info(f"  Created {len(rules)} default Ecore rules")

def create_autoregression_visualization(results, output_dir, intent):
    """Create visualization of auto-regression results"""
    # Extract data
    configs = [r['config'] for r in results]
    qualities = [r['quality'] for r in results]
    improvements = [r['improvement_percent'] for r in results]
    
    # Create directory for figures
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Quality scores
    bars1 = ax1.bar(configs, qualities, color=['lightgray', 'lightblue', 'lightgreen', 'purple'])
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Transformation Quality')
    ax1.set_title(f'Transformation Quality by Configuration ({intent.capitalize()})')
    ax1.set_ylim(0.8, 1.0)  # Adjust based on your score range
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    # Improvement percentages
    bars2 = ax2.bar(configs[1:], improvements[1:], 
                   color=['lightblue', 'lightgreen', 'purple'])
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title(f'Improvement Over Baseline ({intent.capitalize()})')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    # Add horizontal line at y=0
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'autoregression_{intent}.png'), dpi=300)
    plt.savefig(os.path.join(figures_dir, f'autoregression_{intent}.pdf'))
    plt.close()
    
    # Create combined visualization
    plt.figure(figsize=(10, 6))
    
    # Extract data for configurations
    baseline = next(r for r in results if r['config'] == 'Baseline')
    autoregression = next(r for r in results if r['config'] == 'Auto-Regression Only')
    embeddings = next(r for r in results if r['config'] == 'Embeddings Only')
    combined = next(r for r in results if r['config'] == 'Combined (Optimized)')
    
    # Create grouped bar chart for comparison
    x = np.arange(3)  # forward, backward, quality
    width = 0.2
    
    plt.bar(x - width*1.5, 
            [baseline['forward'], baseline['backward'], baseline['quality']], 
            width, label='Baseline')
    
    plt.bar(x - width/2, 
            [autoregression['forward'], autoregression['backward'], autoregression['quality']], 
            width, label='Auto-Regression')
    
    plt.bar(x + width/2, 
            [embeddings['forward'], embeddings['enhanced_backward'], embeddings['quality']], 
            width, label='Embeddings')
    
    plt.bar(x + width*1.5, 
            [combined['forward'], combined['enhanced_backward'], combined['quality']], 
            width, label='Combined')
    
    plt.xlabel('Validation Metric')
    plt.ylabel('Score')
    plt.title(f'Comparison of Approaches ({intent.capitalize()})')
    plt.xticks(x, ['Forward Validation', 'Backward Validation', 'Transformation Quality'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'autoregression_comparison_{intent}.png'), dpi=300)
    plt.savefig(os.path.join(figures_dir, f'autoregression_comparison_{intent}.pdf'))
    plt.close()
    
    logging.info(f"Created visualizations in {figures_dir}")

def main():
    """Main function to run auto-regression experiments with optimized parameters"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run auto-regression experiment with optimized parameters')
    
    parser.add_argument('--modelset', type=str, required=True,
                        help='Path to the ModelSet dataset')
    
    parser.add_argument('--output', type=str, default='autoregression_results',
                        help='Directory to save experiment results')
    
    parser.add_argument('--domain', type=str, default=None,
                        help='Domain to use for model sequence (e.g., travel, statemachine)')
    
    parser.add_argument('--sequence', type=int, default=3,
                        help='Length of model sequence to use')
    
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("Auto-Regression Experiment with Optimized Parameters")
    logging.info("=" * 80)
    logging.info(f"ModelSet path: {args.modelset}")
    logging.info(f"Output directory: {args.output}")
    logging.info(f"Domain: {args.domain if args.domain else 'auto-select'}")
    logging.info(f"Sequence length: {args.sequence}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run auto-regression experiment with optimized parameters
    results = run_optimized_autoregression(
        args.modelset, args.output, args.domain, args.sequence)
    
    logging.info("\n" + "=" * 80)
    logging.info(f"Auto-regression experiment complete! Results saved in {args.output}")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()