import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import networkx as nx
import json
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ========================= MODEL REPRESENTATION =========================

class ModelGraph:
    """Represents a model as a graph structure"""
    
    def __init__(self, model_id, model_type):
        self.id = model_id
        self.type = model_type  # 'source' or 'target'
        self.graph = nx.DiGraph()
        
    def add_node(self, node_id, node_type, attributes=None):
        """Add a node to the model graph"""
        self.graph.add_node(node_id, type=node_type, attrs=attributes or {})
        
    def add_edge(self, source_id, target_id, edge_type, attributes=None):
        """Add an edge between nodes in the model graph"""
        self.graph.add_edge(source_id, target_id, type=edge_type, attrs=attributes or {})
        
    def to_text(self):
        """Convert model to text representation for LLM processing"""
        text = f"Model {self.id} of type {self.type}:\n"
        
        # Add nodes information
        text += "Nodes:\n"
        for node_id, node_data in self.graph.nodes(data=True):
            attrs_str = ", ".join([f"{k}={v}" for k, v in node_data.get('attrs', {}).items()])
            text += f"  - {node_id} ({node_data['type']}): {attrs_str}\n"
        
        # Add edges information
        text += "Edges:\n"
        for source, target, edge_data in self.graph.edges(data=True):
            attrs_str = ", ".join([f"{k}={v}" for k, v in edge_data.get('attrs', {}).items()])
            text += f"  - {source} -> {target} ({edge_data['type']}): {attrs_str}\n"
            
        return text
    
    @staticmethod
    def from_dict(data):
        """Create a model graph from a dictionary representation"""
        model = ModelGraph(data['id'], data['type'])
        
        # Add nodes
        for node in data.get('nodes', []):
            model.add_node(node['id'], node['type'], node.get('attributes', {}))
            
        # Add edges
        for edge in data.get('edges', []):
            model.add_edge(edge['source'], edge['target'], edge['type'], edge.get('attributes', {}))
            
        return model


# ========================= TRANSFORMATION RULES =========================

class TransformationRule:
    """Represents a transformation rule"""
    
    def __init__(self, rule_id, source_pattern, target_pattern, intent=None, constraints=None):
        self.id = rule_id
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern
        self.intent = intent or "revision"  # Default intent
        self.constraints = constraints or []
        
    def to_text(self):
        """Convert rule to text representation for LLM processing"""
        text = f"Rule {self.id} (Intent: {self.intent}):\n"
        text += f"  Source pattern: {self.source_pattern}\n"
        text += f"  Target pattern: {self.target_pattern}\n"
        
        if self.constraints:
            text += "  Constraints:\n"
            for constraint in self.constraints:
                text += f"    - {constraint}\n"
                
        return text


# ========================= LLM-BASED CONTEXT ENCODER =========================

class ContextEncoder:
    """LLM-based encoder for transformation context"""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode(self, text):
        """Encode text into embedding vector"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token embedding as the representation
        return outputs.last_hidden_state[:, 0, :].numpy()
    
    def encode_model(self, model):
        """Encode a model graph into embedding vector"""
        return self.encode(model.to_text())
    
    def encode_rule(self, rule):
        """Encode a transformation rule into embedding vector"""
        return self.encode(rule.to_text())
    
    def encode_history(self, history_models, applied_rules):
        """Encode transformation history"""
        history_text = "Transformation history:\n"
        
        for i, (model, rule) in enumerate(zip(history_models, applied_rules)):
            history_text += f"Step {i+1}:\n"
            history_text += f"Model: {model.id}\n"
            history_text += f"Applied rule: {rule.id}\n\n"
            
        return self.encode(history_text)
    
    def encode_intent(self, intent):
        """Encode transformation intent"""
        intent_text = f"Transformation intent: {intent}\n"
        if intent == "revision":
            intent_text += "Focus on predicting the next delta while maintaining the same metamodel."
        elif intent == "translation":
            intent_text += "Focus on converting between different metamodels while preserving semantics."
        
        return self.encode(intent_text)


# ========================= BIDIRECTIONAL VALIDATOR =========================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BidirectionalValidator:
    """Enhanced Bidirectional Validator with embedding support"""
    
    def __init__(self, encoder):
        self.encoder = encoder
        
    def compute_forward_validation_score(self, target_model, rules):
        """
        Compute how well the target model conforms to transformation rules
        Returns a score between 0 and 1
        """
        # In a real implementation, this would be more sophisticated
        # For simplicity, we'll use a basic compliance check
        
        target_embedding = self.encoder.encode_model(target_model)
        rules_embedding = np.mean([self.encoder.encode_rule(rule) for rule in rules], axis=0)
        
        # Measure cosine similarity between target model and rules
        similarity = cosine_similarity(target_embedding, rules_embedding.reshape(1, -1))[0][0]
        
        # Normalize to [0, 1]
        return max(0, min(1, (similarity + 1) / 2))
    
    def compute_backward_validation_score(self, source_model, target_model, rules=None):
        """
        Compute how well the transformation preserves semantics of the source model
        Returns a score between 0 and 1
        """
        source_embedding = self.encoder.encode_model(source_model)
        target_embedding = self.encoder.encode_model(target_model)
        
        # Measure cosine similarity between source and target models
        similarity = cosine_similarity(source_embedding, target_embedding)[0][0]
        
        # Normalize to [0, 1]
        return max(0, min(1, (similarity + 1) / 2))
    
    def compute_backward_validation_score_with_embeddings(self, source_model, target_model, 
                                                      source_embedding, target_embedding, 
                                                      rules=None):
        """
        Compute how well the transformation preserves semantics using both token pairs and embeddings
        
        Args:
            source_model: Source model
            target_model: Target model
            source_embedding: Embedding of source model text
            target_embedding: Embedding of target model text
            rules: Optional transformation rules used
        
        Returns:
            Enhanced backward validation score
        """
        # Compute base score using token pairs - call the standard method
        base_score = self.compute_backward_validation_score(source_model, target_model, rules)
        
        # Compute embedding similarity directly from provided embeddings
        embedding_similarity = self._compute_embedding_similarity(source_embedding, target_embedding)
        
        # Blend scores with configurable weights
        alpha = 0.4  # Weight for token-pair based score
        enhanced_score = alpha * base_score + (1 - alpha) * embedding_similarity
        
        return enhanced_score
    
    def _compute_embedding_similarity(self, embedding1, embedding2):
        """
        Compute similarity between embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Reshape embeddings if needed
        if len(embedding1.shape) == 1:
            embedding1 = embedding1.reshape(1, -1)
        if len(embedding2.shape) == 1:
            embedding2 = embedding2.reshape(1, -1)
            
        # Compute cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # Normalize to [0, 1]
        normalized_similarity = (similarity + 1) / 2
        
        return normalized_similarity
    
    def compute_transformation_quality(self, source_model, target_model, rules, intent):
        """
        Compute overall transformation quality as weighted combination
        of forward and backward validation scores
        """
        fvs = self.compute_forward_validation_score(target_model, rules)
        bvs = self.compute_backward_validation_score(source_model, target_model)
        
        alpha = 0.7 if intent == "revision" else 0.4
        
        # Compute weighted combination
        tq = alpha * fvs + (1 - alpha) * bvs
        
        return {
            "transformation_quality": tq,
            "forward_validation_score": fvs,
            "backward_validation_score": bvs
        }
    
    def compute_enhanced_transformation_quality(self, source_model, target_model, rules, 
                                           source_embedding, target_embedding, intent):
        """
        Compute enhanced transformation quality using optimized parameters
        
        Args:
            source_model: Source model
            target_model: Target model
            rules: Transformation rules
            source_embedding: Embedding of source model
            target_embedding: Embedding of target model
            intent: Transformation intent
            
        Returns:
            Dictionary with validation scores and quality metrics
        """
        # Compute forward validation score
        fvs = self.compute_forward_validation_score(target_model, rules)
        
        # Compute standard backward validation score
        standard_bvs = self.compute_backward_validation_score(source_model, target_model)
        
        # Compute embedding similarity
        embedding_similarity = self._compute_embedding_similarity(source_embedding, target_embedding)
        
        # Use optimized beta value from parameter optimization (0.7)
        beta = 0.7
        
        # Compute enhanced backward validation score
        enhanced_bvs = beta * standard_bvs + (1 - beta) * embedding_similarity
        
        # Use intent-specific alpha values from parameter optimization
        if intent == "translation":
            alpha = 0.5  # Optimized value for translation
        else:  # revision
            alpha = 0.7  # Optimized value for revision
        
        # Compute weighted combinations
        enhanced_tq = alpha * fvs + (1 - alpha) * enhanced_bvs
        standard_tq = alpha * fvs + (1 - alpha) * standard_bvs
        
        # Calculate improvement
        improvement = enhanced_tq - standard_tq
        
        return {
            "enhanced_transformation_quality": enhanced_tq,
            "standard_transformation_quality": standard_tq,
            "forward_validation_score": fvs,
            "enhanced_backward_validation_score": enhanced_bvs,
            "standard_backward_validation_score": standard_bvs,
            "embedding_similarity": embedding_similarity,
            "improvement": improvement,
            "intent": intent,
            "alpha": alpha,
            "beta": beta
        }
    
    def evaluate_transformation_pair(self, source_model, target_model, rules, 
                                   source_embedding, target_embedding, intent):
        """
        Comprehensive evaluation of a transformation pair
        
        Args:
            source_model: Source model
            target_model: Target model
            rules: Transformation rules
            source_embedding: Embedding of source model
            target_embedding: Embedding of target model
            intent: Transformation intent
            
        Returns:
            Comprehensive evaluation results
        """
        # Get standard and enhanced quality metrics
        quality_metrics = self.compute_enhanced_transformation_quality(
            source_model, target_model, rules, source_embedding, target_embedding, intent
        )
        
        # Compute direct embedding similarity
        direct_similarity = self._compute_embedding_similarity(source_embedding, target_embedding)
        
        # Add additional metrics
        results = {
            **quality_metrics,
            "direct_embedding_similarity": direct_similarity,
            "source_model_id": source_model.id,
            "target_model_id": target_model.id,
            "source_model_type": source_model.type,
            "target_model_type": target_model.type,
            "applied_rules": [rule.id for rule in rules],
            "num_rules": len(rules)
        }
        
        return results

    # Add these methods to the BidirectionalValidator class in bidirectional_validator.py

    def compute_backward_validation_score_with_token_pairs(self, source_model, target_model, 
                                                        source_token_pairs, target_token_pairs,
                                                        source_embedding=None, target_embedding=None):
        """
        Compute how well the transformation preserves semantics using token pairs
        
        Args:
            source_model: Source model
            target_model: Target model
            source_token_pairs: Token pairs from source model
            target_token_pairs: Token pairs from target model
            source_embedding: Optional source model embedding
            target_embedding: Optional target model embedding
            
        Returns:
            Enhanced backward validation score
        """
        # Compute standard backward validation
        standard_score = self.compute_backward_validation_score(source_model, target_model)
        
        # Import TokenPairAdapter locally to avoid circular imports
        from token_pair_adapter import TokenPairAdapter
        adapter = TokenPairAdapter()
        
        # Compute token pair similarity
        token_pair_similarity = adapter.compute_token_pair_similarity(source_token_pairs, target_token_pairs)
        
        # Compute embedding similarity if embeddings are provided
        embedding_similarity = 0.0
        if source_embedding is not None and target_embedding is not None:
            embedding_similarity = self._compute_embedding_similarity(source_embedding, target_embedding)
        
        # Determine weights based on number of token pairs (more pairs = more weight to token pairs)
        token_pair_weight = min(0.6, 0.3 + 0.01 * min(len(source_token_pairs), 30))
        embedding_weight = 0.2
        standard_weight = 1.0 - token_pair_weight - embedding_weight
        
        # Weighted combination
        enhanced_score = (standard_weight * standard_score + 
                        token_pair_weight * token_pair_similarity + 
                        embedding_weight * embedding_similarity)
        
        return enhanced_score

    def compute_enhanced_transformation_quality_with_token_pairs(self, source_model, target_model, rules, 
                                                            source_token_pairs, target_token_pairs,
                                                            source_embedding=None, target_embedding=None, 
                                                            intent="translation"):
        """
        Compute enhanced transformation quality using token pairs
        
        Args:
            source_model: Source model
            target_model: Target model
            rules: Transformation rules
            source_token_pairs: Token pairs from source model
            target_token_pairs: Token pairs from target model
            source_embedding: Optional source model embedding
            target_embedding: Optional target model embedding
            intent: Transformation intent
            
        Returns:
            Dictionary with validation scores and quality metrics
        """
        # Compute forward validation score (remains the same)
        fvs = self.compute_forward_validation_score(target_model, rules)
        
        # Compute enhanced backward validation score using token pairs
        enhanced_bvs = self.compute_backward_validation_score_with_token_pairs(
            source_model, target_model, source_token_pairs, target_token_pairs,
            source_embedding, target_embedding
        )
        
        # Compute standard backward validation score for comparison
        standard_bvs = self.compute_backward_validation_score(source_model, target_model)
        
        # Set alpha based on intent
        alpha = 0.75 if intent == "revision" else 0.5
        
        # Compute weighted combinations
        enhanced_tq = alpha * fvs + (1 - alpha) * enhanced_bvs
        standard_tq = alpha * fvs + (1 - alpha) * standard_bvs
        
        # Calculate improvement
        improvement = enhanced_tq - standard_tq
        
        return {
            "enhanced_transformation_quality": enhanced_tq,
            "standard_transformation_quality": standard_tq,
            "forward_validation_score": fvs,
            "enhanced_backward_validation_score": enhanced_bvs,
            "standard_backward_validation_score": standard_bvs,
            "improvement": improvement,
            "intent": intent
        }


# ========================= INTENT-AWARE TRANSFORMER =========================

class IntentAwareTransformer:
    """Model transformer with intent awareness and auto-regression"""
    
    def __init__(self, encoder, validator):
        self.encoder = encoder
        self.validator = validator
        self.rules_library = []
        self.transformation_history = []
        
    def add_rule(self, rule):
        """Add a transformation rule to the library"""
        self.rules_library.append(rule)
        
    def predict_next_rule(self, source_model, partial_target_model, intent, history_models=None, history_rules=None):
        """
        Predict the next transformation rule to apply based on:
        - Current source model
        - Partial target model (output so far)
        - Transformation intent
        - Historical context (previous models and rules in the chain)
        """
        # Encode current context
        source_embedding = self.encoder.encode_model(source_model)
        target_embedding = self.encoder.encode_model(partial_target_model) if partial_target_model else None
        intent_embedding = self.encoder.encode_intent(intent)
        
        # Encode history if available
        history_embedding = None
        if history_models and history_rules:
            history_embedding = self.encoder.encode_history(history_models, history_rules)
        
        # Score each rule based on the context
        rule_scores = []
        for rule in self.rules_library:
            rule_embedding = self.encoder.encode_rule(rule)
            
            # Compute similarity with source model
            source_sim = cosine_similarity(source_embedding, rule_embedding.reshape(1, -1))[0][0]
            
            # Compute similarity with intent
            intent_sim = cosine_similarity(intent_embedding, rule_embedding.reshape(1, -1))[0][0]
            
            # Compute similarity with history if available
            history_sim = 0
            if history_embedding is not None:
                history_sim = cosine_similarity(history_embedding, rule_embedding.reshape(1, -1))[0][0]
            
            # Compute overall score (higher is better)
            score = source_sim * 0.4 + intent_sim * 0.3
            if history_embedding is not None:
                score += history_sim * 0.3
            
            rule_scores.append((rule, score))
        
        # Sort rules by score (descending)
        rule_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best rule
        if rule_scores:
            return rule_scores[0][0]
        return None
    
    def apply_rule(self, rule, source_model, target_model=None):
        """
        Apply a transformation rule to transform source model
        If target_model is provided, it's updated incrementally
        Otherwise, a new target model is created
        
        In a real implementation, this would parse the rule patterns
        and apply them to the model structures. Here we use a simplified
        simulation of this process.
        """
        if target_model is None:
            # Create a new target model
            target_model = ModelGraph(f"{source_model.id}_transformed", "target")
        
        # Simulate rule application by creating or modifying target model elements
        # In a real implementation, this would be a complex model transformation process
        
        # For demonstration purposes, we'll just transfer some nodes and edges
        # with modifications based on the rule
        for node_id, node_data in source_model.graph.nodes(data=True):
            # Apply transformations based on rule patterns
            if rule.source_pattern in node_data.get('type', ''):
                # Transform node type based on rule
                new_type = node_data['type'].replace(
                    rule.source_pattern, rule.target_pattern
                )
                
                # Create transformed node in target model
                target_model.add_node(
                    f"{node_id}_transformed", 
                    new_type,
                    node_data.get('attrs', {})
                )
        
        # Transform edges
        for source, target, edge_data in source_model.graph.edges(data=True):
            # Create corresponding edge in target model if both nodes exist
            source_t = f"{source}_transformed"
            target_t = f"{target}_transformed"
            
            if (source_t in target_model.graph.nodes() and 
                target_t in target_model.graph.nodes()):
                
                target_model.add_edge(
                    source_t, 
                    target_t,
                    edge_data['type'],
                    edge_data.get('attrs', {})
                )
        
        return target_model
    
    def transform_with_validation(self, source_model, intent, max_rules=3, 
                                  history_models=None, history_rules=None):
        """
        Transform a source model using intent-aware rules with bidirectional validation
        Uses auto-regression from transformation history
        """
        target_model = None
        applied_rules = []
        validation_scores = []
        
        # Apply rules incrementally until max_rules is reached
        for _ in range(max_rules):
            # Predict next rule
            next_rule = self.predict_next_rule(
                source_model, target_model, intent, history_models, history_rules
            )
            
            if next_rule is None:
                break
                
            # Apply the rule
            target_model = self.apply_rule(next_rule, source_model, target_model)
            applied_rules.append(next_rule)
            
            # Validate the transformation
            scores = self.validator.compute_transformation_quality(
                source_model, target_model, applied_rules, intent
            )
            validation_scores.append(scores)
            
            # Record in transformation history
            self.transformation_history.append({
                'source_model': source_model.id,
                'target_model': target_model.id,
                'rule': next_rule.id,
                'intent': intent,
                'scores': scores
            })
            
            # For demonstration, we'll stop if quality is good enough
            if scores['transformation_quality'] > 0.8:
                break
        
        return target_model, applied_rules, validation_scores
    
    def visualize_validation_scores(self, validation_scores):
        """Visualize validation scores over transformation steps"""
        steps = range(1, len(validation_scores) + 1)
        fvs = [score['forward_validation_score'] for score in validation_scores]
        bvs = [score['backward_validation_score'] for score in validation_scores]
        tq = [score['transformation_quality'] for score in validation_scores]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, fvs, 'b-o', label='Forward Validation')
        plt.plot(steps, bvs, 'g-o', label='Backward Validation')
        plt.plot(steps, tq, 'r-o', label='Transformation Quality')
        plt.xlabel('Transformation Step')
        plt.ylabel('Score')
        plt.title('Bidirectional Validation Scores')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        plt.show()

    
    # Add this method to the IntentAwareTransformer class in bidirectional_validator.py

    def transform_with_token_pairs(self, source_model, intent="translation",
                                max_rules=None, history_models=None, history_rules=None,
                                target_model=None):
        """
        Transform a source model using token pair aware bidirectional validation
        
        Args:
            source_model: Source model to transform
            intent: Transformation intent ("translation" or "revision")
            max_rules: Maximum number of rules to apply
            history_models: Optional list of history models for auto-regression
            history_rules: Optional list of history rules for auto-regression
            target_model: Optional target model for reference (if provided)
            
        Returns:
            Tuple of (transformed_model, applied_rules, validation_scores)
        """
        # Import needed components locally to avoid circular imports
        from token_pair_adapter import TokenPairAdapter
        from embedding_generator import EmbeddingGenerator
        
        adapter = TokenPairAdapter()
        embedding_generator = EmbeddingGenerator()
        
        # Generate model text representations
        source_text = source_model.to_text()
        
        # Generate model embeddings
        source_embedding = embedding_generator.generate_embedding(source_text)
        
        # Generate token pairs with embeddings
        source_token_pairs = adapter.convert_to_token_pairs_with_embeddings(
            source_model, source_text, source_embedding
        )
        
        # Create target model if not provided
        if target_model is None:
            # Create a new target model
            target_model = ModelGraph(f"{source_model.id}_transformed", "target")
        
        # Set maximum rules if not specified
        if max_rules is None:
            max_rules = len(self.rules_library)
        
        applied_rules = []
        validation_scores = []
        
        # First phase: Apply rules incrementally
        for i in range(max_rules):
            # Predict next rule based on current state and context
            next_rule = self.predict_next_rule(
                source_model, target_model, intent, history_models, history_rules
            )
            
            if next_rule is None:
                break
                
            # Apply the rule
            target_model = self.apply_rule(next_rule, source_model, target_model)
            applied_rules.append(next_rule)
            
            # Generate target model text and embedding
            target_text = target_model.to_text()
            target_embedding = embedding_generator.generate_embedding(target_text)
            
            # Generate token pairs for target model
            target_token_pairs = adapter.convert_to_token_pairs_with_embeddings(
                target_model, target_text, target_embedding
            )
            
            # Validate the transformation with token pairs
            scores = self.validator.compute_enhanced_transformation_quality_with_token_pairs(
                source_model, target_model, applied_rules, 
                source_token_pairs, target_token_pairs,
                source_embedding, target_embedding, intent
            )
            
            validation_scores.append(scores)
            
            # Record in transformation history
            self.transformation_history.append({
                'source_model': source_model.id,
                'target_model': target_model.id,
                'rule': next_rule.id,
                'intent': intent,
                'scores': scores
            })
            
            # Stop if quality is very good or no improvement
            if scores['enhanced_transformation_quality'] > 0.95 or (i > 0 and scores['improvement'] < 0.01):
                break
        
        return target_model, applied_rules, validation_scores


# ========================= EXAMPLE USAGE =========================

def create_example_models():
    """Create example models for demonstration"""
    
    # Create a source UML state machine model
    source_model = ModelGraph("StateMachine1", "source")
    
    # Add states
    source_model.add_node("State1", "State", {"name": "Idle", "isInitial": True})
    source_model.add_node("State2", "State", {"name": "Active"})
    source_model.add_node("State3", "State", {"name": "Error"})
    
    # Add transitions
    source_model.add_edge("State1", "State2", "Transition", {"trigger": "activate"})
    source_model.add_edge("State2", "State3", "Transition", {"trigger": "error"})
    source_model.add_edge("State3", "State1", "Transition", {"trigger": "reset"})
    
    return source_model

def create_example_rules():
    """Create example transformation rules"""
    
    # Rule for transforming State elements
    rule1 = TransformationRule(
        "StateToNode",
        "State",
        "Node",
        "translation",
        ["All States must be transformed to Nodes"]
    )
    
    # Rule for transforming Transition elements
    rule2 = TransformationRule(
        "TransitionToEdge",
        "Transition",
        "Edge",
        "translation",
        ["All Transitions must be transformed to Edges"]
    )
    
    # Rule for refining Node properties
    rule3 = TransformationRule(
        "RefineNodeProps",
        "Node",
        "Node",
        "revision",
        ["Nodes must have type and label attributes"]
    )
    
    return [rule1, rule2, rule3]

def run_example():
    """Run an example of the bidirectional validation framework"""
    
    print("Initializing Bidirectional Validation Framework...")
    
    # Initialize components
    encoder = ContextEncoder()
    validator = BidirectionalValidator(encoder)
    transformer = IntentAwareTransformer(encoder, validator)
    
    # Create example models and rules
    source_model = create_example_models()
    rules = create_example_rules()
    
    # Add rules to transformer
    for rule in rules:
        transformer.add_rule(rule)
    
    print("\nSource Model:")
    print(source_model.to_text())
    
    print("\nTransformation Rules:")
    for rule in rules:
        print(rule.to_text())
    
    # Transform with validation
    print("\nPerforming transformation with bidirectional validation...")
    target_model, applied_rules, validation_scores = transformer.transform_with_validation(
        source_model, intent="translation"
    )
    
    print("\nTarget Model:")
    print(target_model.to_text())
    
    print("\nApplied Rules:")
    for rule in applied_rules:
        print(f"- {rule.id}")
    
    print("\nValidation Scores:")
    for i, scores in enumerate(validation_scores):
        print(f"Step {i+1}:")
        print(f"  Forward Validation: {scores['forward_validation_score']:.4f}")
        print(f"  Backward Validation: {scores['backward_validation_score']:.4f}")
        print(f"  Transformation Quality: {scores['transformation_quality']:.4f}")
    
    # Visualize validation scores
    print("\nVisualizing validation scores...")
    transformer.visualize_validation_scores(validation_scores)
    
    print("\nComplete!")

# Add this to your BidirectionalValidator class

def compute_backward_validation_score_with_embeddings(self, source_model, target_model, 
                                                   source_embedding, target_embedding):
    """
    Compute how well the transformation preserves semantics using both token pairs and embeddings
    
    Args:
        source_model: Source model
        target_model: Target model
        source_embedding: Embedding of source model text
        target_embedding: Embedding of target model text
    
    Returns:
        Enhanced backward validation score
    """
    # Compute base score using token pairs
    base_score = self.compute_backward_validation_score(source_model, target_model)
    
    # Compute embedding similarity
    embedding_similarity = self._compute_embedding_similarity(source_embedding, target_embedding)
    
    # Blend scores with configurable weights
    alpha = 0.4  # Weight for token-pair based score
    enhanced_score = alpha * base_score + (1 - alpha) * embedding_similarity
    
    return enhanced_score

def _compute_embedding_similarity(self, embedding1, embedding2):
    """Compute similarity between embeddings"""
    # Compute cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    
    # Normalize to [0, 1]
    normalized_similarity = (similarity + 1) / 2
    
    return normalized_similarity


if __name__ == "__main__":
    run_example()