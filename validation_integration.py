"""
Integration module to connect multi-level embedding validation 
with the existing validation framework.
"""

from multi_level_embedding import MultiLevelEmbedding
import numpy as np

class EnhancedValidation:
    """
    Enhances the existing validation framework with multi-level embedding validation.
    """
    
    def __init__(self):
        self.embedding_engine = MultiLevelEmbedding()
        
    def extract_model_elements(self, model):
        """
        Extract individual elements from a model for element-level embedding.
        
        Parameters:
        - model: The model to extract elements from
        
        Returns:
        - Dictionary mapping element IDs to element text representations
        """
        elements = {}
        
        # Extract elements based on your model structure
        # This is a placeholder - you'll need to adapt this to your model representation
        for element_id, element in model.get_elements().items():
            element_text = element.get_text_representation()
            elements[element_id] = element_text
            
        return elements
    
    def extract_token_pairs(self, model):
        """
        Extract token pairs from a model.
        
        Parameters:
        - model: The model to extract token pairs from
        
        Returns:
        - Dictionary mapping token pair IDs to (element_id, meta_element_id) tuples
        """
        token_pairs = {}
        
        # Extract token pairs based on your model structure
        # This is a placeholder - you'll need to adapt this to your model representation
        for tp_id, tp in model.get_token_pairs().items():
            element_id = tp.get_element_id()
            meta_element_id = tp.get_meta_element_id()
            token_pairs[tp_id] = (element_id, meta_element_id)
            
        return token_pairs
    
    def generate_model_text(self, model):
        """
        Generate a textual representation of the entire model.
        
        Parameters:
        - model: The model to convert to text
        
        Returns:
        - String representation of the model
        """
        # Generate model text based on your model structure
        # This is a placeholder - you'll need to adapt this to your model representation
        return model.to_text()
    
    def prepare_embeddings(self, source_model, target_model):
        """
        Prepare embeddings for all levels (element, token pair, model) for both models.
        
        Parameters:
        - source_model: The source model
        - target_model: The target model
        
        Returns:
        - Dictionary with embedding information for both models
        """
        result = {
            "source": {
                "model_id": source_model.get_id(),
                "element_ids": [],
                "token_pair_ids": []
            },
            "target": {
                "model_id": target_model.get_id(),
                "element_ids": [],
                "token_pair_ids": []
            }
        }
        
        # Process source model
        source_elements = self.extract_model_elements(source_model)
        for element_id, element_text in source_elements.items():
            self.embedding_engine.generate_element_embedding(element_id, element_text)
            result["source"]["element_ids"].append(element_id)
        
        source_token_pairs = self.extract_token_pairs(source_model)
        for tp_id, (element_id, meta_element_id) in source_token_pairs.items():
            self.embedding_engine.generate_token_pair_embedding(tp_id, element_id, meta_element_id)
            result["source"]["token_pair_ids"].append(tp_id)
        
        source_text = self.generate_model_text(source_model)
        self.embedding_engine.generate_model_embedding(result["source"]["model_id"], source_text)
        
        # Process target model
        target_elements = self.extract_model_elements(target_model)
        for element_id, element_text in target_elements.items():
            self.embedding_engine.generate_element_embedding(element_id, element_text)
            result["target"]["element_ids"].append(element_id)
        
        target_token_pairs = self.extract_token_pairs(target_model)
        for tp_id, (element_id, meta_element_id) in target_token_pairs.items():
            self.embedding_engine.generate_token_pair_embedding(tp_id, element_id, meta_element_id)
            result["target"]["token_pair_ids"].append(tp_id)
        
        target_text = self.generate_model_text(target_model)
        self.embedding_engine.generate_model_embedding(result["target"]["model_id"], target_text)
        
        return result
    
    def compute_enhanced_backward_validation(self, source_model, target_model, intent="translation"):
        """
        Compute an enhanced backward validation score using multi-level embeddings.
        
        Parameters:
        - source_model: The source model
        - target_model: The target model
        - intent: Transformation intent ("translation" or "revision")
        
        Returns:
        - Enhanced backward validation score between 0 and 1
        """
        # Prepare all embeddings
        embedding_info = self.prepare_embeddings(source_model, target_model)
        
        # Compute multi-level similarity
        similarity_result = self.embedding_engine.compute_multi_level_similarity(
            embedding_info["source"]["model_id"],
            embedding_info["target"]["model_id"],
            embedding_info["source"]["element_ids"],
            embedding_info["target"]["element_ids"],
            embedding_info["source"]["token_pair_ids"],
            embedding_info["target"]["token_pair_ids"],
            intent
        )
        
        # Compute traditional backward validation score
        # This is a placeholder - call your existing backward validation method
        traditional_bvs = source_model.compute_backward_validation(target_model)
        
        # Combine traditional and embedding-based validation
        # You can adjust this weighting based on your experiments
        beta = 0.7  # Weight for traditional backward validation
        enhanced_bvs = beta * traditional_bvs + (1 - beta) * similarity_result["multi_level_similarity"]
        
        # Return full details for analysis
        return {
            "enhanced_backward_validation_score": enhanced_bvs,
            "traditional_backward_validation_score": traditional_bvs,
            "multi_level_similarity": similarity_result["multi_level_similarity"],
            "element_similarity": similarity_result["element_similarity"],
            "token_pair_similarity": similarity_result["token_pair_similarity"],
            "model_similarity": similarity_result["model_similarity"],
            "intent": intent
        }
    
    def compute_intent_aware_transformation_quality(self, source_model, target_model, 
                                                  forward_validation_score, intent="translation"):
        """
        Compute an intent-aware transformation quality score with enhanced backward validation.
        
        Parameters:
        - source_model: The source model
        - target_model: The target model
        - forward_validation_score: Result of forward validation
        - intent: Transformation intent ("translation" or "revision")
        
        Returns:
        - Intent-aware transformation quality score
        """
        # Compute enhanced backward validation
        backward_result = self.compute_enhanced_backward_validation(source_model, target_model, intent)
        
        # Set alpha based on intent
        if intent == "translation":
            alpha = 0.5  # Balanced for translation
        else:  # revision
            alpha = 0.75  # Favor forward validation for revision
        
        # Compute overall transformation quality
        transformation_quality = alpha * forward_validation_score + (1 - alpha) * backward_result["enhanced_backward_validation_score"]
        
        # Return full details for analysis
        return {
            "transformation_quality": transformation_quality,
            "forward_validation_score": forward_validation_score,
            "backward_validation": backward_result,
            "alpha": alpha,
            "intent": intent
        }