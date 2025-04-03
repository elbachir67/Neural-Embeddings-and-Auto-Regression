"""
Multi-level embedding validation for model transformations.
This module implements element-level, token pair-level, and model-level 
embedding generation and similarity computation.
"""

import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Initialize DistilBERT components
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

class MultiLevelEmbedding:
    """
    Manages embeddings at multiple levels of granularity:
    - Element level: Individual model elements
    - Token pair level: Relationships between elements and meta-elements
    - Model level: Complete model representations
    """
    
    def __init__(self):
        self.element_embeddings = {}
        self.token_pair_embeddings = {}
        self.model_embeddings = {}
    
    def generate_element_embedding(self, element_id, element_text):
        """
        Generate embedding for an individual model element.
        
        Parameters:
        - element_id: Unique identifier for the element
        - element_text: Textual representation of the element
        
        Returns:
        - Embedding vector for the element
        """
        inputs = tokenizer(element_text, return_tensors="pt", 
                          truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the [CLS] token embedding as the element representation
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        self.element_embeddings[element_id] = embedding
        return embedding
    
    def generate_token_pair_embedding(self, token_pair_id, element_id, meta_element_id):
        """
        Generate embedding for a token pair connecting an element to its meta-element.
        
        Parameters:
        - token_pair_id: Unique identifier for the token pair
        - element_id: ID of the element in the token pair
        - meta_element_id: ID of the meta-element in the token pair
        
        Returns:
        - Embedding vector for the token pair
        """
        # If we already have embeddings for both elements, combine them
        if element_id in self.element_embeddings and meta_element_id in self.element_embeddings:
            element_emb = self.element_embeddings[element_id]
            meta_element_emb = self.element_embeddings[meta_element_id]
            
            # Combine the embeddings (concatenate and project)
            combined = np.concatenate([element_emb, meta_element_emb], axis=1)
            # Simple projection: average the combined vector to original size
            token_pair_emb = combined.reshape(element_emb.shape)
            
            self.token_pair_embeddings[token_pair_id] = token_pair_emb
            return token_pair_emb
        else:
            raise ValueError("Element or meta-element embedding not found")
    
    def generate_model_embedding(self, model_id, model_text):
        """
        Generate embedding for an entire model.
        
        Parameters:
        - model_id: Unique identifier for the model
        - model_text: Textual representation of the model
        
        Returns:
        - Embedding vector for the model
        """
        inputs = tokenizer(model_text, return_tensors="pt", 
                          truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the [CLS] token embedding as the model representation
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        self.model_embeddings[model_id] = embedding
        return embedding
    
    def compute_element_similarity(self, element_id1, element_id2):
        """
        Compute similarity between two elements based on their embeddings.
        
        Parameters:
        - element_id1: ID of the first element
        - element_id2: ID of the second element
        
        Returns:
        - Similarity score between 0 and 1
        """
        if element_id1 in self.element_embeddings and element_id2 in self.element_embeddings:
            emb1 = self.element_embeddings[element_id1]
            emb2 = self.element_embeddings[element_id2]
            similarity = cosine_similarity(emb1, emb2)[0][0]
            # Normalize from [-1, 1] to [0, 1] range
            normalized_similarity = (similarity + 1) / 2
            return normalized_similarity
        else:
            raise ValueError("Element embedding not found")
    
    def compute_token_pair_similarity(self, token_pair_id1, token_pair_id2):
        """
        Compute similarity between two token pairs based on their embeddings.
        
        Parameters:
        - token_pair_id1: ID of the first token pair
        - token_pair_id2: ID of the second token pair
        
        Returns:
        - Similarity score between 0 and 1
        """
        if token_pair_id1 in self.token_pair_embeddings and token_pair_id2 in self.token_pair_embeddings:
            emb1 = self.token_pair_embeddings[token_pair_id1]
            emb2 = self.token_pair_embeddings[token_pair_id2]
            similarity = cosine_similarity(emb1, emb2)[0][0]
            # Normalize from [-1, 1] to [0, 1] range
            normalized_similarity = (similarity + 1) / 2
            return normalized_similarity
        else:
            raise ValueError("Token pair embedding not found")
    
    def compute_model_similarity(self, model_id1, model_id2):
        """
        Compute similarity between two models based on their embeddings.
        
        Parameters:
        - model_id1: ID of the first model
        - model_id2: ID of the second model
        
        Returns:
        - Similarity score between 0 and 1
        """
        if model_id1 in self.model_embeddings and model_id2 in self.model_embeddings:
            emb1 = self.model_embeddings[model_id1]
            emb2 = self.model_embeddings[model_id2]
            similarity = cosine_similarity(emb1, emb2)[0][0]
            # Normalize from [-1, 1] to [0, 1] range
            normalized_similarity = (similarity + 1) / 2
            return normalized_similarity
        else:
            raise ValueError("Model embedding not found")
    
    def compute_multi_level_similarity(self, source_model_id, target_model_id, 
                                      source_elements, target_elements,
                                      source_token_pairs, target_token_pairs,
                                      intent="translation"):
        """
        Compute a weighted similarity score across all embedding levels.
        
        Parameters:
        - source_model_id: ID of source model
        - target_model_id: ID of target model
        - source_elements: List of element IDs from source model
        - target_elements: List of element IDs from target model
        - source_token_pairs: List of token pair IDs from source model
        - target_token_pairs: List of token pair IDs from target model
        - intent: Transformation intent ("translation" or "revision")
        
        Returns:
        - Multi-level similarity score between 0 and 1
        """
        # Compute model-level similarity
        model_sim = self.compute_model_similarity(source_model_id, target_model_id)
        
        # Compute average element-level similarity
        element_sims = []
        for src_elem in source_elements:
            # Find best matching target element
            best_sim = 0
            for tgt_elem in target_elements:
                try:
                    sim = self.compute_element_similarity(src_elem, tgt_elem)
                    if sim > best_sim:
                        best_sim = sim
                except ValueError:
                    continue
            if best_sim > 0:
                element_sims.append(best_sim)
        
        element_sim = np.mean(element_sims) if element_sims else 0
        
        # Compute average token pair similarity
        token_pair_sims = []
        for src_tp in source_token_pairs:
            # Find best matching target token pair
            best_sim = 0
            for tgt_tp in target_token_pairs:
                try:
                    sim = self.compute_token_pair_similarity(src_tp, tgt_tp)
                    if sim > best_sim:
                        best_sim = sim
                except ValueError:
                    continue
            if best_sim > 0:
                token_pair_sims.append(best_sim)
        
        token_pair_sim = np.mean(token_pair_sims) if token_pair_sims else 0
        
        # Apply intent-specific weighting
        if intent == "translation":
            # For translation, prioritize token pairs and model-level similarity
            weights = {"element": 0.2, "token_pair": 0.5, "model": 0.3}
        else:  # revision
            # For revision, prioritize element-level and model-level similarity
            weights = {"element": 0.4, "token_pair": 0.2, "model": 0.4}
        
        # Compute weighted multi-level similarity
        multi_level_sim = (
            weights["element"] * element_sim +
            weights["token_pair"] * token_pair_sim +
            weights["model"] * model_sim
        )
        
        return {
            "multi_level_similarity": multi_level_sim,
            "element_similarity": element_sim,
            "token_pair_similarity": token_pair_sim,
            "model_similarity": model_sim
        }