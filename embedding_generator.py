import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

class EmbeddingGenerator:
    """
    Enhanced embedding generator for model text representations using pre-trained language models
    """
    
    def __init__(self, model_name="distilbert-base-uncased"):
        """Initialize with a pre-trained model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()  # Set model to evaluation mode
            self.model_name = model_name
            print(f"Successfully initialized EmbeddingGenerator with model: {model_name}")
        except Exception as e:
            print(f"Failed to load model {model_name}: {str(e)}")
            print("Falling back to distilbert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModel.from_pretrained("distilbert-base-uncased")
            self.model.eval()
            self.model_name = "distilbert-base-uncased"
    
    def generate_embedding(self, text):
        """
        Generate embedding for a given text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        # Handle empty or None text
        if not text:
            # Return zero vector of appropriate size
            with torch.no_grad():
                # Get embedding size from model config
                embedding_size = self.model.config.hidden_size
                return np.zeros(embedding_size)
        
        # Tokenize text
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use CLS token embedding (first token) as text embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            return embedding[0]  # Return the embedding vector
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return zero vector
            embedding_size = self.model.config.hidden_size
            return np.zeros(embedding_size)
    
    def generate_token_pair_embedding(self, token_pair, model_text=None):
        """
        Generate embedding specific to a token pair
        
        Args:
            token_pair: Token pair dictionary
            model_text: Text representation of the model (optional)
            
        Returns:
            Embedding vector specific to this token pair
        """
        # Get element information
        element_type = token_pair['element']['type']
        element_id = token_pair['element']['id']
        
        # Use element-specific text if available
        if 'element_context' in token_pair and 'text' in token_pair['element_context']:
            element_text = token_pair['element_context']['text']
        else:
            # Create a text representation based on token pair properties
            element_text = f"Element of type {element_type} with id {element_id}."
        
        # Create token pair specific text description
        pair_text = f"Token pair: {element_text} "
        
        # Add meta-element information
        if 'meta_element' in token_pair:
            pair_text += f"Meta-element type: {token_pair['meta_element']['type']}. "
            
            # Add constraints
            constraints = token_pair['meta_element'].get('constraints', [])
            if constraints:
                pair_text += f"Constraints: {', '.join(constraints)}. "
        
        # Add relationship information if available
        if 'pair_context' in token_pair and 'relations' in token_pair['pair_context']:
            relations = token_pair['pair_context']['relations']
            if relations:
                pair_text += "Relationships: "
                for idx, rel in enumerate(relations[:3]):  # Limit to first 3 relationships
                    if rel['type'] == 'outgoing':
                        pair_text += f"connects to {rel['target_type']} via {rel['edge_type']}"
                    else:
                        pair_text += f"connected from {rel['source_type']} via {rel['edge_type']}"
                        
                    if idx < min(2, len(relations) - 1):
                        pair_text += ", "
        
        # Generate embedding for this specific token pair text
        return self.generate_embedding(pair_text)
    
    def generate_token_pair_embeddings(self, token_pairs):
        """
        Generate embeddings for a list of token pairs
        
        Args:
            token_pairs: List of token pair dictionaries
            
        Returns:
            Updated token pairs with embeddings
        """
        updated_pairs = []
        
        for pair in token_pairs:
            # Generate pair-specific embedding
            pair_embedding = self.generate_token_pair_embedding(pair)
            
            # Add the embedding to the token pair
            pair_copy = pair.copy()
            pair_copy['embedding'] = pair_embedding
            updated_pairs.append(pair_copy)
        
        return updated_pairs
    
        def compute_similarity(self, text1, text2):
            """
            Compute semantic similarity between two texts
            
            Args:
                text1: First text
                text2: Second text
                
            Returns:
                Similarity score between -1 and 1
            """
            embedding1 = self.generate_embedding(text1)
            embedding2 = self.generate_embedding(text2)
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(embedding1, embedding2)
            return similarity
    
    def _cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]
    
    def compute_token_pair_similarity(self, source_pairs, target_pairs):
        """
        Compute similarity between token pairs using their embeddings
        
        Args:
            source_pairs: List of token pairs from source model with embeddings
            target_pairs: List of token pairs from target model with embeddings
            
        Returns:
            Similarity score between 0 and 1
        """
        # First, ensure all pairs have embeddings
        for pairs in [source_pairs, target_pairs]:
            for pair in pairs:
                if 'embedding' not in pair:
                    pair['embedding'] = self.generate_token_pair_embedding(pair)
        
        # Compute similarity matrix between all pairs
        similarities = []
        for s_pair in source_pairs:
            for t_pair in target_pairs:
                sim = self._cosine_similarity(s_pair['embedding'], t_pair['embedding'])
                # Normalize to [0, 1]
                sim = (sim + 1) / 2
                similarities.append(sim)
        
        # Return average similarity
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0
    
    def analyze_text_embeddings(self, texts, labels=None, output_path=None):
        """
        Analyze a collection of text embeddings
        
        Args:
            texts: List of texts to analyze
            labels: Optional list of labels for the texts
            output_path: Optional path to save visualizations
            
        Returns:
            Dictionary with analysis results
        """
        if not texts:
            return {"error": "No texts provided"}
        
        if labels is None:
            labels = [f"Text {i+1}" for i in range(len(texts))]
        
        # Generate embeddings
        embeddings = np.array([self.generate_embedding(text) for text in texts])
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Calculate statistics
        mean_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        min_similarity = np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        max_similarity = np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        
        results = {
            "mean_similarity": float(mean_similarity),
            "min_similarity": float(min_similarity),
            "max_similarity": float(max_similarity),
            "similarity_matrix": similarity_matrix.tolist(),
            "num_texts": len(texts)
        }
        
        # Create visualization if output path provided
        if output_path:
            self._visualize_similarity_matrix(similarity_matrix, labels, output_path)
        
        return results
    
    def _visualize_similarity_matrix(self, similarity_matrix, labels, output_path):
        """
        Visualize similarity matrix
        
        Args:
            similarity_matrix: Matrix of similarity scores
            labels: Labels for the matrix
            output_path: Path to save visualization
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Cosine Similarity')
        
        # Add labels
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        
        plt.title(f'Semantic Similarity Matrix using {self.model_name}')
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path)
        plt.close()