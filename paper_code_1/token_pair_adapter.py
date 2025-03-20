from bidirectional_validator import ModelGraph, TransformationRule
import numpy as np

class TokenPairAdapter:
    """Adapter for converting models to token pairs with embedding support"""
    
    def __init__(self):
        pass
    
    def convert_to_token_pairs(self, model):
        """
        Convert a ModelGraph to token pairs
        
        Args:
            model: ModelGraph instance
            
        Returns:
            List of token pairs (element, meta-element)
        """
        token_pairs = []
        
        # Process each node in the model
        for node_id, node_data in model.graph.nodes(data=True):
            node_type = node_data.get('type', 'Unknown')
            node_attrs = node_data.get('attrs', {})
            
            # Create the token pair (element, meta-element)
            token_pair = {
                'element': {
                    'id': node_id,
                    'type': node_type,
                    **node_attrs
                },
                'meta_element': {
                    'type': f"{model.type}:{node_type}",
                    'constraints': self._infer_constraints(node_type, model.type)
                }
            }
            
            token_pairs.append(token_pair)
        
        # Process edges for relationships
        for source, target, edge_data in model.graph.edges(data=True):
            edge_type = edge_data.get('type', 'Unknown')
            edge_attrs = edge_data.get('attrs', {})
            
            # Create token pair for the relationship
            edge_token_pair = {
                'element': {
                    'id': f"{source}_{target}_{edge_type}",
                    'type': edge_type,
                    'source': source,
                    'target': target,
                    **edge_attrs
                },
                'meta_element': {
                    'type': f"{model.type}:{edge_type}",
                    'constraints': self._infer_constraints(edge_type, model.type)
                }
            }
            
            token_pairs.append(edge_token_pair)
        
        return token_pairs
    
    def convert_to_token_pairs_with_embeddings(self, model, model_text, model_embedding):
        """
        Convert a ModelGraph to token pairs with model embedding context
        
        Args:
            model: ModelGraph instance
            model_text: Text representation of the model
            model_embedding: Embedding vector for the model
            
        Returns:
            List of token pairs with embedding context
        """
        # Get basic token pairs
        token_pairs = self.convert_to_token_pairs(model)
        
        # Add model embedding to each token pair
        for token_pair in token_pairs:
            token_pair['model_context'] = {
                'embedding': model_embedding,
                'model_type': model.type,
                'text_representation': model_text
            }
            
            # Add element-specific embeddings if possible
            element_id = token_pair['element']['id']
            element_type = token_pair['element']['type']
            # Extract text specific to this element from the full model text
            element_text = self._extract_element_text(model_text, element_id, element_type)
            
            if element_text:
                token_pair['element_context'] = {
                    'text': element_text
                }
        
        return token_pairs
    
    def _extract_element_text(self, model_text, element_id, element_type):
        """
        Extract text representation of a specific model element from the full model text
        This is a simplified implementation - in practice, you'd need more sophisticated parsing
        
        Args:
            model_text: Full text representation of the model
            element_id: ID of the element to extract
            element_type: Type of the element to extract
            
        Returns:
            Text representation of the element or None if not found
        """
        lines = model_text.split('\n')
        element_lines = []
        capturing = False
        element_identifier = f"{element_type}:"
        
        for line in lines:
            if element_identifier in line and not capturing:
                capturing = True
                element_lines.append(line)
            elif capturing and line.strip() and not line.startswith(' '):
                # Stop capturing when we hit the next non-indented line
                break
            elif capturing:
                element_lines.append(line)
        
        return '\n'.join(element_lines) if element_lines else None
    
    def _infer_constraints(self, element_type, model_type):
        """Infer constraints for the meta-element based on element type"""
        constraints = []
        
        # Add basic constraints based on element type
        if element_type == 'Class' or element_type == 'EClass':
            constraints.append('must_have_name')
        elif element_type == 'State':
            constraints.append('must_have_name')
        elif element_type == 'Transition' or element_type == 'EReference':
            constraints.append('must_have_source_and_target')
        
        return constraints
    
    def create_transformation_rules(self, source_model, target_model, intent=None):
        """Create transformation rules based on the models"""
        rules = []
        
        # Determine intent if not specified
        if intent is None:
            intent = "translation" if source_model.type != target_model.type else "revision"
        
        # Get node types from both models
        source_node_types = {data['type'] for _, data in source_model.graph.nodes(data=True)}
        target_node_types = {data['type'] for _, data in target_model.graph.nodes(data=True)}
        
        # Create rules for node transformations
        for s_type in source_node_types:
            for t_type in target_node_types:
                if self._is_potential_match(s_type, t_type, source_model.type, target_model.type):
                    rule_id = f"{s_type}To{t_type}"
                    rule = TransformationRule(
                        rule_id,
                        s_type,
                        t_type,
                        intent,
                        [f"Transform {source_model.type} {s_type} to {target_model.type} {t_type}"]
                    )
                    rules.append(rule)
        
        # If no rules were created, add some default ones
        if not rules:
            if source_model.type.lower() == 'uml' and target_model.type.lower() == 'ecore':
                rules.append(TransformationRule(
                    "ClassToEClass",
                    "Class",
                    "EClass",
                    intent,
                    ["UML Classes must be transformed to Ecore EClasses"]
                ))
                
                rules.append(TransformationRule(
                    "PropertyToEAttribute",
                    "Property",
                    "EAttribute",
                    intent,
                    ["UML Properties must be transformed to Ecore EAttributes"]
                ))
                
                rules.append(TransformationRule(
                    "AssociationToEReference",
                    "Association",
                    "EReference",
                    intent,
                    ["UML Associations must be transformed to Ecore EReferences"]
                ))
            elif intent == "revision":
                # Generic revision rules
                for node_type in source_node_types:
                    rule_id = f"Revise{node_type}"
                    rules.append(TransformationRule(
                        rule_id,
                        node_type,
                        node_type,
                        "revision",
                        [f"Revise {node_type} properties"]
                    ))
        
        return rules
    
    def _is_potential_match(self, source_type, target_type, source_model_type, target_model_type):
        """Determine if two types are potential matches for transformation"""
        # Common mappings between UML and Ecore
        mappings = [
            ('Class', 'EClass'),
            ('Property', 'EAttribute'),
            ('Association', 'EReference'),
            ('Package', 'EPackage'),
            ('Operation', 'EOperation'),
            ('Parameter', 'EParameter'),
            ('State', 'EClass'),
            ('Transition', 'EReference')
        ]
        
        # Check if source_type and target_type form a known mapping
        if source_model_type.lower() == 'uml' and target_model_type.lower() == 'ecore':
            return (source_type, target_type) in mappings
        
        # For revision, types should match
        if source_model_type == target_model_type:
            return source_type == target_type
        
        # Default case - allow if they share common substrings
        return source_type in target_type or target_type in source_type
    
    def compute_token_pair_similarity(self, source_pairs, target_pairs):
        """
        Compute similarity between two sets of token pairs
        
        Args:
            source_pairs: List of token pairs from source model
            target_pairs: List of token pairs from target model
            
        Returns:
            Similarity score between 0 and 1
        """
        # Count matched pairs
        matched_pairs = 0
        total_pairs = max(len(source_pairs), len(target_pairs))
        
        if total_pairs == 0:
            return 1.0  # Both are empty, consider them identical
        
        # For each source pair, find the best matching target pair
        for source_pair in source_pairs:
            source_element = source_pair['element']
            source_meta = source_pair['meta_element']
            
            best_match_score = 0
            
            for target_pair in target_pairs:
                target_element = target_pair['element']
                target_meta = target_pair['meta_element']
                
                # Calculate element similarity
                element_similarity = self._calculate_element_similarity(source_element, target_element)
                
                # Calculate meta-element similarity
                meta_similarity = self._calculate_meta_similarity(source_meta, target_meta)
                
                # Weighted combination
                pair_similarity = 0.7 * element_similarity + 0.3 * meta_similarity
                
                # Update best match
                if pair_similarity > best_match_score:
                    best_match_score = pair_similarity
            
            # Add proportional contribution to matched_pairs
            matched_pairs += best_match_score
        
        return matched_pairs / total_pairs
    
    def _calculate_element_similarity(self, source_element, target_element):
        """Calculate similarity between two model elements"""
        # Type similarity
        if source_element['type'] == target_element['type']:
            type_similarity = 1.0
        elif source_element['type'] in target_element['type'] or target_element['type'] in source_element['type']:
            type_similarity = 0.5
        else:
            type_similarity = 0.0
        
        # Name similarity if available
        name_similarity = 0.0
        if 'name' in source_element and 'name' in target_element:
            if source_element['name'] == target_element['name']:
                name_similarity = 1.0
            elif source_element['name'] in target_element['name'] or target_element['name'] in source_element['name']:
                name_similarity = 0.5
        
        # Weighted combination
        return 0.6 * type_similarity + 0.4 * name_similarity
    
    def _calculate_meta_similarity(self, source_meta, target_meta):
        """Calculate similarity between two meta-elements"""
        # Type similarity
        if source_meta['type'] == target_meta['type']:
            type_similarity = 1.0
        elif source_meta['type'].split(':')[1] == target_meta['type'].split(':')[1]:
            type_similarity = 0.8
        elif source_meta['type'] in target_meta['type'] or target_meta['type'] in source_meta['type']:
            type_similarity = 0.4
        else:
            type_similarity = 0.0
        
        # Constraint similarity
        constraint_similarity = self._calculate_constraint_similarity(source_meta['constraints'], target_meta['constraints'])
        
        # Weighted combination
        return 0.7 * type_similarity + 0.3 * constraint_similarity
    
    def _calculate_constraint_similarity(self, source_constraints, target_constraints):
        """Calculate similarity between two sets of constraints"""
        if not source_constraints and not target_constraints:
            return 1.0  # Both empty, consider them identical
        
        if not source_constraints or not target_constraints:
            return 0.0  # One is empty, the other isn't
        
        # Count constraints in both sets
        common = set(source_constraints).intersection(set(target_constraints))
        return len(common) / max(len(source_constraints), len(target_constraints))