from bidirectional_validator import ModelGraph, TransformationRule
import numpy as np
import networkx as nx

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
            # Add model-level context
            token_pair['model_context'] = {
                'embedding': model_embedding,
                'model_type': model.type,
                'text_representation': model_text
            }
            
            # Add pair-specific context
            element_id = token_pair['element']['id']
            element_type = token_pair['element']['type']
            
            # Extract context for this specific token pair
            pair_context = self._extract_token_pair_context(model, element_id, element_type)
            token_pair['pair_context'] = pair_context
            
            # Extract text representation for this element
            element_text = self._extract_element_text(model_text, element_id, element_type)
            if element_text:
                token_pair['element_context'] = {
                    'text': element_text
                }
        
        return token_pairs
    
    def _extract_token_pair_context(self, model, element_id, element_type):
        """
        Extract token pair-specific context including relationships with other elements
        
        Args:
            model: ModelGraph instance
            element_id: ID of the element
            element_type: Type of the element
            
        Returns:
            Context dictionary with relationship information
        """
        context = {
            'relations': [],
            'neighborhood': [],
            'constraints': [],
            'position': {}
        }
        
        # Extract relationships (incoming and outgoing edges)
        if element_id in model.graph:
            # Get outgoing edges
            for target in model.graph.successors(element_id):
                edge_data = model.graph.get_edge_data(element_id, target)
                target_type = model.graph.nodes[target].get('type', 'Unknown')
                edge_type = edge_data.get('type', 'Unknown') if edge_data else 'Unknown'
                
                context['relations'].append({
                    'type': 'outgoing',
                    'target_id': target,
                    'target_type': target_type,
                    'edge_type': edge_type
                })
            
            # Get incoming edges
            for source in model.graph.predecessors(element_id):
                edge_data = model.graph.get_edge_data(source, element_id)
                source_type = model.graph.nodes[source].get('type', 'Unknown')
                edge_type = edge_data.get('type', 'Unknown') if edge_data else 'Unknown'
                
                context['relations'].append({
                    'type': 'incoming',
                    'source_id': source,
                    'source_type': source_type,
                    'edge_type': edge_type
                })
        
        # Get neighborhood (all connected nodes)
        if element_id in model.graph:
            try:
                neighbors = list(nx.all_neighbors(model.graph, element_id))
                for neighbor in neighbors:
                    neighbor_type = model.graph.nodes[neighbor].get('type', 'Unknown')
                    context['neighborhood'].append({
                        'id': neighbor,
                        'type': neighbor_type
                    })
            except:
                # Fallback if all_neighbors fails
                neighbors = list(model.graph.neighbors(element_id))
                for neighbor in neighbors:
                    neighbor_type = model.graph.nodes[neighbor].get('type', 'Unknown')
                    context['neighborhood'].append({
                        'id': neighbor,
                        'type': neighbor_type
                    })
        
        # Add structural position information
        if element_id in model.graph:
            total_nodes = max(1, len(model.graph.nodes))
            try:
                neighbors_count = len(list(model.graph.neighbors(element_id)))
                in_degree = model.graph.in_degree(element_id)
                out_degree = model.graph.out_degree(element_id)
                
                context['position'] = {
                    'degree': in_degree + out_degree,
                    'in_degree': in_degree,
                    'out_degree': out_degree,
                    'centrality': neighbors_count / total_nodes
                }
            except:
                # Fallback if degree calculations fail
                context['position'] = {
                    'degree': 0,
                    'in_degree': 0,
                    'out_degree': 0,
                    'centrality': 0
                }
        
        # Add constraints from meta-element if available
        context['constraints'] = self._infer_constraints(element_type, model.type)
        
        return context
    
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
    
    def align_token_pairs(self, source_pairs, target_pairs):
        """
        Find alignments between source and target token pairs
        
        Args:
            source_pairs: List of token pairs from source model
            target_pairs: List of token pairs from target model
            
        Returns:
            List of (source_pair, target_pair, similarity) tuples
        """
        alignments = []
        
        for source_pair in source_pairs:
            best_match = None
            best_similarity = 0
            
            for target_pair in target_pairs:
                # Compare element types
                element_similarity = self._calculate_element_similarity(
                    source_pair['element'], target_pair['element'])
                
                # Compare meta-element types
                meta_similarity = self._calculate_meta_similarity(
                    source_pair['meta_element'], target_pair['meta_element'])
                
                # Compare contexts if available
                context_similarity = 0
                if ('pair_context' in source_pair and 'pair_context' in target_pair):
                    context_similarity = self._calculate_context_similarity(
                        source_pair['pair_context'], target_pair['pair_context'])
                    
                # Weighted combination
                similarity = (0.4 * element_similarity + 
                             0.3 * meta_similarity + 
                             0.3 * context_similarity)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = target_pair
            
            if best_match and best_similarity > 0.5:  # Threshold for valid alignment
                alignments.append((source_pair, best_match, best_similarity))
        
        return alignments
    
    def _calculate_context_similarity(self, context1, context2):
        """
        Calculate similarity between two context dictionaries
        
        Args:
            context1: First context dictionary
            context2: Second context dictionary
            
        Returns:
            Similarity score between 0 and 1
        """
        if not context1 or not context2:
            return 0.0
        
        similarities = []
        
        # Compare relations
        if 'relations' in context1 and 'relations' in context2:
            rel_sim = self._compare_relations(context1['relations'], context2['relations'])
            similarities.append(rel_sim)
        
        # Compare neighborhood
        if 'neighborhood' in context1 and 'neighborhood' in context2:
            neigh_sim = self._compare_neighborhoods(context1['neighborhood'], context2['neighborhood'])
            similarities.append(neigh_sim)
        
        # Compare position metrics
        if 'position' in context1 and 'position' in context2:
            pos_sim = self._compare_positions(context1['position'], context2['position'])
            similarities.append(pos_sim)
        
        # Return average similarity
        return sum(similarities) / max(1, len(similarities))
    
    def _compare_relations(self, relations1, relations2):
        """Compare relation sets between two contexts"""
        if not relations1 or not relations2:
            return 0.0
        
        # Extract relation types
        types1 = [(r.get('edge_type', ''), r.get('target_type', '')) for r in relations1 
                  if r.get('type') == 'outgoing']
        types2 = [(r.get('edge_type', ''), r.get('target_type', '')) for r in relations2 
                  if r.get('type') == 'outgoing']
        
        # Count common relation types
        common = set(types1).intersection(set(types2))
        total = max(1, len(set(types1).union(set(types2))))
        
        return len(common) / total
    
    def _compare_neighborhoods(self, neigh1, neigh2):
        """Compare neighborhoods between two contexts"""
        if not neigh1 or not neigh2:
            return 0.0
        
        # Extract neighbor types
        types1 = [n.get('type', '') for n in neigh1]
        types2 = [n.get('type', '') for n in neigh2]
        
        # Count common types
        common_count = 0
        for type1 in types1:
            for type2 in types2:
                if type1 == type2 or type1 in type2 or type2 in type1:
                    common_count += 1
                    break
        
        return common_count / max(1, len(types1))
    
    def _compare_positions(self, pos1, pos2):
        """Compare structural positions between two contexts"""
        if not pos1 or not pos2:
            return 0.0
        
        # Compare degree centrality
        degree_diff = abs(pos1.get('degree', 0) - pos2.get('degree', 0))
        max_degree = max(pos1.get('degree', 1), pos2.get('degree', 1), 1)
        degree_sim = 1 - (degree_diff / max_degree)
        
        # Compare centrality
        centrality_diff = abs(pos1.get('centrality', 0) - pos2.get('centrality', 0))
        centrality_sim = 1 - min(1, centrality_diff)
        
        return (degree_sim + centrality_sim) / 2
    
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
    
    def create_transformation_rules_with_alignments(self, source_model, target_model, source_text, target_text, intent=None):
        """
        Create transformation rules based on token pair alignments
        
        Args:
            source_model: Source ModelGraph
            target_model: Target ModelGraph
            source_text: Text representation of source model
            target_text: Text representation of target model
            intent: Transformation intent
            
        Returns:
            List of TransformationRule objects
        """
        # If intent is not specified, determine it based on model types
        if intent is None:
            intent = "translation" if source_model.type != target_model.type else "revision"
        
        # Import necessary components locally to avoid circular imports
        from embedding_generator import EmbeddingGenerator
        
        # Generate embeddings
        embedding_generator = EmbeddingGenerator()
        source_embedding = embedding_generator.generate_embedding(source_text)
        target_embedding = embedding_generator.generate_embedding(target_text)
        
        # Generate token pairs with context
        source_pairs = self.convert_to_token_pairs_with_embeddings(
            source_model, source_text, source_embedding)
        target_pairs = self.convert_to_token_pairs_with_embeddings(
            target_model, target_text, target_embedding)
        
        # Find alignments between token pairs
        alignments = self.align_token_pairs(source_pairs, target_pairs)
        
        # Create rules based on alignments
        rules = []
        for source_pair, target_pair, similarity in alignments:
            if similarity > 0.7:  # Only use high-confidence alignments
                source_type = source_pair['element']['type']
                target_type = target_pair['element']['type']
                
                rule_id = f"{source_type}To{target_type}"
                constraints = [
                    f"Transform {source_type} to {target_type} with confidence {similarity:.2f}"
                ]
                
                # Add relationships and context to constraints
                if 'pair_context' in source_pair and 'relations' in source_pair['pair_context']:
                    for relation in source_pair['pair_context']['relations']:
                        if relation['type'] == 'outgoing':
                            constraints.append(
                                f"Preserve relationship with {relation['target_type']} via {relation['edge_type']}")
                
                # Create rule with enhanced constraints
                from bidirectional_validator import TransformationRule
                rule = TransformationRule(
                    rule_id,
                    source_type,
                    target_type,
                    intent,
                    constraints
                )
                
                # Check if this rule already exists to avoid duplicates
                if not any(r.id == rule_id for r in rules):
                    rules.append(rule)
        
        # If no alignments found, fall back to traditional rule creation
        if not rules:
            rules = self.create_transformation_rules(source_model, target_model, intent)
        
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
                
                # Calculate context similarity if available
                context_similarity = 0.0
                if 'pair_context' in source_pair and 'pair_context' in target_pair:
                    context_similarity = self._calculate_context_similarity(
                        source_pair['pair_context'], target_pair['pair_context'])
                
                # Weighted combination
                pair_similarity = 0.5 * element_similarity + 0.3 * meta_similarity + 0.2 * context_similarity
                
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