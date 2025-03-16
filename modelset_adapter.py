from bidirectional_validator import ModelGraph, TransformationRule, ContextEncoder, BidirectionalValidator, IntentAwareTransformer

class ModelSetAdapter:
    """
    Adapter to connect the ModelSet loader with your token pair bidirectional validation framework
    """
    
    def __init__(self, transformer):
        """
        Initialize the adapter with your IntentAwareTransformer
        
        Args:
            transformer: Your IntentAwareTransformer instance
        """
        self.transformer = transformer
    
    def convert_to_token_pairs(self, model):
        """
        Convert a ModelGraph into token pairs
        
        Args:
            model: ModelGraph instance
            
        Returns:
            List of token pairs (element, meta-element)
        """
        token_pairs = []
        
        print(f"Converting model {model.id} of type {model.type}")
        print(f"Model has {len(model.graph.nodes)} nodes and {len(model.graph.edges)} edges")
        
        # Process each node in the model
        for node_id, node_data in model.graph.nodes(data=True):
            print(f"Processing node: {node_id}, data: {node_data}")
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
            for _, target, edge_data in model.graph.out_edges(node_id, data=True):
                edge_type = edge_data.get('type', 'Unknown')
                edge_attrs = edge_data.get('attrs', {})
                
                # Create token pair for the relationship
                edge_token_pair = {
                    'element': {
                        'id': f"{node_id}_{target}_{edge_type}",
                        'type': edge_type,
                        'source': node_id,
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
    
    def extract_transformation_rules(self, source_model, target_model):
        """
        Extract transformation rules by analyzing source and target models
        
        Args:
            source_model: Source ModelGraph
            target_model: Target ModelGraph
            
        Returns:
            List of TransformationRule objects
        """
        rules = []
        
        # Determine the intent based on model types
        intent = "translation" if source_model.type != target_model.type else "revision"
        
        # Analyze node types in both models
        source_node_types = {data['type'] for _, data in source_model.graph.nodes(data=True)}
        target_node_types = {data['type'] for _, data in target_model.graph.nodes(data=True)}
        
        # Create rules for node type transformations
        for s_type in source_node_types:
            # Try to find a corresponding type in the target model
            # This is a simplified approach - in reality, you would use more sophisticated matching
            for t_type in target_node_types:
                if (s_type in t_type) or (t_type in s_type) or self._are_semantically_related(s_type, t_type):
                    rule_id = f"{s_type}To{t_type}"
                    rules.append(TransformationRule(
                        rule_id,
                        s_type,
                        t_type,
                        intent,
                        [f"{source_model.type} {s_type} must be transformed to {target_model.type} {t_type}"]
                    ))
        
        return rules
    
    def _are_semantically_related(self, type1, type2):
        """Determine if two types are semantically related"""
        # This is a simplified implementation
        # In a real system, you would use your LLM or more sophisticated analysis
        
        # Common semantic relationships
        semantic_pairs = [
            ('Class', 'EClass'),
            ('Property', 'EAttribute'),
            ('Association', 'EReference'),
            ('Package', 'EPackage'),
            ('State', 'EClass'),
            ('Transition', 'EReference')
        ]
        
        return (type1, type2) in semantic_pairs or (type2, type1) in semantic_pairs
    
    def transform_and_validate(self, source_model, target_model=None, intent=None):
        """
        Transform and validate using your framework
        
        Args:
            source_model: Source ModelGraph
            target_model: Target ModelGraph (optional, for reference)
            intent: Transformation intent (translation or revision)
            
        Returns:
            Transformation results
        """
        # Convert models to token pairs
        source_token_pairs = self.convert_to_token_pairs(source_model)
        expected_target_token_pairs = self.convert_to_token_pairs(target_model) if target_model else None
        
        # Extract transformation rules
        if target_model:
            rules = self.extract_transformation_rules(source_model, target_model)
        else:
            # Create default rules based on model type
            rules = self._create_default_rules(source_model, intent)
        
        # Add rules to the transformer
        for rule in rules:
            self.transformer.add_rule(rule)
        
        # Perform transformation with validation
        transformed_model, applied_rules, validation_scores = self.transformer.transform_with_validation(
            source_model, 
            intent=intent or "translation",
            max_rules=len(rules)
        )
        
        # Return the results
        result = {
            'transformed_model': transformed_model,
            'applied_rules': [rule.id for rule in applied_rules],
            'forward_validation': validation_scores[-1]['forward_validation_score'] if validation_scores else 0,
            'backward_validation': validation_scores[-1]['backward_validation_score'] if validation_scores else 0,
            'transformation_quality': validation_scores[-1]['transformation_quality'] if validation_scores else 0
        }
        
        return result
    
    def _create_default_rules(self, source_model, intent):
        """Create default transformation rules based on the source model type"""
        rules = []
        
        # Set the target type based on source type and intent
        if source_model.type.lower() == 'uml' and intent == 'translation':
            target_type = 'ecore'
            
            # Common UML to Ecore transformation rules
            rules.append(TransformationRule(
                "ClassToEClass",
                "Class",
                "EClass",
                "translation",
                ["UML Classes must be transformed to Ecore EClasses"]
            ))
            
            rules.append(TransformationRule(
                "PropertyToEAttribute",
                "Property",
                "EAttribute",
                "translation",
                ["UML Properties must be transformed to Ecore EAttributes"]
            ))
            
            rules.append(TransformationRule(
                "AssociationToEReference",
                "Association",
                "EReference",
                "translation",
                ["UML Associations must be transformed to Ecore EReferences"]
            ))
            
            rules.append(TransformationRule(
                "PackageToEPackage",
                "Package",
                "EPackage",
                "translation",
                ["UML Packages must be transformed to Ecore EPackages"]
            ))
            
            # State machine specific rules
            rules.append(TransformationRule(
                "StateToEClass",
                "State",
                "EClass",
                "translation",
                ["UML States must be transformed to Ecore EClasses"]
            ))
            
            rules.append(TransformationRule(
                "TransitionToEReference",
                "Transition",
                "EReference",
                "translation",
                ["UML Transitions must be transformed to Ecore EReferences"]
            ))
            
        elif intent == 'revision':
            # Rules for revising models within the same metamodel
            if source_model.type.lower() == 'uml':
                rules.append(TransformationRule(
                    "AddGuardCondition",
                    "Transition",
                    "Transition",
                    "revision",
                    ["Add guard condition to transitions"]
                ))
                
                rules.append(TransformationRule(
                    "AddEventHandler",
                    "State",
                    "State",
                    "revision",
                    ["Add event handlers to states"]
                ))
                
            elif source_model.type.lower() == 'ecore':
                rules.append(TransformationRule(
                    "AddEOperation",
                    "EClass",
                    "EClass",
                    "revision",
                    ["Add operations to EClasses"]
                ))
                
                rules.append(TransformationRule(
                    "MakeEReferencesComposition",
                    "EReference",
                    "EReference",
                    "revision",
                    ["Make EReferences composition relationships"]
                ))
        
        return rules