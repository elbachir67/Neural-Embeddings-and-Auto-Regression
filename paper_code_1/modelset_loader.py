import os
import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from bidirectional_validator import ModelGraph, TransformationRule

class ModelSetLoader:
    """Loads models from the ModelSet dataset for the token pair framework"""
    
    def __init__(self, modelset_path):
        self.modelset_path = Path(modelset_path)
        self.data_path = self.modelset_path / 'data'
        self.txt_path = self.modelset_path / 'txt'
        self.uml_categories = None
        self.ecore_categories = None
        self.transformation_pairs = []
        
        # Load metadata
        self.load_metadata()
    
    def load_metadata(self):
        """Load the category metadata for UML and Ecore models"""
        uml_categories_path = self.data_path / 'categories_uml.csv'
        ecore_categories_path = self.data_path / 'categories_ecore.csv'
        
        if uml_categories_path.exists():
            self.uml_categories = pd.read_csv(uml_categories_path)
            print(f"Loaded UML categories with {len(self.uml_categories)} entries")
        else:
            print(f"Warning: UML categories file not found at {uml_categories_path}")
        
        if ecore_categories_path.exists():
            self.ecore_categories = pd.read_csv(ecore_categories_path)
            print(f"Loaded Ecore categories with {len(self.ecore_categories)} entries")
        else:
            print(f"Warning: Ecore categories file not found at {ecore_categories_path}")
        
        # Identify transformation pairs
        self.identify_transformation_pairs()
    
    def identify_transformation_pairs(self):
        """Identify potential transformation pairs based on matching categories"""
        if self.uml_categories is None or self.ecore_categories is None:
            print("Cannot identify transformation pairs without categories data")
            return
        
        # Filter out models that aren't in English (if language column exists)
        if 'language' in self.uml_categories.columns:
            uml_categories = self.uml_categories[self.uml_categories['language'] == 'english']
        else:
            uml_categories = self.uml_categories
        
        if 'language' in self.ecore_categories.columns:
            ecore_categories = self.ecore_categories[self.ecore_categories['language'] == 'english']
        else:
            ecore_categories = self.ecore_categories
        
        # Filter out domains with too few models
        uml_counts = uml_categories.groupby(['domain'], as_index=False).count()
        ecore_counts = ecore_categories.groupby(['domain'], as_index=False).count()
        
        uml_domains = list(uml_counts[uml_counts['id'] >= 7]['domain'])
        ecore_domains = list(ecore_counts[ecore_counts['id'] >= 7]['domain'])
        
        # Remove 'unknown' and 'dummy' if present
        for domain_list in [uml_domains, ecore_domains]:
            try:
                domain_list.remove('unknown')
            except:
                pass
            try:
                domain_list.remove('dummy')
            except:
                pass
        
        # Find common domains
        common_domains = set(uml_domains).intersection(set(ecore_domains))
        
        # Create transformation pairs for common domains
        for domain in common_domains:
            uml_models = uml_categories[uml_categories['domain'] == domain]
            ecore_models = ecore_categories[ecore_categories['domain'] == domain]
            
            # Create pairs (limit to first 5 models of each domain)
            for _, uml_row in uml_models.head(5).iterrows():
                for _, ecore_row in ecore_models.head(5).iterrows():
                    self.transformation_pairs.append({
                        'name': f"{domain} Translation",
                        'source': {
                            'id': uml_row['id'],
                            'domain': domain,
                            'type': 'UML'
                        },
                        'target': {
                            'id': ecore_row['id'],
                            'domain': domain,
                            'type': 'Ecore'
                        },
                        'type': 'translation'
                    })
        
        # Also identify revision pairs (same type, same domain)
        for categories, model_type in [(uml_categories, 'UML'), (ecore_categories, 'Ecore')]:
            for domain in categories['domain'].unique():
                models = categories[categories['domain'] == domain]
                if len(models) >= 2:
                    model_ids = models['id'].tolist()
                    for i in range(len(model_ids) - 1):
                        self.transformation_pairs.append({
                            'name': f"{domain} Revision",
                            'source': {
                                'id': model_ids[i],
                                'domain': domain,
                                'type': model_type
                            },
                            'target': {
                                'id': model_ids[i+1],
                                'domain': domain,
                                'type': model_type
                            },
                            'type': 'revision'
                        })
        
        print(f"Identified {len(self.transformation_pairs)} potential transformation pairs")
    
    def load_model(self, model_info):
        """Load a model from ModelSet and convert it to a ModelGraph"""
        model_id = model_info['id']
        model_type = model_info['type']
        
        # Create a ModelGraph for the model
        model_graph = ModelGraph(model_id, model_type.lower())
        
        # Load the model text representation from txt folder
        model_dir = self.txt_path / model_id
        if model_dir.exists():
            # Find the text file for this model
            text_files = list(model_dir.glob('*.txt'))
            if text_files:
                model_path = text_files[0]
                
                # Load the model content
                with open(model_path, 'r') as f:
                    model_text = f.read()
                
                # Create a model graph from the content
                self._populate_model_graph(model_graph, model_text, model_type)
                
                return model_graph
            else:
                print(f"No text file found for model {model_id}")
        else:
            print(f"Directory not found for model {model_id}")
        
        # If we couldn't load the model, create a synthetic one
        print(f"Creating synthetic {model_type} model for {model_id}")
        return self._create_synthetic_model(model_id, model_type, model_info.get('domain', 'unknown'))
    
    def load_model_with_text(self, model_info):
        """
        Load a model and its text representation
        
        Args:
            model_info: Dictionary with model information
            
        Returns:
            Tuple of (ModelGraph, model_text) or (ModelGraph, None) if text not found
        """
        model_id = model_info['id']
        model_type = model_info['type']
        
        # Create a ModelGraph for the model
        model_graph = ModelGraph(model_id, model_type.lower())
        
        # Try to load the model text representation from txt folder
        model_text = None
        model_dir = self.txt_path / model_id
        if model_dir.exists():
            # Find the text file for this model
            text_files = list(model_dir.glob('*.txt'))
            if text_files:
                model_path = text_files[0]
                
                # Load the model content
                try:
                    with open(model_path, 'r', encoding='utf-8') as f:
                        model_text = f.read()
                    
                    # Create a model graph from the text content
                    self._populate_model_graph_from_text(model_graph, model_text)
                    return model_graph, model_text
                except Exception as e:
                    print(f"Error reading text file for model {model_id}: {e}")
        
        # If text file not found or couldn't be read, fall back to synthetic model
        print(f"Text representation not found for model {model_id}")
        
        # Create a synthetic model
        synthetic_model = self._create_synthetic_model(model_id, model_type, model_info.get('domain', 'unknown'))
        
        # Generate synthetic text representation
        synthetic_text = self._generate_synthetic_text(synthetic_model)
        
        return synthetic_model, synthetic_text
    
    def _populate_model_graph(self, model_graph, model_text, model_type):
        """Populate a ModelGraph from model text content"""
        # Split the text into lines
        lines = model_text.strip().split('\n')
        
        # Process each line to extract information
        current_node = None
        node_id_counter = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line defines a new element
            if line.endswith(':') or ':' in line and not line.startswith(' '):
                # New element - create a node
                element_type = line.split(':')[0].strip()
                node_id = f"node_{node_id_counter}"
                node_id_counter += 1
                
                # Extract name if present
                name = line.split(':')[1].strip() if ':' in line else "unnamed"
                
                # Add the node
                model_graph.add_node(node_id, element_type, {"name": name})
                current_node = node_id
            elif current_node and line.startswith(' '):
                # This is an attribute or reference from the current node
                if '->' in line or 'reference' in line.lower() or 'connection' in line.lower():
                    # This is likely a reference/edge
                    parts = line.split('->')
                    target_name = parts[1].strip() if len(parts) > 1 else line.split(':')[1].strip()
                    edge_type = "reference"
                    
                    # Create a target node if it doesn't seem to exist
                    target_node = f"target_{node_id_counter}"
                    node_id_counter += 1
                    model_graph.add_node(target_node, "Referenced", {"name": target_name})
                    
                    # Add the edge
                    model_graph.add_edge(current_node, target_node, edge_type)
                else:
                    # This is likely an attribute - we'll ignore for simplicity
                    pass
        
        # If the model is too simple, add some default elements based on type
        if len(model_graph.graph.nodes) < 3:
            self._add_default_elements(model_graph, model_type)
    
    def _populate_model_graph_from_text(self, model_graph, model_text):
        """Populate a ModelGraph from text content"""
        # Split the text into lines
        lines = model_text.strip().split('\n')
        
        # Process lines to extract model elements
        current_element = None
        node_id_counter = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for elements by checking for patterns like "Element: " or "Element {"
            if ': ' in line or ' {' in line:
                # Likely an element definition
                element_type = line.split(':')[0].strip() if ': ' in line else line.split(' {')[0].strip()
                element_name = line.split(': ')[1].strip() if ': ' in line else ""
                
                # Create a node ID
                node_id = f"node_{node_id_counter}"
                node_id_counter += 1
                
                # Add node to graph
                model_graph.add_node(node_id, element_type, {"name": element_name})
                current_element = node_id
            
            # Look for relationships or properties
            elif current_element and '->' in line:
                # This seems to be a relationship
                parts = line.split('->')
                target_name = parts[1].strip()
                relationship_type = parts[0].strip()
                
                # Create target element if it doesn't exist
                target_node = None
                for node, data in model_graph.graph.nodes(data=True):
                    if data.get('attrs', {}).get('name') == target_name:
                        target_node = node
                        break
                
                if not target_node:
                    target_node = f"node_{node_id_counter}"
                    node_id_counter += 1
                    model_graph.add_node(target_node, "Referenced", {"name": target_name})
                
                # Add the edge
                model_graph.add_edge(current_element, target_node, relationship_type)
        
        # If the model is too simple, add some default elements
        if len(model_graph.graph.nodes) < 3:
            self._add_default_elements(model_graph, model_graph.type)
    
    def _add_default_elements(self, model_graph, model_type):
        """Add default elements to a model based on its type"""
        if model_type.lower() == 'uml':
            # Add some common UML elements
            model_graph.add_node("class1", "Class", {"name": "MainClass"})
            model_graph.add_node("class2", "Class", {"name": "RelatedClass"})
            model_graph.add_node("attr1", "Attribute", {"name": "attribute1"})
            model_graph.add_node("method1", "Operation", {"name": "operation1"})
            
            model_graph.add_edge("class1", "attr1", "hasAttribute")
            model_graph.add_edge("class1", "method1", "hasOperation")
            model_graph.add_edge("class1", "class2", "associates")
            
        elif model_type.lower() == 'ecore':
            # Add some common Ecore elements
            model_graph.add_node("eclass1", "EClass", {"name": "MainEClass"})
            model_graph.add_node("eclass2", "EClass", {"name": "RelatedEClass"})
            model_graph.add_node("eattr1", "EAttribute", {"name": "attribute1"})
            model_graph.add_node("eref1", "EReference", {"name": "reference1"})
            
            model_graph.add_edge("eclass1", "eattr1", "hasAttribute")
            model_graph.add_edge("eclass1", "eref1", "hasReference")
            model_graph.add_edge("eref1", "eclass2", "references")
    
    def _create_synthetic_model(self, model_id, model_type, domain):
        """Create a synthetic model based on type and domain"""
        model_graph = ModelGraph(model_id, model_type.lower())
        
        if model_type.lower() == 'uml':
            if domain == 'statemachine':
                # Create a state machine model
                model_graph.add_node("state1", "State", {"name": "Initial", "kind": "initial"})
                model_graph.add_node("state2", "State", {"name": "Processing"})
                model_graph.add_node("state3", "State", {"name": "Final", "kind": "final"})
                model_graph.add_node("trans1", "Transition", {"trigger": "start"})
                model_graph.add_node("trans2", "Transition", {"trigger": "complete"})
                
                model_graph.add_edge("state1", "trans1", "outgoing")
                model_graph.add_edge("trans1", "state2", "target")
                model_graph.add_edge("state2", "trans2", "outgoing")
                model_graph.add_edge("trans2", "state3", "target")
                
            elif domain == 'class':
                # Create a class diagram model
                model_graph.add_node("class1", "Class", {"name": "Person"})
                model_graph.add_node("class2", "Class", {"name": "Address"})
                model_graph.add_node("attr1", "Property", {"name": "name", "type": "String"})
                model_graph.add_node("attr2", "Property", {"name": "age", "type": "Integer"})
                model_graph.add_node("assoc1", "Association", {"name": "lives-at"})
                
                model_graph.add_edge("class1", "attr1", "ownedAttribute")
                model_graph.add_edge("class1", "attr2", "ownedAttribute")
                model_graph.add_edge("class1", "assoc1", "sourceConnection")
                model_graph.add_edge("assoc1", "class2", "targetConnection")
            else:
                # Generic UML model
                self._add_default_elements(model_graph, model_type)
        
        elif model_type.lower() == 'ecore':
            if domain == 'statemachine':
                # Create a state machine Ecore model
                model_graph.add_node("eclass1", "EClass", {"name": "State"})
                model_graph.add_node("eclass2", "EClass", {"name": "Transition"})
                model_graph.add_node("eclass3", "EClass", {"name": "StateMachine"})
                model_graph.add_node("eattr1", "EAttribute", {"name": "name", "eType": "EString"})
                model_graph.add_node("eattr2", "EAttribute", {"name": "isInitial", "eType": "EBoolean"})
                model_graph.add_node("eref1", "EReference", {"name": "source"})
                model_graph.add_node("eref2", "EReference", {"name": "target"})
                
                model_graph.add_edge("eclass1", "eattr1", "eStructuralFeatures")
                model_graph.add_edge("eclass1", "eattr2", "eStructuralFeatures")
                model_graph.add_edge("eclass2", "eref1", "eStructuralFeatures")
                model_graph.add_edge("eclass2", "eref2", "eStructuralFeatures")
                model_graph.add_edge("eref1", "eclass1", "eType")
                model_graph.add_edge("eref2", "eclass1", "eType")
            
            elif domain == 'class':
                # Create a class diagram Ecore model
                model_graph.add_node("eclass1", "EClass", {"name": "Person"})
                model_graph.add_node("eclass2", "EClass", {"name": "Address"})
                model_graph.add_node("eattr1", "EAttribute", {"name": "name", "eType": "EString"})
                model_graph.add_node("eattr2", "EAttribute", {"name": "age", "eType": "EInt"})
                model_graph.add_node("eref1", "EReference", {"name": "address"})
                
                model_graph.add_edge("eclass1", "eattr1", "eStructuralFeatures")
                model_graph.add_edge("eclass1", "eattr2", "eStructuralFeatures")
                model_graph.add_edge("eclass1", "eref1", "eStructuralFeatures")
                model_graph.add_edge("eref1", "eclass2", "eType")
            else:
                # Generic Ecore model
                self._add_default_elements(model_graph, model_type)
        
        return model_graph
    
    def get_transformation_pairs(self, type_filter=None, limit=5):
        """Get transformation pairs, optionally filtering by type"""
        if type_filter:
            pairs = [p for p in self.transformation_pairs if p['type'] == type_filter]
        else:
            pairs = self.transformation_pairs
        
        return pairs[:limit]
    
    def get_model_sequence(self, domain=None, model_type=None, limit=3):
        """Get a sequence of models for auto-regression experiments"""
        # Find all models for the specified domain and type
        if model_type.lower() == 'uml':
            categories = self.uml_categories
        elif model_type.lower() == 'ecore':
            categories = self.ecore_categories
        else:
            print(f"Unknown model type: {model_type}")
            return []
        
        if domain:
            models = categories[categories['domain'] == domain]
        else:
            # Get the domain with the most models
            counts = categories.groupby(['domain']).count()
            if not counts.empty:
                domain = counts.nlargest(1, 'id').index[0]
                models = categories[categories['domain'] == domain]
            else:
                return []
        
        # Get a sequence of models
        model_ids = models['id'].tolist()[:limit]
        
        # Load each model in the sequence
        model_sequence = []
        for model_id in model_ids:
            model_info = {'id': model_id, 'type': model_type, 'domain': domain}
            model = self.load_model(model_info)
            if model:
                model_sequence.append(model)
        
        return model_sequence
    
    def _generate_synthetic_text(self, model):
        """
        Generate synthetic text representation for a model
        """
        text = f"Model: {model.id} (Type: {model.type})\n\n"
        
        # Add elements
        for node_id, node_data in model.graph.nodes(data=True):
            node_type = node_data.get('type', 'Unknown')
            node_attrs = node_data.get('attrs', {})
            name = node_attrs.get('name', 'unnamed')
            
            text += f"{node_type}: {name}\n"
            
            # Add attributes
            for attr_name, attr_value in node_attrs.items():
                if attr_name != 'name':
                    text += f"  {attr_name}: {attr_value}\n"
        
        # Add relationships
        text += "\nRelationships:\n"
        for source, target, edge_data in model.graph.edges(data=True):
            edge_type = edge_data.get('type', 'relates to')
            source_name = model.graph.nodes[source].get('attrs', {}).get('name', source)
            target_name = model.graph.nodes[target].get('attrs', {}).get('name', target)
            
            text += f"{source_name} -> {target_name} ({edge_type})\n"
        
        return text