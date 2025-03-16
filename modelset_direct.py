"""
Direct ModelSet integration with your bidirectional validation framework
using synthetic models that match your framework's structure
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bidirectional_validator import ModelGraph, TransformationRule, ContextEncoder, BidirectionalValidator, IntentAwareTransformer

class SimpleModelSetIntegration:
    """
    Simple integration that uses the ModelSet metadata but creates
    synthetic models directly compatible with your framework
    """
    
    def __init__(self, modelset_path):
        self.modelset_path = Path(modelset_path)
        self.index_file = self.modelset_path / "index.json"
        self.model_index = self._load_index()
        
    def _load_index(self):
        """Load the model index or create a synthetic one"""
        if not self.index_file.exists():
            print("Creating synthetic index...")
            return self._create_synthetic_index()
        
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading index: {e}")
            return self._create_synthetic_index()
    
    def _create_synthetic_index(self):
        """Create a synthetic model index for testing"""
        return [
            {
                "name": "State Machine Transformation",
                "transformations": [
                    {
                        "source": {
                            "id": "uml1",
                            "path": "modelset/uml1.uml",
                            "type": "UML"
                        },
                        "target": {
                            "id": "ecore1",
                            "path": "modelset/ecore1.ecore",
                            "type": "Ecore"
                        },
                        "type": "translation"
                    }
                ]
            },
            {
                "name": "Class Diagram Transformation",
                "transformations": [
                    {
                        "source": {
                            "id": "uml2",
                            "path": "modelset/uml2.uml",
                            "type": "UML"
                        },
                        "target": {
                            "id": "ecore2",
                            "path": "modelset/ecore2.ecore",
                            "type": "Ecore"
                        },
                        "type": "translation"
                    }
                ]
            },
            {
                "name": "State Machine Revision",
                "transformations": [
                    {
                        "source": {
                            "id": "uml3",
                            "path": "modelset/uml3.uml",
                            "type": "UML"
                        },
                        "target": {
                            "id": "uml3_v2",
                            "path": "modelset/uml3_v2.uml",
                            "type": "UML"
                        },
                        "type": "revision"
                    }
                ]
            }
        ]
    
    def create_synthetic_model(self, model_info):
        """Create a synthetic model based on model info"""
        model_id = model_info.get('id', 'unknown')
        model_type = model_info.get('type', 'unknown')
        
        model = ModelGraph(model_id, model_type)
        
        # Create different contents based on model type and id
        if model_type.lower() == 'uml':
            if 'stateMachine' in model_id.lower() or 'state' in model_id.lower():
                # State machine model
                self._create_state_machine_uml(model)
            else:
                # Class diagram model
                self._create_class_diagram_uml(model)
        elif model_type.lower() == 'ecore':
            if 'stateMachine' in model_id.lower() or 'state' in model_id.lower():
                # State machine model
                self._create_state_machine_ecore(model)
            else:
                # Class diagram model
                self._create_class_diagram_ecore(model)
        
        return model
    
    def _create_state_machine_uml(self, model):
        """Create a UML state machine model"""
        # Add states
        model.add_node("State1", "State", {"name": "Initial", "kind": "initial"})
        model.add_node("State2", "State", {"name": "Processing"})
        model.add_node("State3", "State", {"name": "Final", "kind": "final"})
        
        # Add package
        model.add_node("Package1", "Package", {"name": "StateMachine"})
        
        # Add transitions
        model.add_edge("State1", "State2", "Transition", {"trigger": "start"})
        model.add_edge("State2", "State3", "Transition", {"trigger": "complete"})
        
        # Link package to states
        model.add_edge("Package1", "State1", "packagedElement")
        model.add_edge("Package1", "State2", "packagedElement")
        model.add_edge("Package1", "State3", "packagedElement")
    
    def _create_class_diagram_uml(self, model):
        """Create a UML class diagram model"""
        # Add classes
        model.add_node("Class1", "Class", {"name": "Person"})
        model.add_node("Class2", "Class", {"name": "Address"})
        
        # Add attributes
        model.add_node("Attr1", "Property", {"name": "name", "type": "String"})
        model.add_node("Attr2", "Property", {"name": "age", "type": "Integer"})
        model.add_node("Attr3", "Property", {"name": "street", "type": "String"})
        model.add_node("Attr4", "Property", {"name": "city", "type": "String"})
        
        # Add associations
        model.add_node("Assoc1", "Association", {"name": "lives-at"})
        
        # Add package
        model.add_node("Package1", "Package", {"name": "PersonModel"})
        
        # Link everything together
        model.add_edge("Class1", "Attr1", "ownedAttribute")
        model.add_edge("Class1", "Attr2", "ownedAttribute")
        model.add_edge("Class2", "Attr3", "ownedAttribute")
        model.add_edge("Class2", "Attr4", "ownedAttribute")
        model.add_edge("Class1", "Assoc1", "sourceConnection")
        model.add_edge("Assoc1", "Class2", "targetConnection")
        model.add_edge("Package1", "Class1", "packagedElement")
        model.add_edge("Package1", "Class2", "packagedElement")
    
    def _create_state_machine_ecore(self, model):
        """Create an Ecore state machine model"""
        # Add EClasses
        model.add_node("EClass1", "EClass", {"name": "State"})
        model.add_node("EClass2", "EClass", {"name": "Transition"})
        model.add_node("EClass3", "EClass", {"name": "StateMachine"})
        
        # Add EAttributes
        model.add_node("EAttr1", "EAttribute", {"name": "name", "eType": "EString"})
        model.add_node("EAttr2", "EAttribute", {"name": "isInitial", "eType": "EBoolean"})
        model.add_node("EAttr3", "EAttribute", {"name": "isFinal", "eType": "EBoolean"})
        model.add_node("EAttr4", "EAttribute", {"name": "trigger", "eType": "EString"})
        
        # Add EReferences
        model.add_node("ERef1", "EReference", {"name": "states", "containment": "true"})
        model.add_node("ERef2", "EReference", {"name": "transitions", "containment": "true"})
        model.add_node("ERef3", "EReference", {"name": "source"})
        model.add_node("ERef4", "EReference", {"name": "target"})
        
        # Add EPackage
        model.add_node("EPkg1", "EPackage", {"name": "StateMachinePackage"})
        
        # Link everything together
        model.add_edge("EPkg1", "EClass1", "eClassifiers")
        model.add_edge("EPkg1", "EClass2", "eClassifiers")
        model.add_edge("EPkg1", "EClass3", "eClassifiers")
        model.add_edge("EClass1", "EAttr1", "eStructuralFeatures")
        model.add_edge("EClass1", "EAttr2", "eStructuralFeatures")
        model.add_edge("EClass1", "EAttr3", "eStructuralFeatures")
        model.add_edge("EClass2", "EAttr4", "eStructuralFeatures")
        model.add_edge("EClass2", "ERef3", "eStructuralFeatures")
        model.add_edge("EClass2", "ERef4", "eStructuralFeatures")
        model.add_edge("EClass3", "ERef1", "eStructuralFeatures")
        model.add_edge("EClass3", "ERef2", "eStructuralFeatures")
        model.add_edge("ERef3", "EClass1", "eType")
        model.add_edge("ERef4", "EClass1", "eType")
        model.add_edge("ERef1", "EClass1", "eType")
        model.add_edge("ERef2", "EClass2", "eType")
    
    def _create_class_diagram_ecore(self, model):
        """Create an Ecore class diagram model"""
        # Add EClasses
        model.add_node("EClass1", "EClass", {"name": "Person"})
        model.add_node("EClass2", "EClass", {"name": "Address"})
        
        # Add EAttributes
        model.add_node("EAttr1", "EAttribute", {"name": "name", "eType": "EString"})
        model.add_node("EAttr2", "EAttribute", {"name": "age", "eType": "EInt"})
        model.add_node("EAttr3", "EAttribute", {"name": "street", "eType": "EString"})
        model.add_node("EAttr4", "EAttribute", {"name": "city", "eType": "EString"})
        
        # Add EReferences
        model.add_node("ERef1", "EReference", {"name": "address"})
        
        # Add EPackage
        model.add_node("EPkg1", "EPackage", {"name": "PersonModel"})
        
        # Link everything together
        model.add_edge("EPkg1", "EClass1", "eClassifiers")
        model.add_edge("EPkg1", "EClass2", "eClassifiers")
        model.add_edge("EClass1", "EAttr1", "eStructuralFeatures")
        model.add_edge("EClass1", "EAttr2", "eStructuralFeatures")
        model.add_edge("EClass1", "ERef1", "eStructuralFeatures")
        model.add_edge("EClass2", "EAttr3", "eStructuralFeatures")
        model.add_edge("EClass2", "EAttr4", "eStructuralFeatures")
        model.add_edge("ERef1", "EClass2", "eType")
    
    def get_transformation_pair(self, pair_index=0, pair_type=None):
        """Get a source-target model pair for transformation"""
        # Find pairs based on type if specified
        if pair_type:
            pairs = []
            for entry in self.model_index:
                if 'transformations' in entry:
                    for transform in entry['transformations']:
                        if transform.get('type') == pair_type:
                            source = transform.get('source')
                            target = transform.get('target')
                            if source and target:
                                pairs.append({
                                    'name': entry.get('name', 'Unknown'),
                                    'source': source,
                                    'target': target,
                                    'type': transform.get('type')
                                })
            
            if pairs and pair_index < len(pairs):
                pair = pairs[pair_index]
            else:
                # Fallback to any pair
                pair = self._get_any_pair()
        else:
            # Get any pair at the specified index
            pairs = []
            for entry in self.model_index:
                if 'transformations' in entry:
                    for transform in entry['transformations']:
                        source = transform.get('source')
                        target = transform.get('target')
                        if source and target:
                            pairs.append({
                                'name': entry.get('name', 'Unknown'),
                                'source': source,
                                'target': target,
                                'type': transform.get('type')
                            })
            
            if pairs and pair_index < len(pairs):
                pair = pairs[pair_index]
            else:
                # Create a fallback pair
                pair = self._get_any_pair()
        
        # Create the models
        source_model = self.create_synthetic_model(pair['source'])
        target_model = self.create_synthetic_model(pair['target'])
        
        return {
            'name': pair['name'],
            'type': pair['type'],
            'source_model': source_model,
            'target_model': target_model
        }
    
    def _get_any_pair(self):
        """Get any transformation pair from the index or create one"""
        for entry in self.model_index:
            if 'transformations' in entry and entry['transformations']:
                transform = entry['transformations'][0]
                return {
                    'name': entry.get('name', 'Unknown'),
                    'source': transform.get('source'),
                    'target': transform.get('target'),
                    'type': transform.get('type')
                }
        
        # If no pairs found, create a synthetic one
        return {
            'name': 'Synthetic Transformation',
            'source': {'id': 'synthetic_uml', 'type': 'UML'},
            'target': {'id': 'synthetic_ecore', 'type': 'Ecore'},
            'type': 'translation'
        }
    
    def get_model_sequence(self, sequence_index=0):
        """Get a sequence of models for auto-regression experiments"""
        # Try to find a sequence in the index
        sequences = []
        
        for entry in self.model_index:
            if 'transformations' in entry and len(entry['transformations']) >= 2:
                # We can use transformations to form a sequence
                models = []
                for transform in entry['transformations']:
                    source = transform.get('source')
                    if source and source not in models:
                        models.append(source)
                    
                    target = transform.get('target')
                    if target and target not in models:
                        models.append(target)
                
                if len(models) >= 2:
                    sequences.append({
                        'name': entry.get('name', 'Unknown'),
                        'models': models
                    })
        
        # If no sequences found in index, create synthetic ones
        if not sequences:
            # Create a state machine evolution sequence
            sequences = [{
                'name': 'State Machine Evolution',
                'models': [
                    {'id': 'state_v1', 'type': 'UML'},
                    {'id': 'state_v2', 'type': 'UML'},
                    {'id': 'state_v3', 'type': 'UML'}
                ]
            }]
        
        # Get the requested sequence or the first one
        if sequence_index < len(sequences):
            sequence = sequences[sequence_index]
        else:
            sequence = sequences[0]
        
        # Create the models
        models = []
        for model_info in sequence['models']:
            model = self.create_synthetic_model(model_info)
            models.append(model)
        
        return {
            'name': sequence['name'],
            'models': models
        }