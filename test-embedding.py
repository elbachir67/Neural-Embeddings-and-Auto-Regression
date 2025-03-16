#!/usr/bin/env python3
import os
import sys
import traceback

print("Testing embedding generator functionality")
print(f"Python version: {sys.version}")

try:
    print("Importing required packages...")
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    print("Successfully imported required packages")
    
    print("\nTesting tokenizer and model loading...")
    model_name = "distilbert-base-uncased"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully")
    
    model = AutoModel.from_pretrained(model_name)
    print("Model loaded successfully")
    model.eval()
    
    print("\nTesting text encoding...")
    test_text = "This is a test text for embedding generation."
    print(f"Test text: '{test_text}'")
    
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    print("Tokenization successful")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    
    with torch.no_grad():
        outputs = model(**inputs)
    print("Model inference successful")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    print("Embedding extraction successful")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First few values: {embedding[0, :5]}")
    
    # Test cosine similarity
    text1 = "UML Class with attributes"
    text2 = "Ecore EClass with EAttributes"
    
    print("\nTesting similarity computation...")
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
    embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()
    
    # Compute cosine similarity
    dot_product = np.dot(embedding1[0], embedding2[0])
    norm1 = np.linalg.norm(embedding1[0])
    norm2 = np.linalg.norm(embedding2[0])
    similarity = dot_product / (norm1 * norm2)
    
    print("Similarity computation successful")
    print(f"Cosine similarity: {similarity}")
    
    print("\nAll embedding tests passed successfully!")
    
except Exception as e:
    print(f"ERROR during testing: {str(e)}")
    traceback.print_exc()
    sys.exit(1)