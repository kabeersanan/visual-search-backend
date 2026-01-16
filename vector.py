from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os
import pickle
from scipy.spatial.distance import cosine

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_image_embedding(image_path):
    """Generate vector embedding for a single image"""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Normalize the embedding
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()

def build_vector_database(image_folder):
    """Build vector database for all images in folder"""
    image_embeddings = {}
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(image_folder, filename)
            try:
                embedding = generate_image_embedding(image_path)
                image_embeddings[filename] = embedding
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Save to file
    with open('image_embeddings.pkl', 'wb') as f:
        pickle.dump(image_embeddings, f)
    
    return image_embeddings

def find_similar_images(uploaded_image_path, image_embeddings, top_k=5):
    """Find top-k similar images using cosine similarity"""
    # Generate embedding for uploaded image
    query_embedding = generate_image_embedding(uploaded_image_path)
    
    # Calculate similarity scores
    similarities = {}
    for filename, embedding in image_embeddings.items():
        # Using cosine similarity (1 - cosine_distance)
        similarity = 1 - cosine(query_embedding, embedding)
        similarities[filename] = similarity
    
    # Sort by similarity (highest first)
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results[:top_k]

# Main execution
if __name__ == "__main__":
    # Step 1: Build the database (run this once)
    print("Building vector database...")
    image_embeddings = build_vector_database('photos')
    
    # Step 2: Find similar images
    print("\nSearching for similar images...")
    similar_images = find_similar_images('uploaded_image.png', image_embeddings, top_k=10)
    
    print("\nTop 10 similar images:")
    for filename, score in similar_images:
        print(f"{filename}: {score:.4f}")