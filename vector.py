from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os
import pickle
from scipy.spatial.distance import cosine
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# Load environment variables for Cloudinary
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.getenv("CLOUDINARY_API_KEY", ""),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", ""),
    secure=True
)

# Load CLIP model
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded successfully")

def generate_image_embedding(image_path):
    """Generate vector embedding for a single image"""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Normalize the embedding
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()

def build_vector_database(image_folder, upload_to_cloudinary=True):
    """
    Build vector database for all images in folder.
    Also uploads images to Cloudinary and saves URLs.
    
    Args:
        image_folder: Path to folder containing images
        upload_to_cloudinary: If True, upload images to Cloudinary and save URLs
    """
    image_embeddings = {}
    cloudinary_enabled = upload_to_cloudinary and os.getenv("CLOUDINARY_CLOUD_NAME")
    
    if not cloudinary_enabled and upload_to_cloudinary:
        print("Warning: Cloudinary not configured. Images will not be uploaded.")
        print("Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET in .env file")
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(image_folder, filename)
            try:
                # Generate embedding
                embedding = generate_image_embedding(image_path)
                
                # Upload to Cloudinary if enabled
                cloudinary_url = None
                if cloudinary_enabled:
                    try:
                        upload_result = cloudinary.uploader.upload(
                            image_path,
                            folder="brandkit",
                            public_id=filename.replace(os.path.splitext(filename)[1], ""),  # Remove extension
                            resource_type="image"
                        )
                        cloudinary_url = upload_result.get("secure_url") or upload_result.get("url")
                        print(f"  ✓ Uploaded to Cloudinary: {cloudinary_url}")
                    except Exception as upload_error:
                        print(f"  ⚠ Failed to upload {filename} to Cloudinary: {upload_error}")
                
                # Store with new format: {embedding, cloudinary_url}
                image_embeddings[filename] = {
                    "embedding": embedding,
                    "cloudinary_url": cloudinary_url
                }
                print(f"✓ Processed: {filename}")
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
    
    # Save to file
    with open('image_embeddings.pkl', 'wb') as f:
        pickle.dump(image_embeddings, f)
    
    print(f"\n✓ Saved {len(image_embeddings)} embeddings to image_embeddings.pkl")
    return image_embeddings

def find_similar_images(uploaded_image_path, image_embeddings, top_k=5):
    """Find top-k similar images using cosine similarity"""
    # Generate embedding for uploaded image
    query_embedding = generate_image_embedding(uploaded_image_path)
    
    # Calculate similarity scores
    similarities = {}
    for filename, data in image_embeddings.items():
        # Handle both old format (direct embedding) and new format (dict)
        if isinstance(data, dict):
            embedding = data.get("embedding", data)
            # Convert list to numpy array if needed
            if isinstance(embedding, list):
                embedding = np.array(embedding)
        else:
            embedding = data
        
        # Ensure embedding is numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
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