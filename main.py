from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
import uuid
from datetime import datetime
from pathlib import Path
import pickle
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from scipy.spatial.distance import cosine

app = FastAPI()

# Configure CORS
# Allow requests from Adobe Express Add-on frontend and local development
origins = [
    "https://localhost:5241",  # Adobe Express Add-on frontend
    "https://127.0.0.1:5241",  # Alternative frontend address
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load CLIP model for image embeddings
print("Loading CLIP model...")
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()  # Set to evaluation mode
    print("CLIP model loaded successfully")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    clip_model = None
    clip_processor = None

# Load pre-computed image embeddings
EMBEDDINGS_FILE = "image_embeddings.pkl"
image_embeddings = {}

def load_embeddings():
    """Load pre-computed image embeddings from pickle file"""
    global image_embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                image_embeddings = pickle.load(f)
            print(f"Loaded {len(image_embeddings)} image embeddings from {EMBEDDINGS_FILE}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            image_embeddings = {}
    else:
        print(f"Embeddings file {EMBEDDINGS_FILE} not found. Please build the vector database first.")
        image_embeddings = {}

# Load embeddings on startup
load_embeddings()

def generate_image_embedding(image_path: str) -> np.ndarray:
    """Generate vector embedding for a single image using CLIP"""
    if clip_model is None or clip_processor is None:
        raise HTTPException(status_code=500, detail="CLIP model not loaded")
    
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = clip_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        
        # Normalize the embedding
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

def find_similar_images(uploaded_image_path: str, top_k: int = 10) -> List[Dict]:
    """Find top-k similar images using cosine similarity"""
    if not image_embeddings:
        return []
    
    try:
        # Generate embedding for uploaded image
        query_embedding = generate_image_embedding(uploaded_image_path)
        
        # Calculate similarity scores
        similarities = []
        for filename, embedding in image_embeddings.items():
            # Using cosine similarity (1 - cosine_distance)
            similarity_score = 1 - cosine(query_embedding, embedding)
            similarities.append({
                "filename": filename,
                "similarity": float(similarity_score),
                "filepath": str(UPLOAD_DIR / filename) if (UPLOAD_DIR / filename).exists() else None
            })
        
        # Sort by similarity (highest first) and return top_k
        sorted_results = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
        return sorted_results[:top_k]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar images: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI with venv!"}

@app.post("/search")
async def search(
    file: UploadFile = File(...), 
    query: Optional[str] = Form(None),
    top_k: int = Form(10)
):
    """
    Visual search endpoint that accepts an image file and returns similar images.
    
    The frontend should send:
    - 'file': A Blob or File object containing the image to search for
    - 'query': Optional search query string (for future use)
    - 'top_k': Number of similar images to return (default: 10)
    """
    saved_file_path = None
    try:
        # Read the uploaded file
        contents = await file.read()
        file_type = file.content_type
        
        # Generate a unique filename if original filename is not provided
        if file.filename:
            # Extract file extension from original filename
            original_name = Path(file.filename)
            file_extension = original_name.suffix or ".png"
            # Create filename with timestamp and UUID to ensure uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}{file_extension}"
        else:
            # Default to PNG if no filename provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.png"
        
        # Save the file
        saved_file_path = UPLOAD_DIR / filename
        with open(saved_file_path, "wb") as f:
            f.write(contents)
        
        # Find similar images using vector search
        similar_images = find_similar_images(str(saved_file_path), top_k=top_k)
        
        return {
            "message": "Search completed successfully",
            "query": query or "",
            "query_image": {
                "original_filename": file.filename,
                "saved_filename": filename,
                "saved_path": str(saved_file_path),
                "content_type": file_type,
                "size": len(contents)
            },
            "similar_images": similar_images,
            "total_results": len(similar_images),
            "uploaded_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file if saving failed
        if saved_file_path and saved_file_path.exists():
            try:
                saved_file_path.unlink()
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}