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
    "https://127.0.0.1:5241","*"  # Alternative frontend address
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:5241",
        "https://127.0.0.1:5241",
        "*",
    ],
    allow_credentials=False,  # IMPORTANT
    allow_methods=["*"],
    allow_headers=["*"],
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
        
        # Prepare response
        response = {
            "message": "Search completed successfully",
            "query": query or "",
            "query_image": {
                "original_filename": file.filename,
                "saved_filename": filename,
                "content_type": file_type,
                "size": len(contents)
            },
            "similar_images": similar_images,
            "total_results": len(similar_images),
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Delete the uploaded file after processing
        if saved_file_path and saved_file_path.exists():
            try:
                saved_file_path.unlink()
                print(f"Deleted uploaded file: {saved_file_path}")
            except Exception as delete_error:
                print(f"Warning: Could not delete uploaded file {saved_file_path}: {delete_error}")
        
        return response
    except HTTPException:
        # Clean up file on error
        if saved_file_path and saved_file_path.exists():
            try:
                saved_file_path.unlink()
            except:
                pass
        raise
    except Exception as e:
        # Clean up file if processing failed
        if saved_file_path and saved_file_path.exists():
            try:
                saved_file_path.unlink()
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")


@app.post("/store")
async def store_brandkit_image(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Store brandkit image endpoint.
    Users can upload their personal brand kit images which will be saved to the photos folder.
    
    The frontend should send:
    - 'file': A Blob or File object containing the image to store (via FormData)
    - 'metadata': Optional JSON string with metadata (e.g., {"category": "logo", "brand": "Nike"})
    
    File size limit: 10MB
    Supported formats: PNG, JPG, JPEG, WEBP
    
    Example frontend usage:
        const formData = new FormData();
        formData.append('file', fileObject);
        fetch('/store', { method: 'POST', body: formData });
    """
    saved_file_path = None
    try:
        # Read the uploaded file
        contents = await file.read()
        file_type = file.content_type or ""
        
        # File size validation (10MB limit)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size is 10MB, received {len(contents) / 1024 / 1024:.2f}MB"
            )
        
        # Validate file type
        allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp']
        if file_type:
            file_type_lower = file_type.lower()
            if file_type_lower not in allowed_types:
                # Try to detect from filename extension as fallback
                if file.filename:
                    ext = Path(file.filename).suffix.lower()
                    if ext not in ['.png', '.jpg', '.jpeg', '.webp']:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Invalid file type. Allowed: PNG, JPG, JPEG, WEBP. Received: {file_type}"
                        )
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid file type. Allowed: PNG, JPG, JPEG, WEBP. Received: {file_type}"
                    )
        elif file.filename:
            # No content-type but has filename - check extension
            ext = Path(file.filename).suffix.lower()
            if ext not in ['.png', '.jpg', '.jpeg', '.webp']:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type. Allowed: PNG, JPG, JPEG, WEBP"
                )
        else:
            raise HTTPException(
                status_code=400, 
                detail="File must be an image. No content-type or filename provided."
            )
        
        # Generate filename (keep original if provided, or generate one)
        if file.filename:
            # Sanitize filename - remove any path components for security
            original_name = Path(file.filename).name
            # Keep original extension
            file_extension = Path(original_name).suffix or ".png"
            # Create a safe filename (optional: add timestamp for uniqueness)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            # Preserve original name but add timestamp to avoid conflicts
            base_name = Path(original_name).stem
            filename = f"{base_name}_{timestamp}_{unique_id}{file_extension}"
        else:
            # Default filename if not provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"brandkit_{timestamp}_{unique_id}.png"
        
        # Save to photos folder
        PHOTOS_DIR = Path("photos")
        PHOTOS_DIR.mkdir(exist_ok=True)
        
        saved_file_path = PHOTOS_DIR / filename
        with open(saved_file_path, "wb") as f:
            f.write(contents)
        
        # Generate embedding for the new image
        try:
            embedding = generate_image_embedding(str(saved_file_path))
            # Add to in-memory embeddings
            image_embeddings[filename] = embedding
            # Save updated embeddings to pickle file
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(image_embeddings, f)
            print(f"Generated embedding for {filename} and updated {EMBEDDINGS_FILE}")
        except Exception as embed_error:
            print(f"Warning: Failed to generate embedding for {filename}: {embed_error}")
            # Still return success - image is saved even if embedding fails
            # The embedding can be regenerated later by rebuilding the database
        
        # Parse additional metadata if provided
        image_metadata = {
            "filename": filename,
            "original_filename": file.filename or filename,
            "filepath": str(saved_file_path),
            "content_type": file_type,
            "size": len(contents),
            "uploaded_at": datetime.now().isoformat(),
            "embedding_generated": filename in image_embeddings
        }
        
        if metadata:
            try:
                import json
                additional_metadata = json.loads(metadata)
                image_metadata.update(additional_metadata)
            except:
                pass
        
        return {
            "message": "Brandkit image stored successfully",
            "filename": filename,
            "saved_path": str(saved_file_path),
            "embedding_generated": filename in image_embeddings,
            "total_images_in_database": len(image_embeddings),
            "metadata": image_metadata
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
        raise HTTPException(status_code=500, detail=f"Failed to store image: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}