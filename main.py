import os
import uuid
from io import BytesIO

import cv2
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sentence_transformers import SentenceTransformer

# --- Configuration ---
STORAGE_DIR = "stored_screenshots"
EMBEDDING_DIM = 512
TOP_K_RESULTS = 3

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Visual RAG - Sketch to Image Search",
    description="A backend for searching images using sketch/doodle queries via CLIP embeddings and FAISS.",
    version="1.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Static Files ---
app.mount("/files", StaticFiles(directory=STORAGE_DIR), name="files")

# --- Global State ---
clip_model: SentenceTransformer = None
faiss_index: faiss.IndexFlatIP = None
id_to_filename: dict[int, str] = {}
current_id: int = 0


@app.on_event("startup")
async def startup_event():
    """Load the CLIP model and initialize FAISS index on startup."""
    global clip_model, faiss_index
    
    print("Loading CLIP model (clip-ViT-B-32)...")
    clip_model = SentenceTransformer("clip-ViT-B-32")
    print("CLIP model loaded successfully!")
    
    print("Initializing FAISS index (IndexFlatIP for cosine similarity)...")
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    print("FAISS index initialized!")


def process_image_for_matching(image_bytes: bytes) -> Image.Image:
    """
    Process an image to create a sketch-like version for matching.
    
    Steps:
    1. Convert to grayscale
    2. Apply Canny Edge Detection
    3. Invert colors (black lines on white background)
    
    Args:
        image_bytes: Raw bytes of the uploaded image
        
    Returns:
        PIL Image of the processed sketch version
    """
    # Read image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Invert colors (black lines on white background)
    inverted = cv2.bitwise_not(edges)
    
    # Convert to PIL Image (RGB mode for CLIP compatibility)
    pil_image = Image.fromarray(inverted).convert("RGB")
    
    return pil_image


def embed_image(image: Image.Image) -> np.ndarray:
    """
    Generate a normalized embedding for an image using CLIP.
    
    Args:
        image: PIL Image to embed
        
    Returns:
        Normalized numpy array of shape (512,)
    """
    # CLIP model expects PIL images directly
    embedding = clip_model.encode(image)
    
    # Normalize for cosine similarity with IndexFlatIP
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.astype(np.float32)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image for indexing.
    
    - Saves the original file with a unique UUID filename
    - Converts the image to a sketch version using edge detection
    - Embeds the sketch version using CLIP
    - Adds the embedding to the FAISS index
    
    Returns:
        JSON with the generated filename and index ID
    """
    global current_id
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file bytes
        image_bytes = await file.read()
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename or "image.png")[1] or ".png"
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(STORAGE_DIR, unique_filename)
        
        # Save original image
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        # Process image to sketch version
        sketch_image = process_image_for_matching(image_bytes)
        
        # Generate embedding from sketch version
        embedding = embed_image(sketch_image)
        
        # Add to FAISS index
        embedding_2d = embedding.reshape(1, -1)
        faiss_index.add(embedding_2d)
        
        # Store mapping
        id_to_filename[current_id] = unique_filename
        assigned_id = current_id
        current_id += 1
        
        return {
            "success": True,
            "filename": unique_filename,
            "id": assigned_id,
            "message": "Image uploaded and indexed successfully",
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/search")
async def search_by_doodle(file: UploadFile = File(...)):
    """
    Search for similar images using a doodle/sketch query.
    
    - Embeds the doodle directly using CLIP (no edge detection)
    - Searches FAISS index for top K nearest neighbors
    
    Returns:
        JSON with matching filenames and similarity scores
    """
    # Check if index has any vectors
    if faiss_index.ntotal == 0:
        return {
            "success": True,
            "results": [],
            "message": "No images indexed yet",
        }
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file bytes
        image_bytes = await file.read()
        
        # Load as PIL Image directly (no edge detection for query)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Generate embedding
        embedding = embed_image(pil_image)
        embedding_2d = embedding.reshape(1, -1)
        
        # Search FAISS index
        k = min(TOP_K_RESULTS, faiss_index.ntotal)
        distances, indices = faiss_index.search(embedding_2d, k)
        
        # Build results
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            score = float(distances[0][i])
            
            if idx in id_to_filename:
                results.append({
                    "filename": id_to_filename[idx],
                    "score": score,
                    "url": f"/files/{id_to_filename[idx]}",
                })
        
        return {
            "success": True,
            "results": results,
            "message": f"Found {len(results)} matching images",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Visual RAG - Sketch to Image Search",
        "indexed_images": faiss_index.ntotal if faiss_index else 0,
    }


@app.get("/stats")
async def get_stats():
    """Get index statistics."""
    return {
        "total_indexed": faiss_index.ntotal if faiss_index else 0,
        "embedding_dimension": EMBEDDING_DIM,
        "storage_directory": STORAGE_DIR,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
