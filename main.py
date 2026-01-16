from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uuid
from datetime import datetime
from pathlib import Path

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

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI with venv!"}

@app.post("/search")
async def search(file: UploadFile = File(...), query: Optional[str] = Form(None)):
    """
    Search endpoint that accepts an image file via FormData.
    The frontend should send:
    - 'file': A Blob or File object containing the image
    - 'query': Optional search query string
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
        
        # Process the image here (add your search logic)
        # For now, just return a success response with file info
        return {
            "message": "Search results for query: " + (query or ""),
            "original_filename": file.filename,
            "saved_filename": filename,
            "saved_path": str(saved_file_path),
            "content_type": file_type,
            "size": len(contents),
            "uploaded_at": datetime.now().isoformat()
        }
    except Exception as e:
        # Clean up file if saving failed
        if saved_file_path and saved_file_path.exists():
            try:
                saved_file_path.unlink()
            except:
                pass
        return {"error": str(e), "message": "Failed to save or process image"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}