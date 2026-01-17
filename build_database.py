"""
Build both DINOv2 and edge feature databases from your photos folder
Run this once before using the search API
"""

import os
import pickle
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import cv2
import numpy as np
from scipy.spatial.distance import cosine

# Configuration
PHOTOS_DIR = Path("photos")
DINOV2_OUTPUT = "image_embeddings_dinov2.pkl"
EDGE_OUTPUT = "edge_features.pkl"
TARGET_SIZE = (224, 224)

class DatabaseBuilder:
    """Build both DINOv2 and edge feature databases"""
    
    def __init__(self):
        self.setup_dinov2()
    
    def setup_dinov2(self):
        """Load DINOv2 model"""
        print("\nLoading DINOv2 model...")
        try:
            self.dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base')
            self.dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dinov2_model.to(self.device)
            self.dinov2_model.eval()
            print(f"✓ DINOv2 loaded on {self.device}")
        except Exception as e:
            print(f"✗ Failed to load DINOv2: {e}")
            self.dinov2_model = None
    
    def extract_dinov2_embedding(self, image_path: str) -> np.ndarray:
        """Extract DINOv2 embedding from image"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.dinov2_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    
    def extract_edge_features(self, image_path: str) -> np.ndarray:
        """Extract edge features from image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, TARGET_SIZE)
        edges = cv2.Canny(gray, 50, 150)
        
        # Compute features
        hog = self._compute_hog(edges)
        contours = self._compute_contours(edges)
        density = self._compute_density(edges)
        
        # Combine
        return self._combine_features(hog, contours, density)
    
    def _compute_hog(self, edges: np.ndarray) -> np.ndarray:
        """Compute HOG features"""
        hog = cv2.HOGDescriptor(TARGET_SIZE, (16, 16), (8, 8), (8, 8), 9)
        features = hog.compute(edges).flatten()
        return features / (np.linalg.norm(features) + 1e-6)
    
    def _compute_contours(self, edges: np.ndarray) -> np.ndarray:
        """Compute Hu moments"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(7)
        
        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        
        if moments['m00'] != 0:
            hu = cv2.HuMoments(moments).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        else:
            hu = np.zeros(7)
        
        return hu
    
    def _compute_density(self, edges: np.ndarray, grid_size: int = 8) -> np.ndarray:
        """Compute edge density grid"""
        h, w = edges.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        density = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                density.append(np.sum(cell > 0) / (cell_h * cell_w))
        
        return np.array(density)
    
    def _combine_features(self, hog: np.ndarray, contours: np.ndarray, 
                         density: np.ndarray) -> np.ndarray:
        """Combine edge features"""
        hog_norm = hog / (np.linalg.norm(hog) + 1e-6)
        contours_norm = contours / (np.linalg.norm(contours) + 1e-6)
        density_norm = density / (np.linalg.norm(density) + 1e-6)
        
        combined = np.concatenate([
            hog_norm * 0.5,
            contours_norm * 0.3,
            density_norm * 0.2
        ])
        
        return combined / (np.linalg.norm(combined) + 1e-6)
    
    def build_databases(self, photos_dir: Path):
        """Build both databases"""
        if not photos_dir.exists():
            print(f"\n✗ ERROR: Photos directory '{photos_dir}' not found!")
            print("  Please create it and add your images")
            return False
        
        # Get image files
        image_files = [f for f in os.listdir(photos_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        
        if not image_files:
            print(f"\n✗ ERROR: No images found in '{photos_dir}'")
            return False
        
        print(f"\n{'='*70}")
        print(f" BUILDING DATABASES FROM {len(image_files)} IMAGES")
        print(f"{'='*70}")
        
        dinov2_embeddings = {}
        edge_features = {}
        
        successful = 0
        failed = 0
        
        for idx, filename in enumerate(image_files, 1):
            image_path = photos_dir / filename
            
            print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
            
            try:
                # DINOv2 embedding
                if self.dinov2_model is not None:
                    print("  - Extracting DINOv2 embedding...", end=" ")
                    emb = self.extract_dinov2_embedding(str(image_path))
                    dinov2_embeddings[filename] = emb
                    print("✓")
                
                # Edge features
                print("  - Extracting edge features...", end=" ")
                edge_feat = self.extract_edge_features(str(image_path))
                edge_features[filename] = edge_feat
                print("✓")
                
                successful += 1
                
            except Exception as e:
                print(f"\n  ✗ Error: {e}")
                failed += 1
        
        # Save databases
        print(f"\n{'='*70}")
        print(" SAVING DATABASES")
        print(f"{'='*70}")
        
        if dinov2_embeddings:
            print(f"\nSaving DINOv2 embeddings to '{DINOV2_OUTPUT}'...")
            with open(DINOV2_OUTPUT, 'wb') as f:
                pickle.dump(dinov2_embeddings, f)
            print(f"✓ Saved {len(dinov2_embeddings)} DINOv2 embeddings")
        
        if edge_features:
            print(f"\nSaving edge features to '{EDGE_OUTPUT}'...")
            with open(EDGE_OUTPUT, 'wb') as f:
                pickle.dump({
                    'database': edge_features,
                    'feature_type': 'combined'
                }, f)
            print(f"✓ Saved {len(edge_features)} edge features")
        
        # Summary
        print(f"\n{'='*70}")
        print(" BUILD SUMMARY")
        print(f"{'='*70}")
        print(f"  Total images: {len(image_files)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  DINOv2 embeddings: {len(dinov2_embeddings)}")
        print(f"  Edge features: {len(edge_features)}")
        print(f"{'='*70}\n")
        
        return successful > 0

def main():
    print(f"\n{'='*70}")
    print(" DATABASE BUILDER")
    print(f"{'='*70}")
    print(f"\nPhotos directory: {PHOTOS_DIR}")
    print(f"DINOv2 output: {DINOV2_OUTPUT}")
    print(f"Edge features output: {EDGE_OUTPUT}")
    
    # Check for existing databases
    existing = []
    if Path(DINOV2_OUTPUT).exists():
        existing.append(DINOV2_OUTPUT)
    if Path(EDGE_OUTPUT).exists():
        existing.append(EDGE_OUTPUT)
    
    if existing:
        print(f"\n⚠️  WARNING: Existing databases found:")
        for db in existing:
            print(f"  - {db}")
        
        response = input("\nOverwrite existing databases? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\n✗ Build cancelled")
            return
    
    # Build databases
    builder = DatabaseBuilder()
    success = builder.build_databases(PHOTOS_DIR)
    
    if success:
        print("\n✅ BUILD COMPLETE!")
        print("\nNext steps:")
        print("  1. Test the databases: python test_search.py")
        print("  2. Start the API: python main.py")
        print(f"\n{'='*70}\n")
    else:
        print("\n✗ Build failed. Please check errors above.")

if __name__ == "__main__":
    main()