import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.spatial.distance import euclidean, cosine

class EdgeFeatureExtractor:
    """Extract and compare edge-based features for image matching"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def extract_edge_features(self, image_path: str) -> Dict:
        """
        Extract multiple edge-based features from an image
        Returns a dictionary with different feature types
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size for consistency
        gray = cv2.resize(gray, self.target_size)
        
        # 1. Canny Edge Detection
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. HOG (Histogram of Oriented Gradients) - excellent for shape
        hog_features = self._compute_hog(edges)
        
        # 3. Contour-based features
        contour_features = self._compute_contour_features(edges)
        
        # 4. Edge density map (divide image into grid and count edges)
        edge_density = self._compute_edge_density(edges, grid_size=8)
        
        # 5. Raw edge image (normalized)
        edge_normalized = edges.flatten() / 255.0
        
        return {
            'hog': hog_features,
            'contours': contour_features,
            'edge_density': edge_density,
            'edge_image': edge_normalized,
            'combined': self._combine_features(hog_features, contour_features, edge_density)
        }
    
    def _compute_hog(self, edges: np.ndarray) -> np.ndarray:
        """Compute HOG features from edge image"""
        win_size = self.target_size
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(edges)
        
        # Normalize
        hog_features = hog_features.flatten()
        hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-6)
        
        return hog_features
    
    def _compute_contour_features(self, edges: np.ndarray) -> np.ndarray:
        """Extract contour-based shape descriptors"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(7)  # Return zero vector if no contours
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute shape descriptors
        moments = cv2.moments(largest_contour)
        
        # Hu Moments (7 invariant moments)
        if moments['m00'] != 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log transform to make them more manageable
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        else:
            hu_moments = np.zeros(7)
        
        return hu_moments
    
    def _compute_edge_density(self, edges: np.ndarray, grid_size: int = 8) -> np.ndarray:
        """
        Divide image into grid and compute edge density in each cell
        This captures spatial distribution of edges
        """
        h, w = edges.shape
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        density = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                density.append(np.sum(cell > 0) / (cell_h * cell_w))
        
        return np.array(density)
    
    def _combine_features(self, hog: np.ndarray, contours: np.ndarray, 
                         edge_density: np.ndarray) -> np.ndarray:
        """Combine different features into single vector"""
        # Normalize each feature type
        hog_norm = hog / (np.linalg.norm(hog) + 1e-6)
        contours_norm = contours / (np.linalg.norm(contours) + 1e-6)
        density_norm = edge_density / (np.linalg.norm(edge_density) + 1e-6)
        
        # Concatenate with weights
        combined = np.concatenate([
            hog_norm * 0.5,      # 50% weight to HOG
            contours_norm * 0.3,  # 30% weight to contours
            density_norm * 0.2    # 20% weight to density
        ])
        
        # Final normalization
        combined = combined / (np.linalg.norm(combined) + 1e-6)
        
        return combined

class EdgeDatabase:
    """Build and query edge-based image database"""
    
    def __init__(self, feature_type='combined'):
        """
        feature_type: 'hog', 'contours', 'edge_density', 'edge_image', or 'combined'
        """
        self.extractor = EdgeFeatureExtractor()
        self.feature_type = feature_type
        self.database = {}
    
    def build_database(self, image_folder: str, save_path: str = 'edge_features.pkl'):
        """Build edge feature database from images in folder"""
        image_folder = Path(image_folder)
        
        print(f"Building edge feature database from: {image_folder}")
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        
        print(f"Found {len(image_files)} images")
        
        for idx, filename in enumerate(image_files, 1):
            image_path = image_folder / filename
            try:
                features = self.extractor.extract_edge_features(str(image_path))
                self.database[filename] = features[self.feature_type]
                print(f"[{idx}/{len(image_files)}] Processed: {filename}")
            except Exception as e:
                print(f"[{idx}/{len(image_files)}] Error processing {filename}: {e}")
        
        # Save database
        with open(save_path, 'wb') as f:
            pickle.dump({
                'database': self.database,
                'feature_type': self.feature_type
            }, f)
        
        print(f"\nSaved {len(self.database)} edge features to {save_path}")
        return self.database
    
    def load_database(self, load_path: str = 'edge_features.pkl'):
        """Load pre-computed edge features"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.database = data['database']
            self.feature_type = data['feature_type']
        
        print(f"Loaded {len(self.database)} edge features")
        return self.database
    
    def find_similar(self, query_image_path: str, top_k: int = 10, 
                    metric: str = 'cosine') -> List[Tuple[str, float]]:
        """
        Find similar images based on edge features
        metric: 'cosine' or 'euclidean'
        """
        # Extract features from query image
        query_features = self.extractor.extract_edge_features(query_image_path)
        query_vector = query_features[self.feature_type]
        
        # Calculate similarities
        similarities = []
        for filename, db_vector in self.database.items():
            if metric == 'cosine':
                # Cosine similarity (1 - cosine distance)
                similarity = 1 - cosine(query_vector, db_vector)
            else:  # euclidean
                # Convert distance to similarity (smaller distance = higher similarity)
                distance = euclidean(query_vector, db_vector)
                similarity = 1 / (1 + distance)
            
            similarities.append((filename, float(similarity)))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


# Standalone functions for direct use
def extract_and_save_edges(image_path: str, output_path: str):
    """Extract and save edge image for visualization"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    edges = cv2.Canny(img, 50, 150)
    cv2.imwrite(output_path, edges)
    print(f"Saved edge image to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize database with 'combined' features (best for most cases)
    db = EdgeDatabase(feature_type='combined')
    
    # Build database
    db.build_database('photos', save_path='edge_features.pkl')
    
    # Search for similar images
    query_image = 'uploaded_image.png'
    results = db.find_similar(query_image, top_k=10, metric='cosine')
    
    print("\n" + "="*60)
    print("Top 10 Similar Images (by edge features):")
    print("="*60)
    for rank, (filename, score) in enumerate(results, 1):
        print(f"{rank:2d}. {filename:40s} | Similarity: {score:.4f}")
    
    # Optional: Save edge visualization
    extract_and_save_edges(query_image, 'query_edges.png')