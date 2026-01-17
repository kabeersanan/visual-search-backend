import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine, euclidean
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

from sketch_preprocessor import SketchPreprocessor


class HybridImageSearch:
    """
    Hybrid search combining:
    1. DINOv2 semantic embeddings
    2. Edge-based features
    3. Sketch preprocessing
    """
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.preprocessor = SketchPreprocessor(target_size)
        
        # Load DINOv2
        print("Loading DINOv2 model...")
        try:
            self.dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base')
            self.dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dinov2_model.to(self.device)
            self.dinov2_model.eval()
            print(f"DINOv2 loaded on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load DINOv2: {e}")
            self.dinov2_model = None
            self.dinov2_processor = None
            self.device = None
        
        # Databases
        self.dinov2_db = {}
        self.edge_db = {}
    
    def load_databases(self, dinov2_path='image_embeddings_dinov2.pkl', 
                      edge_path='edge_features.pkl'):
        """Load pre-computed databases"""
        # Load DINOv2 embeddings
        if os.path.exists(dinov2_path):
            try:
                with open(dinov2_path, 'rb') as f:
                    self.dinov2_db = pickle.load(f)
                print(f"Loaded {len(self.dinov2_db)} DINOv2 embeddings")
            except Exception as e:
                print(f"Error loading DINOv2 database: {e}")
        
        # Load edge features
        if os.path.exists(edge_path):
            try:
                with open(edge_path, 'rb') as f:
                    data = pickle.load(f)
                    self.edge_db = data.get('database', {})
                print(f"Loaded {len(self.edge_db)} edge features")
            except Exception as e:
                print(f"Error loading edge database: {e}")
    
    def extract_dinov2_embedding(self, image_path: str) -> np.ndarray:
        """Extract DINOv2 embedding"""
        if self.dinov2_model is None:
            raise ValueError("DINOv2 model not loaded")
        
        image = Image.open(image_path).convert('RGB')
        inputs = self.dinov2_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    
    def extract_edge_features(self, image_path: str) -> np.ndarray:
        """Extract edge features (HOG + Contours + Density)"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.target_size)
        edges = cv2.Canny(gray, 50, 150)
        
        # Compute features
        hog = self._compute_hog(edges)
        contours = self._compute_contours(edges)
        density = self._compute_density(edges)
        
        # Combine
        combined = self._combine_features(hog, contours, density)
        return combined
    
    def _compute_hog(self, edges: np.ndarray) -> np.ndarray:
        """Compute HOG features"""
        hog = cv2.HOGDescriptor(
            self.target_size, (16, 16), (8, 8), (8, 8), 9
        )
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
    
    def search_with_preprocessing(self, query_image_path: str, top_k: int = 10,
                                 preprocess: bool = True, 
                                 method: str = 'hybrid',
                                 weights: Dict[str, float] = None) -> List[Dict]:
        """
        Search with optional preprocessing
        
        Args:
            query_image_path: Path to query image
            top_k: Number of results to return
            preprocess: Whether to preprocess sketch
            method: 'dinov2', 'edge', or 'hybrid'
            weights: {'dinov2': 0.6, 'edge': 0.4} for hybrid
        
        Returns:
            List of similar images with scores
        """
        if weights is None:
            weights = {'dinov2': 0.6, 'edge': 0.4}
        
        # Preprocess if needed
        if preprocess:
            print("Preprocessing sketch...")
            processed = self.preprocessor.preprocess_sketch(query_image_path, 'auto')
            temp_path = 'temp_preprocessed.png'
            cv2.imwrite(temp_path, processed)
            query_path = temp_path
        else:
            query_path = query_image_path
        
        results = {}
        
        # DINOv2 search
        if method in ['dinov2', 'hybrid'] and self.dinov2_db:
            print("Computing DINOv2 similarities...")
            try:
                query_emb = self.extract_dinov2_embedding(query_path)
                
                for filename, db_emb in self.dinov2_db.items():
                    sim = 1 - cosine(query_emb, db_emb)
                    if filename not in results:
                        results[filename] = {'dinov2': 0, 'edge': 0}
                    results[filename]['dinov2'] = float(sim)
            except Exception as e:
                print(f"DINOv2 search failed: {e}")
        
        # Edge search
        if method in ['edge', 'hybrid'] and self.edge_db:
            print("Computing edge similarities...")
            try:
                query_edges = self.extract_edge_features(query_path)
                
                for filename, db_edges in self.edge_db.items():
                    sim = 1 - cosine(query_edges, db_edges)
                    if filename not in results:
                        results[filename] = {'dinov2': 0, 'edge': 0}
                    results[filename]['edge'] = float(sim)
            except Exception as e:
                print(f"Edge search failed: {e}")
        
        # Clean up temp file
        if preprocess and os.path.exists('temp_preprocessed.png'):
            os.remove('temp_preprocessed.png')
        
        # Compute final scores
        final_results = []
        for filename, scores in results.items():
            if method == 'hybrid':
                final_score = (
                    scores['dinov2'] * weights['dinov2'] + 
                    scores['edge'] * weights['edge']
                )
            elif method == 'dinov2':
                final_score = scores['dinov2']
            else:  # edge
                final_score = scores['edge']
            
            final_results.append({
                'filename': filename,
                'similarity': final_score,
                'dinov2_score': scores['dinov2'],
                'edge_score': scores['edge'],
                'filepath': str(Path('photos') / filename)
            })
        
        # Sort and return top_k
        final_results.sort(key=lambda x: x['similarity'], reverse=True)
        return final_results[:top_k]
    
    def compare_methods(self, query_image_path: str, top_k: int = 5) -> Dict:
        """Compare all search methods"""
        results = {
            'without_preprocessing': {},
            'with_preprocessing': {}
        }
        
        # Without preprocessing
        print("\n" + "="*60)
        print("SEARCH WITHOUT PREPROCESSING")
        print("="*60)
        
        for method in ['dinov2', 'edge', 'hybrid']:
            print(f"\nMethod: {method}")
            res = self.search_with_preprocessing(
                query_image_path, top_k, preprocess=False, method=method
            )
            results['without_preprocessing'][method] = res
            
            print(f"Top 3 results for {method}:")
            for i, r in enumerate(res[:3], 1):
                print(f"  {i}. {r['filename']} - Score: {r['similarity']:.4f}")
        
        # With preprocessing
        print("\n" + "="*60)
        print("SEARCH WITH PREPROCESSING")
        print("="*60)
        
        for method in ['dinov2', 'edge', 'hybrid']:
            print(f"\nMethod: {method}")
            res = self.search_with_preprocessing(
                query_image_path, top_k, preprocess=True, method=method
            )
            results['with_preprocessing'][method] = res
            
            print(f"Top 3 results for {method}:")
            for i, r in enumerate(res[:3], 1):
                print(f"  {i}. {r['filename']} - Score: {r['similarity']:.4f}")
        
        return results


if __name__ == "__main__":
    # Initialize hybrid search
    searcher = HybridImageSearch()
    
    # Load databases
    searcher.load_databases()
    
    # Test query
    query_image = "uploaded_sketch.png"
    
    # Compare all methods
    print("Comparing all search methods...")
    results = searcher.compare_methods(query_image, top_k=10)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("\nCheck the results above to see which method works best!")
    print("\nRecommended: Look for the method with the highest similarity scores")
    print("and where the correct image appears in the top results.")