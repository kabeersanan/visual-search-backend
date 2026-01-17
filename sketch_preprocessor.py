import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class SketchPreprocessor:
    """Preprocess sketches to improve matching with photos"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def preprocess_sketch(self, image_path: str, method: str = 'auto') -> np.ndarray:
        """
        Preprocess sketch image to look more like edge-detected photos
        
        Args:
            image_path: Path to sketch image
            method: 'auto', 'invert', 'edge', or 'adaptive'
        
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize
        gray = cv2.resize(gray, self.target_size)
        
        if method == 'auto':
            return self._auto_preprocess(gray)
        elif method == 'invert':
            return self._invert_method(gray)
        elif method == 'edge':
            return self._edge_method(gray)
        elif method == 'adaptive':
            return self._adaptive_method(gray)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _auto_preprocess(self, gray: np.ndarray) -> np.ndarray:
        """Automatically detect best preprocessing method"""
        mean_val = np.mean(gray)
        
        # Detect if sketch has light or dark background
        if mean_val > 200:
            # Very light background (white) - likely a sketch
            return self._invert_and_threshold(gray)
        elif mean_val < 50:
            # Very dark background - likely already good
            return self._edge_method(gray)
        else:
            # Medium brightness - use adaptive method
            return self._adaptive_method(gray)
    
    def _invert_and_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Invert and threshold - best for sketches on white background"""
        # Invert: make background dark, lines light
        inverted = 255 - gray
        
        # Threshold to get binary image
        _, binary = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)
        
        # Remove noise with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Thin the lines to match edge-detected photos
        kernel_thin = np.ones((2, 2), np.uint8)
        thinned = cv2.erode(cleaned, kernel_thin, iterations=1)
        
        # Apply edge detection to get crisp edges
        edges = cv2.Canny(thinned, 50, 150)
        
        return edges
    
    def _invert_method(self, gray: np.ndarray) -> np.ndarray:
        """Simple inversion method"""
        mean_val = np.mean(gray)
        
        if mean_val > 127:
            # Light background - invert
            return 255 - gray
        else:
            # Dark background - keep as is
            return gray
    
    def _edge_method(self, gray: np.ndarray) -> np.ndarray:
        """Apply edge detection directly"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Edge detection with adjusted thresholds
        edges = cv2.Canny(blurred, 30, 100)
        
        return edges
    
    def _adaptive_method(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding followed by edge detection"""
        # Apply adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        # Apply edge detection
        edges = cv2.Canny(cleaned, 50, 150)
        
        return edges
    
    def save_preprocessed(self, image_path: str, output_path: str, method: str = 'auto'):
        """Preprocess and save image"""
        processed = self.preprocess_sketch(image_path, method)
        cv2.imwrite(output_path, processed)
        print(f"Saved preprocessed image to: {output_path}")
    
    def compare_methods(self, image_path: str, output_dir: str = 'preprocessing_comparison'):
        """Compare all preprocessing methods and save results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        methods = ['auto', 'invert', 'edge', 'adaptive']
        
        for method in methods:
            try:
                processed = self.preprocess_sketch(image_path, method)
                output_path = Path(output_dir) / f"{method}_result.png"
                cv2.imwrite(str(output_path), processed)
                print(f"Saved {method} result to: {output_path}")
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        # Also save original for comparison
        original = cv2.imread(image_path)
        if original is not None:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, self.target_size)
            cv2.imwrite(str(Path(output_dir) / "original.png"), gray)
            print(f"Saved original to: {Path(output_dir) / 'original.png'}")


# Standalone function for quick use
def preprocess_sketch_simple(input_path: str, output_path: str):
    """Simple preprocessing function for quick use"""
    preprocessor = SketchPreprocessor()
    preprocessor.save_preprocessed(input_path, output_path, method='auto')


if __name__ == "__main__":
    # Example usage
    preprocessor = SketchPreprocessor()
    
    # Test with your sketch
    sketch_path = "uploaded_sketch.png"
    
    # Compare all methods
    print("Comparing preprocessing methods...")
    preprocessor.compare_methods(sketch_path)
    
    # Save best result
    print("\nSaving preprocessed sketch...")
    preprocessor.save_preprocessed(sketch_path, "preprocessed_sketch.png", method='auto')
    
    print("\nDone! Check the 'preprocessing_comparison' folder to see all methods.")