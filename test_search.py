"""
Test script to evaluate different search methods
Run this to find the best approach for your sketch-to-photo matching
"""

from hybrid_search import HybridImageSearch
from sketch_preprocessor import SketchPreprocessor
import cv2
from pathlib import Path

def visualize_preprocessing(sketch_path: str):
    """
    Step 1: Visualize how preprocessing affects your sketch
    This helps you understand what the model "sees"
    """
    print("\n" + "="*60)
    print("STEP 1: PREPROCESSING VISUALIZATION")
    print("="*60)
    
    preprocessor = SketchPreprocessor()
    
    # Create comparison
    output_dir = "preprocessing_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\nTesting preprocessing methods on: {sketch_path}")
    preprocessor.compare_methods(sketch_path, output_dir)
    
    print(f"\n✓ Results saved to '{output_dir}/' folder")
    print("  Check these images to see which preprocessing looks best")
    print("  The 'auto' method is usually best, but you can compare all")

def test_all_methods(sketch_path: str, top_k: int = 10):
    """
    Step 2: Test all search methods and compare results
    """
    print("\n" + "="*60)
    print("STEP 2: TESTING ALL SEARCH METHODS")
    print("="*60)
    
    searcher = HybridImageSearch()
    searcher.load_databases()
    
    if not searcher.dinov2_db and not searcher.edge_db:
        print("\n❌ ERROR: No databases found!")
        print("   Please build the databases first:")
        print("   - Run: python build_databases.py")
        print("   - Or use the API: POST /databases/rebuild")
        return
    
    print(f"\n✓ Loaded databases:")
    print(f"  - DINOv2: {len(searcher.dinov2_db)} images")
    print(f"  - Edge features: {len(searcher.edge_db)} images")
    
    # Compare all methods
    results = searcher.compare_methods(sketch_path, top_k=top_k)
    
    # Analyze results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for preprocess_status in ['without_preprocessing', 'with_preprocessing']:
        print(f"\n{preprocess_status.upper().replace('_', ' ')}:")
        print("-" * 60)
        
        for method in ['dinov2', 'edge', 'hybrid']:
            method_results = results[preprocess_status][method]
            
            if method_results:
                top_score = method_results[0]['similarity']
                top_file = method_results[0]['filename']
                
                print(f"\n  {method.upper()}:")
                print(f"    Top result: {top_file}")
                print(f"    Top score: {top_score:.4f}")
                
                # Show top 3
                print(f"    Top 3:")
                for i, r in enumerate(method_results[:3], 1):
                    print(f"      {i}. {r['filename']:30s} | Score: {r['similarity']:.4f}")

def recommend_best_method(sketch_path: str):
    """
    Step 3: Get automatic recommendation for best method
    """
    print("\n" + "="*60)
    print("STEP 3: RECOMMENDATION")
    print("="*60)
    
    searcher = HybridImageSearch()
    searcher.load_databases()
    
    # Test each combination
    test_configs = [
        {'preprocess': False, 'method': 'dinov2', 'name': 'DINOv2 (no preprocess)'},
        {'preprocess': True, 'method': 'dinov2', 'name': 'DINOv2 (with preprocess)'},
        {'preprocess': False, 'method': 'edge', 'name': 'Edge (no preprocess)'},
        {'preprocess': True, 'method': 'edge', 'name': 'Edge (with preprocess)'},
        {'preprocess': False, 'method': 'hybrid', 'name': 'Hybrid (no preprocess)'},
        {'preprocess': True, 'method': 'hybrid', 'name': 'Hybrid (with preprocess)'},
    ]
    
    scores = {}
    
    for config in test_configs:
        try:
            results = searcher.search_with_preprocessing(
                sketch_path,
                top_k=5,
                preprocess=config['preprocess'],
                method=config['method']
            )
            
            if results:
                avg_top3 = sum(r['similarity'] for r in results[:3]) / 3
                scores[config['name']] = {
                    'avg_score': avg_top3,
                    'top_score': results[0]['similarity'],
                    'top_file': results[0]['filename']
                }
        except Exception as e:
            print(f"  Error testing {config['name']}: {e}")
    
    # Find best
    if scores:
        best_method = max(scores.items(), key=lambda x: x[1]['avg_top3'])
        
        print("\n📊 ANALYSIS:")
        print("-" * 60)
        for name, score_data in sorted(scores.items(), key=lambda x: x[1]['avg_top3'], reverse=True):
            print(f"  {name:35s} | Avg: {score_data['avg_top3']:.4f} | Top: {score_data['top_score']:.4f}")
        
        print("\n✅ RECOMMENDATION:")
        print("-" * 60)
        print(f"  Best method: {best_method[0]}")
        print(f"  Average top-3 score: {best_method[1]['avg_top3']:.4f}")
        print(f"  Top result: {best_method[1]['top_file']}")
        print(f"  Top score: {best_method[1]['top_score']:.4f}")
        
        # Parse recommendation
        method_name = best_method[0]
        if 'DINOv2' in method_name:
            method = 'dinov2'
        elif 'Edge' in method_name:
            method = 'edge'
        else:
            method = 'hybrid'
        
        preprocess = 'with preprocess' in method_name.lower()
        
        print("\n💡 USE THIS IN YOUR API:")
        print("-" * 60)
        print(f"  method='{method}'")
        print(f"  preprocess={preprocess}")

def quick_test(sketch_path: str, expected_result: str = None):
    """
    Quick test: Just run the recommended config and show results
    """
    print("\n" + "="*60)
    print("QUICK TEST")
    print("="*60)
    
    searcher = HybridImageSearch()
    searcher.load_databases()
    
    # Try hybrid with preprocessing (usually best)
    print("\nTesting: Hybrid method with preprocessing...")
    results = searcher.search_with_preprocessing(
        sketch_path,
        top_k=10,
        preprocess=True,
        method='hybrid'
    )
    
    print("\n📋 TOP 10 RESULTS:")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        marker = "✓" if expected_result and expected_result in r['filename'] else " "
        print(f"{marker} {i:2d}. {r['filename']:40s} | Score: {r['similarity']:.4f}")
        print(f"      DINOv2: {r['dinov2_score']:.4f} | Edge: {r['edge_score']:.4f}")
    
    if expected_result:
        # Check if expected result is in top 10
        found = any(expected_result in r['filename'] for r in results)
        if found:
            rank = next(i for i, r in enumerate(results, 1) if expected_result in r['filename'])
            print(f"\n✅ Expected result '{expected_result}' found at rank {rank}!")
        else:
            print(f"\n❌ Expected result '{expected_result}' not in top 10")

def main():
    """
    Main test workflow
    """
    print("\n" + "="*70)
    print(" HYBRID IMAGE SEARCH - TEST SUITE")
    print("="*70)
    
    # Configuration
    sketch_path = "uploaded_sketch.png"  # Change this to your sketch path
    expected_result = None  # Change this to expected filename if you know it
    
    # Check if sketch exists
    if not Path(sketch_path).exists():
        print(f"\n❌ ERROR: Sketch not found at '{sketch_path}'")
        print("   Please update the 'sketch_path' variable in this script")
        return
    
    print(f"\nTesting with sketch: {sketch_path}")
    if expected_result:
        print(f"Expected result: {expected_result}")
    
    # Run tests
    try:
        # Full analysis
        print("\n" + "="*70)
        print(" RUNNING FULL ANALYSIS")
        print("="*70)
        
        visualize_preprocessing(sketch_path)
        test_all_methods(sketch_path, top_k=10)
        recommend_best_method(sketch_path)
        
        # Quick test
        print("\n" + "="*70)
        print(" QUICK TEST WITH BEST METHOD")
        print("="*70)
        quick_test(sketch_path, expected_result)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print(" TEST COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check 'preprocessing_results/' folder to see preprocessing output")
    print("  2. Review the recommendation above")
    print("  3. Use the recommended method in your API calls")
    print("  4. Start your API: python main.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()