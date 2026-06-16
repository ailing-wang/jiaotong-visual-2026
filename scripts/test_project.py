import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    print("Testing imports...")
    
    try:
        from src.config import LANDMARKS, TOP_K_VALUES, BASE_DB_DIR, QUERY_DIR
        print(f"✓ Config loaded: {len(LANDMARKS)} landmarks, K values: {TOP_K_VALUES}")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from src.utils.io import load_json, save_json, get_image_files
        print("✓ IO utils loaded")
    except Exception as e:
        print(f"✗ IO utils import failed: {e}")
        return False
    
    try:
        from src.retrieval.feature_extractor import FeatureExtractor
        print("✓ FeatureExtractor loaded")
    except Exception as e:
        print(f"✗ FeatureExtractor import failed: {e}")
        return False
    
    try:
        from src.retrieval.retrieval_engine import RetrievalEngine
        print("✓ RetrievalEngine loaded")
    except Exception as e:
        print(f"✗ RetrievalEngine import failed: {e}")
        return False
    
    try:
        from src.retrieval.metrics import evaluate_retrieval, plot_precision_curve
        print("✓ Metrics loaded")
    except Exception as e:
        print(f"✗ Metrics import failed: {e}")
        return False
    
    try:
        from src.detection.text_detector import TextDetector
        print("✓ TextDetector loaded")
    except Exception as e:
        print(f"✗ TextDetector import failed: {e}")
        return False
    
    try:
        from src.detection.visualization import draw_bboxes, create_retrieval_detection_pair
        print("✓ Visualization loaded")
    except Exception as e:
        print(f"✗ Visualization import failed: {e}")
        return False
    
    return True

def test_data_paths():
    print("\nTesting data paths...")
    
    from src.config import BASE_DB_DIR, QUERY_DIR, DETECTION_DATA_DIR
    
    base_files = []
    if os.path.exists(BASE_DB_DIR):
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            base_files.extend([f for f in os.listdir(BASE_DB_DIR) if f.lower().endswith(tuple(ext.split('.')[1:]))])
    
    query_files = []
    if os.path.exists(QUERY_DIR):
        query_files = [f for f in os.listdir(QUERY_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    detection_files = []
    if os.path.exists(DETECTION_DATA_DIR):
        detection_files = [f for f in os.listdir(DETECTION_DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Base images: {len(base_files)}")
    print(f"Query images: {len(query_files)}")
    print(f"Detection images: {len(detection_files)}")
    
    return len(query_files) > 0

def main():
    print("=== Project Test ===")
    
    if test_imports() and test_data_paths():
        print("\n✓ All tests passed!")
        print("\nTo run the project:")
        print("1. python scripts/download_models.py")
        print("2. python scripts/run_retrieval.py")
        print("3. python scripts/run_detection.py")
        print("4. python scripts/run_evaluation.py")
    else:
        print("\n✗ Some tests failed")

if __name__ == '__main__':
    main()