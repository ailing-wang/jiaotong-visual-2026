import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.retrieval_engine import RetrievalEngine
from src.utils.io import get_image_files, save_json, get_landmark_from_filename
from src.config import QUERY_DIR, BASE_DB_DIR, RETRIEVAL_OUTPUT_DIR, TOP_K_VALUES

def main():
    print("=== Image Retrieval ===")
    
    engine = RetrievalEngine(model_name='resnet50')
    
    try:
        engine.load_index()
    except FileNotFoundError:
        print("Building new index...")
        engine.build_index(BASE_DB_DIR)
    
    query_files = get_image_files(QUERY_DIR)
    print(f"Found {len(query_files)} query images")
    
    results = {}
    max_k = max(TOP_K_VALUES)
    
    for query_path in query_files:
        print(f"Processing: {os.path.basename(query_path)}")
        result = engine.search_by_path(query_path, top_k=max_k)
        results[query_path] = result
    
    output_path = os.path.join(RETRIEVAL_OUTPUT_DIR, 'retrieval_results.json')
    save_json(results, output_path)
    print(f"Retrieval results saved to {output_path}")
    
    landmark_results = {}
    for query_path, result in results.items():
        landmark = get_landmark_from_filename(query_path)
        if landmark not in landmark_results:
            landmark_results[landmark] = []
        landmark_results[landmark].append({
            'query': query_path,
            'results': result
        })
    
    output_path = os.path.join(RETRIEVAL_OUTPUT_DIR, 'landmark_retrieval_results.json')
    save_json(landmark_results, output_path)
    print(f"Landmark-wise results saved to {output_path}")
    
    print("Retrieval completed!")

if __name__ == '__main__':
    main()