import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.metrics import evaluate_retrieval, plot_all_precision_curves, print_metrics_summary
from src.detection.visualization import create_retrieval_detection_pair
from src.utils.io import load_json, get_landmark_from_filename
from src.config import RETRIEVAL_OUTPUT_DIR, DETECTION_OUTPUT_DIR, METRICS_OUTPUT_DIR, LANDMARKS

def evaluate_retrieval_task():
    print("=== Evaluating Retrieval ===")
    
    retrieval_results_path = os.path.join(RETRIEVAL_OUTPUT_DIR, 'retrieval_results.json')
    if not os.path.exists(retrieval_results_path):
        print("Retrieval results not found. Please run run_retrieval.py first.")
        return
    
    results = load_json(retrieval_results_path)
    
    query_landmark_groups = {}
    for query_path in results.keys():
        landmark = get_landmark_from_filename(query_path)
        if landmark not in query_landmark_groups:
            query_landmark_groups[landmark] = []
        query_landmark_groups[landmark].append(query_path)
    
    precision_results = evaluate_retrieval(results, query_landmark_groups)
    
    print_metrics_summary(precision_results)
    
    plot_all_precision_curves(precision_results)
    print(f"Precision curves saved to {METRICS_OUTPUT_DIR}")
    
    metrics_output_path = os.path.join(METRICS_OUTPUT_DIR, 'precision_metrics.json')
    from src.utils.io import save_json
    save_json(precision_results, metrics_output_path)
    print(f"Metrics saved to {metrics_output_path}")

def evaluate_detection_task():
    print("\n=== Evaluating Detection ===")
    
    retrieval_detection_path = os.path.join(DETECTION_OUTPUT_DIR, 'retrieval_detection_results.json')
    if not os.path.exists(retrieval_detection_path):
        print("Retrieval+Detection results not found. Please run run_detection.py first.")
        return
    
    retrieval_detection = load_json(retrieval_detection_path)
    
    landmark_samples = {landmark: [] for landmark in LANDMARKS}
    
    for query_path, data in retrieval_detection.items():
        landmark = get_landmark_from_filename(query_path)
        if landmark in landmark_samples and len(landmark_samples[landmark]) < 2:
            landmark_samples[landmark].append(query_path)
    
    for landmark in LANDMARKS:
        for idx, query_path in enumerate(landmark_samples[landmark]):
            data = retrieval_detection[query_path]
            retrieval_results = [{
                'path': r['path'],
                'similarity': r['similarity']
            } for r in data['results']]
            
            detection_results = {query_path: data['query_bboxes']}
            for r in data['results']:
                detection_results[r['path']] = r['bboxes']
            
            output_path = os.path.join(DETECTION_OUTPUT_DIR, f'{landmark}_{idx+1}_pair.png')
            create_retrieval_detection_pair(query_path, retrieval_results, detection_results, output_path)
            print(f"Saved visualization: {output_path}")
    
    print("Detection visualization completed!")

def main():
    evaluate_retrieval_task()
    evaluate_detection_task()
    
    print("\n=== Evaluation Completed ===")
    print(f"1. Precision curves: {METRICS_OUTPUT_DIR}")
    print(f"2. Detection visualizations: {DETECTION_OUTPUT_DIR}")

if __name__ == '__main__':
    main()