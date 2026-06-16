import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.text_detector import TextDetector
from src.detection.visualization import draw_bboxes
from src.utils.io import get_image_files, save_json
from src.config import DETECTION_DATA_DIR, DETECTION_OUTPUT_DIR, RETRIEVAL_OUTPUT_DIR

def main():
    print("=== Text Detection ===")
    
    detector = TextDetector()
    print(f"Using detector type: {detector.detector_type}")
    
    image_files = get_image_files(DETECTION_DATA_DIR)
    print(f"Found {len(image_files)} images for detection")
    
    results = {}
    
    for image_path in image_files:
        print(f"Processing: {os.path.basename(image_path)}")
        bboxes = detector.detect(image_path)
        results[image_path] = bboxes
        
        output_filename = os.path.basename(image_path)
        output_path = os.path.join(DETECTION_OUTPUT_DIR, output_filename)
        draw_bboxes(image_path, bboxes, output_path)
    
    output_path = os.path.join(DETECTION_OUTPUT_DIR, 'detection_results.json')
    save_json(results, output_path)
    print(f"Detection results saved to {output_path}")
    
    retrieval_results_path = os.path.join(RETRIEVAL_OUTPUT_DIR, 'retrieval_results.json')
    if os.path.exists(retrieval_results_path):
        from src.utils.io import load_json
        retrieval_results = load_json(retrieval_results_path)
        
        retrieval_detection = {}
        for query_path, retrieval_result in retrieval_results.items():
            query_bboxes = detector.detect(query_path)
            retrieval_detection[query_path] = {
                'query_bboxes': query_bboxes,
                'results': []
            }
            
            for result in retrieval_result[:5]:
                result_bboxes = detector.detect(result['path'])
                retrieval_detection[query_path]['results'].append({
                    'path': result['path'],
                    'bboxes': result_bboxes,
                    'similarity': result['similarity']
                })
        
        output_path = os.path.join(DETECTION_OUTPUT_DIR, 'retrieval_detection_results.json')
        save_json(retrieval_detection, output_path)
        print(f"Retrieval+Detection results saved to {output_path}")
    
    print("Detection completed!")

if __name__ == '__main__':
    main()