import cv2
import os
import numpy as np

def cv_imread(image_path):
    """读取中文路径的图片"""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except:
        return None

def cv_imwrite(output_path, image):
    """写入中文路径的图片"""
    try:
        cv2.imencode('.png', image)[1].tofile(output_path)
        return True
    except:
        return False

def draw_bboxes(image_path, bboxes, output_path):
    image = cv_imread(image_path)
    if image is None:
        return
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox['bbox']
        confidence = bbox.get('confidence', 0.0)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f'{confidence:.2f}'
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(image, (x1, y1 - label_height - 10), 
                      (x1 + label_width, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv_imwrite(output_path, image)

def create_retrieval_detection_pair(query_path, retrieval_results, detection_results, 
                                    output_path, top_n=3):
    query_image = cv_imread(query_path)
    if query_image is None:
        return
    
    rows = 2
    cols = top_n + 1
    
    heights = []
    widths = []
    
    heights.append(query_image.shape[0])
    widths.append(query_image.shape[1])
    
    result_images = []
    for i, result in enumerate(retrieval_results[:top_n]):
        img = cv_imread(result['path'])
        if img is not None:
            result_images.append(img)
            heights.append(img.shape[0])
            widths.append(img.shape[1])
    
    max_height = max(heights)
    total_width = sum(widths[:cols])
    
    combined = np.zeros((max_height * rows, total_width, 3), dtype=np.uint8)
    
    combined[:query_image.shape[0], :query_image.shape[1]] = query_image
    
    current_x = query_image.shape[1]
    for i, img in enumerate(result_images):
        combined[:img.shape[0], current_x:current_x + img.shape[1]] = img
        current_x += img.shape[1]
    
    query_detection = cv_imread(query_path)
    for bbox in detection_results.get(query_path, []):
        x1, y1, x2, y2 = bbox['bbox']
        cv2.rectangle(query_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    combined[max_height:max_height + query_detection.shape[0], :query_detection.shape[1]] = query_detection
    
    current_x = query_image.shape[1]
    for i, result in enumerate(retrieval_results[:top_n]):
        det_result = detection_results.get(result['path'], [])
        img = cv_imread(result['path'])
        if img is not None:
            for bbox in det_result:
                x1, y1, x2, y2 = bbox['bbox']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            combined[max_height:max_height + img.shape[0], current_x:current_x + img.shape[1]] = img
            current_x += img.shape[1]
    
    cv_imwrite(output_path, combined)