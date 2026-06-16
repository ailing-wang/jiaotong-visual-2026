import cv2
import numpy as np
import os
from PIL import Image

def cv_imread(image_path):
    """读取中文路径的图片"""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except:
        return None

class TextDetector:
    def __init__(self):
        self.net = None
        self.load_model()
    
    def load_model(self):
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'models', 'frozen_east_text_detection.pb')
        
        if not os.path.exists(model_path):
            print(f"Warning: EAST model not found at {model_path}")
            print("Using OpenCV MSER detector as fallback")
            self.detector_type = 'mser'
            self.mser = cv2.MSER_create()
        else:
            self.detector_type = 'east'
            self.net = cv2.dnn.readNet(model_path)
            self.layer_names = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"
            ]
    
    def detect_east(self, image_path, conf_threshold=0.5, nms_threshold=0.4):
        if self.net is None:
            return self.detect_mser(image_path)
        
        image = cv_imread(image_path)
        if image is None:
            return []
        
        orig_height, orig_width = image.shape[:2]
        new_width, new_height = 320, 320
        
        blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        
        self.net.setInput(blob)
        scores, geometry = self.net.forward(self.layer_names)
        
        num_rows, num_cols = scores.shape[2:4]
        rects = []
        confidences = []
        
        for y in range(num_rows):
            scores_data = scores[0, 0, y, :]
            x_data0 = geometry[0, 0, y, :]
            x_data1 = geometry[0, 1, y, :]
            x_data2 = geometry[0, 2, y, :]
            x_data3 = geometry[0, 3, y, :]
            angles_data = geometry[0, 4, y, :]
            
            for x in range(num_cols):
                score = scores_data[x]
                if score < conf_threshold:
                    continue
                
                offset_x = x * 4.0
                offset_y = y * 4.0
                
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]
                
                end_x = int(offset_x + cos * x_data1[x] + sin * x_data2[x])
                end_y = int(offset_y - sin * x_data1[x] + cos * x_data2[x])
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                
                rects.append((start_x, start_y, end_x, end_y))
                confidences.append(float(score))
        
        indices = cv2.dnn.NMSBoxes(rects, confidences, conf_threshold, nms_threshold)
        
        results = []
        scale_x = orig_width / new_width
        scale_y = orig_height / new_height
        
        if len(indices) > 0:
            for i in indices.flatten():
                start_x, start_y, end_x, end_y = rects[i]
                start_x = int(start_x * scale_x)
                start_y = int(start_y * scale_y)
                end_x = int(end_x * scale_x)
                end_y = int(end_y * scale_y)
                
                results.append({
                    'bbox': (start_x, start_y, end_x, end_y),
                    'confidence': confidences[i]
                })
        
        return results
    
    def detect_mser(self, image_path):
        image = cv_imread(image_path)
        if image is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        regions, _ = self.mser.detectRegions(bw)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        
        results = []
        for hull in hulls:
            x, y, w, h = cv2.boundingRect(hull)
            if w > 5 and h > 5 and w < 500 and h < 500:
                results.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.8
                })
        
        return results
    
    def detect(self, image_path):
        if self.detector_type == 'east':
            return self.detect_east(image_path)
        else:
            return self.detect_mser(image_path)