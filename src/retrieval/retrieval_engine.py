import numpy as np
from scipy.spatial.distance import cdist
import os
from src.config import *
from src.utils.io import save_pickle, load_pickle, get_image_files
from src.retrieval.feature_extractor import FeatureExtractor
from src.utils.dataset import RetrievalDataset

class RetrievalEngine:
    def __init__(self, model_name='resnet50'):
        self.extractor = FeatureExtractor(model_name=model_name)
        self.features = None
        self.image_paths = None
        self.index_path = os.path.join(MODEL_DIR, f'{model_name}_features.pkl')
    
    def build_index(self, image_dir=BASE_DB_DIR):
        image_files = get_image_files(image_dir)
        print(f"Building index from {len(image_files)} images...")
        
        dataset = RetrievalDataset(image_files, transform=self.extractor.transform)
        self.features, self.image_paths = self.extractor.extract_batch(dataset)
        
        save_pickle({
            'features': self.features,
            'image_paths': self.image_paths
        }, self.index_path)
        
        print(f"Index saved to {self.index_path}")
    
    def load_index(self):
        if os.path.exists(self.index_path):
            data = load_pickle(self.index_path)
            self.features = data['features']
            self.image_paths = data['image_paths']
            print(f"Loaded index with {len(self.image_paths)} images")
        else:
            raise FileNotFoundError(f"Index not found at {self.index_path}")
    
    def search(self, query_image, top_k=20):
        query_feature = self.extractor.extract(query_image)
        similarities = 1 - cdist(query_feature.reshape(1, -1), self.features, 'cosine')[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            results.append({
                'path': self.image_paths[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def search_by_path(self, query_path, top_k=20):
        from PIL import Image
        query_image = Image.open(query_path).convert('RGB')
        return self.search(query_image, top_k)
    
    def batch_search(self, query_paths, top_k=20):
        results = {}
        for query_path in query_paths:
            results[query_path] = self.search_by_path(query_path, top_k)
        return results