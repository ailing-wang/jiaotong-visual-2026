import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class RetrievalDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        # 过滤掉无法打开的图片
        self.image_paths = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                    self.image_paths.append(path)
            except:
                print(f"Skipping corrupted image: {path}")
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image_path
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return torch.zeros(3, 224, 224), image_path

class DetectionDataset(Dataset):
    def __init__(self, image_paths, annotation_paths, transform=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image_path, annotation_path
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None, image_path, annotation_path

def split_query_by_landmark(query_files):
    landmark_groups = {}
    for file in query_files:
        basename = os.path.splitext(os.path.basename(file))[0]
        for landmark in ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]:
            if basename.startswith(landmark + '-'):
                if landmark not in landmark_groups:
                    landmark_groups[landmark] = []
                landmark_groups[landmark].append(file)
                break
    return landmark_groups