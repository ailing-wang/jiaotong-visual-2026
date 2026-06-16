import os
import json
import pickle
from PIL import Image

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_image(image_path):
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_image_files(directory, extensions=None):
    if extensions is None:
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)

def get_landmark_from_filename(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    for landmark in ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]:
        if basename.startswith(landmark + '-'):
            return landmark
    return None