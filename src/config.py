import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'image_retrieval')
BASE_DB_DIR = os.path.join(DATA_DIR, 'base', 'BJTU')
QUERY_DIR = os.path.join(DATA_DIR, 'query')
DETECTION_DATA_DIR = os.path.join(BASE_DIR, 'object_detection', 'data')

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
RETRIEVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'retrieval')
DETECTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'detection')
METRICS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'metrics')

MODEL_DIR = os.path.join(BASE_DIR, 'models')

LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]

TOP_K_VALUES = [20, 40, 60]

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

FEATURE_DIM = 2048

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RETRIEVAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)