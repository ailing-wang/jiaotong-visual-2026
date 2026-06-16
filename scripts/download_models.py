import os
import sys
import urllib.request

def download_file(url, save_path):
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def main():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    east_model_url = "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.pb?dl=1"
    east_model_path = os.path.join(model_dir, 'frozen_east_text_detection.pb')
    
    print("=== Downloading Pre-trained Models ===")
    
    if not os.path.exists(east_model_path):
        success = download_file(east_model_url, east_model_path)
        if not success:
            print("EAST model download failed. Will use MSER detector as fallback.")
    else:
        print("EAST model already exists.")
    
    print("\nModel download completed!")

if __name__ == '__main__':
    main()