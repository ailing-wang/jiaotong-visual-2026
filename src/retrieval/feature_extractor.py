import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(FeatureExtractor, self).__init__()
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            self.feature_dim = 4096
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(image)
        feature = feature.view(feature.size(0), -1).cpu().numpy().flatten()
        feature = feature / np.linalg.norm(feature)
        return feature
    
    def extract_batch(self, dataset, batch_size=32):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        features = []
        paths = []
        
        with torch.no_grad():
            for images, image_paths in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                batch_features = self.model(images)
                batch_features = batch_features.view(batch_features.size(0), -1).cpu().numpy()
                batch_features = batch_features / np.linalg.norm(batch_features, axis=1, keepdims=True)
                features.extend(batch_features)
                paths.extend(image_paths)
        
        return np.array(features), paths