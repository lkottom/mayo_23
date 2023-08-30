import torch
import torchvision.transforms as transforms
from PIL import Image

def extract_features(image_path, model):
    img = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img_tensor)

    return features[0]