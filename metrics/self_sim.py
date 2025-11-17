import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from torchvision.models import vision_transformer as vit
import matplotlib.pyplot as plt
import numpy as np
from .extractor import VitExtractor

# Load a pretrained DINO model (make sure you have the DINO implementation)
# For example purposes, using a Vision Transformer model here as a placeholder.
# Replace with DINO-specific model if necessary.
#https://github.com/omerbt/Splice/blob/master/keys_self_sim_pca.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_extractor = VitExtractor('dino_vitb8', device)

def get_ssim(img, target):
    img = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])(img).unsqueeze(0).to(device)

    target = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])(target).unsqueeze(0).to(device)
    # define the extractor
    dino_preprocess = T.Compose([
        T.Resize(224),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    with torch.no_grad():
        keys_ssim = vit_extractor.get_keys_self_sim_from_input(dino_preprocess(img), 11)
        target_keys_self_sim = vit_extractor.get_keys_self_sim_from_input(dino_preprocess(target), 11)
        loss = F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss.item()
