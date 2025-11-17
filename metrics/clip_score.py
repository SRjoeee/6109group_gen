import clip
import torch
from PIL import Image

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32")
model = model.to(device)

def get_clip(img, prompt):
    # Prepare the inputs
    img = preprocess(img).unsqueeze(0).to(device)

    text = clip.tokenize([prompt]).to(device)

    # Calculate feature vectors
    with torch.no_grad():
        image_features = model.encode_image(img)
        text_features = model.encode_text(text)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    return similarity.item()