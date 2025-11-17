import lpips
import torch
from PIL import Image
from torchvision import transforms

# Initialize LPIPS model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS(net='alex').to(device)

def get_LPIPS(img1, img2):
    # Load images
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    # Compute LPIPS distance
    with torch.no_grad():
        distance = loss_fn(img1, img2)
        return distance.item()
