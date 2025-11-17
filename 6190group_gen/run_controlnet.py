import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
 "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.bfloat16
)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
  "stable-diffusion-v1-5/stable-diffusion-v1-5",
  controlnet=controlnet,
  torch_dtype=torch.bfloat16
).to("cuda")

canny_image = load_image("https://huggingface.com/datasets/huggingface/documentation-images/resolve/main/diffusers/canny-cat.png")
pipeline(
  "a photo of cat",
  image=canny_image,
        controlnet_conditioning_scale=0.7,
    num_inference_steps=50,
  width=512,
    height=512,
).images[0]