import os

from diffusers.utils import load_image
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from config_utils.utils import *
from PIL import Image

def load_images_from_folder(folder):
    images = {}
    filenames = sorted(os.listdir(folder))
    for subfolder in filenames:
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        #if subfolder != 'panda':
        #    continue
        for filename in sorted(os.listdir(subfolder_path)):
            img_path = os.path.join(subfolder_path, filename)
            try:
                short_name = filename.replace(subfolder+'_', '')
                key = (subfolder, short_name.split('_')[0] + '_' + short_name.split('_')[1].replace('avg','').replace('.jpg',''))  # Match based on 'art_5' pattern
                images[key] = img_path
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return images


def main(strength, modality,ti2i=False):
    control_inputs = load_images_from_folder(f'data/{modality}')
    generator = torch.Generator(device="cuda").manual_seed(0)

    MODEL_PATH = os.getenv("MODEL_PATH", "stable-diffusion-v1-5/stable-diffusion-v1-5")

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)


    #quantize(pipeline.text_encoder, weights=qint4, exclude="proj_out")
    #freeze(pipeline.text_encoder)

    #quantize(pipeline.text_encoder_2, weights=qint4, exclude="proj_out")
    #freeze(pipeline.text_encoder_2)

    #quantize(pipeline.transformer, weights=qint8, exclude="proj_out")
    #freeze(pipeline.transformer)

    pipeline = pipeline.to("cuda")
    subfoldname="5prompt" if ti2i else "real"

    output_dir = f'./output/{subfoldname}/sdedit{strength}'
    prompt_path = './data/imnetr-ti2i.yaml'
    os.makedirs(output_dir, exist_ok=True)
    for key in control_inputs:
        obj_name, filename = key
        control_image = load_image(control_inputs[key])
        if ti2i:
            prompts = find_prompts_by_image(prompt_path, control_inputs[key])
        else:
            prompts = get_real_prompts(obj_name)
        for idx, prompt in enumerate(prompts):
            image = pipeline(prompt,
                             image=control_image,
                             guidance_scale=3.5,
                             generator=generator,
                             height=512, width=512,
                             num_inference_steps=28,
                             strength=strength
                             ).images[0]

            os.makedirs(os.path.join(output_dir, modality,obj_name), exist_ok=True)
            image.save(os.path.join(output_dir, modality,obj_name, obj_name+'_'+filename+f'_prompt{idx}.png'))

if __name__ == '__main__':
    main(0.85, "depth")
    main(0.85, "canny")
    main(0.85, "normal")
    main(0.85, "hed")
    main(0.75, "depth")
    main(0.75, "canny")
    main(0.75, "normal")
    main(0.75, "hed") # 3h
    main(0.85, "canny",ti2i=True)
    main(0.85, "normal",ti2i=True)
    main(0.85, "hed",ti2i=True)
    main(0.75, "depth",ti2i=True)
    main(0.75, "canny",ti2i=True)
    main(0.75, "normal",ti2i=True)
    main(0.75, "hed",ti2i=True) # 4.5h