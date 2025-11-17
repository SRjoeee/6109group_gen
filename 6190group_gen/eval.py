import os
import numpy as np
import torch
from PIL import Image
from metrics import *
#import pyiqa
from torchvision import transforms
from config_utils.utils import *
import sys
prompt_temp = 'a photo of '
import re
def extract_prompt_idx(filename):
    """
    Extract the prompt index from filenames like:
    'sdffsf_dfsfsf_Sdffd_prompt3.png' → 3
    Returns int or None if not found.
    """
    base = os.path.basename(filename)
    match = re.search(r'_prompt(\d+)', base, re.IGNORECASE)
    return int(match.group(1)) if match else None

def load_images_from_folder(folder,prompt_list=''):
    images = {}
    filenames = sorted(os.listdir(folder))

    for subfolder in filenames:
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for filename in sorted(os.listdir(subfolder_path)):
            img_path = os.path.join(subfolder_path, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                if filename.split('_')[0]==subfolder:
                    key2 = filename.split('_')[1] + '_' + filename.split('_')[2].split('.')[0]
                else:
                    key2= filename.split('_')[0] + '_' + filename.split('_')[1].split('.')[0]
                key = (subfolder, key2)  # Match based on 'art_5' pattern
                print(key)
                if prompt_list:
                    prompt_idx = extract_prompt_idx(filename)
                    prompts = find_prompts_by_objectfile(prompt_list,key)
                    if key not in images:
                        images[key] = []
                    images[key].append((img,prompts[prompt_idx]))
                else:
                    images[key] = img
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return images

'''
def niqe(img):
    # Load image and convert to tensor [0,1]
    tf = transforms.Compose([
        transforms.ToTensor()
    ])
    iqa_metric = pyiqa.create_metric('niqe')
    img = tf(img).unsqueeze(0)  # shape [1,3,H,W]

    # Compute NIQE
    niqe_score = iqa_metric(img)
    print(f"NIQE: {niqe_score.item():.3f}")
    return niqe_score.item()
'''

def compute_metrics_singleprompt(imgs_pred, imgs_gt):
    clip_scores, ssim_scores, lpips_scores = [], [], []

    for key in imgs_pred:
        if key in imgs_gt:
            prompt = prompt_temp + key[0]
            clip_s = get_clip(imgs_pred[key], prompt)
            clip_scores.append(clip_s)
            lpips_s = get_LPIPS(imgs_pred[key], imgs_gt[key])
            lpips_scores.append(lpips_s)
            ssim_scores.append(get_ssim(imgs_pred[key], imgs_gt[key]))
            #niqes=niqe(imgs_pred[key])
            #niqe_scores.append(niqe(imgs_pred[key]))
            print(f"{key}: clip:{clip_s} lpips:{lpips_s} ssim:{ssim_scores[-1]}")
    print(f"Mean clip: {np.mean(clip_scores):.4f}")
    print(f"Mean LPIPS: {np.mean(lpips_scores):.4f}")
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")
    return np.mean(clip_scores), np.mean(lpips_scores), np.mean(ssim_scores)


def compute_metrics_5prompt(imgs_pred, imgs_gt,imgs_cond):
    clip_scores, ssim_scores, lpips_scores = [], [], []
    for key in imgs_pred:
        if key in imgs_cond:
            all_prompt_imgs = imgs_pred[key]
            for img_prompt in all_prompt_imgs:
                img_pred,prompt =img_prompt
                clip_s = get_clip(img_pred, prompt)
                clip_scores.append(clip_s)
                lpips_s = get_LPIPS(img_pred, imgs_cond[key])#lpips between condi and pre
                lpips_scores.append(lpips_s)
                ssim_scores.append(get_ssim(img_pred, imgs_gt[key]))# ssim between gt and pre
            #niqes=niqe(imgs_pred[key])
            #niqe_scores.append(niqe(imgs_pred[key]))
                print(f"{key}{prompt}: clip:{clip_s} lpips:{lpips_s} ssim:{ssim_scores[-1]}")
    print(f"Mean clip: {np.mean(clip_scores):.4f}")
    print(f"Mean LPIPS: {np.mean(lpips_scores):.4f}")
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")
    return np.mean(clip_scores), np.mean(lpips_scores), np.mean(ssim_scores)

def main(gt_folder, condition_folder, pred_folder,log_file,prompt_list=''):

    print("Loading Generated images...")
    imgs_pred = load_images_from_folder(pred_folder, prompt_list)
    imgs_cond = load_images_from_folder(condition_folder)
    imgs_gt = load_images_from_folder(gt_folder)


    with open(log_file, "w") as f:
        # Save original stdout
        original_stdout = sys.stdout
        # Redirect
        sys.stdout = f
        try:
            if prompt_list:
                compute_metrics_5prompt(imgs_pred, imgs_gt,imgs_cond)
            else:
                compute_metrics_singleprompt(imgs_pred, imgs_gt)
        finally:
            # Restore stdout so you can print normally after
            sys.stdout = original_stdout
    print(f"Logs saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run eval.")
    parser.add_argument(
        "--modality",
        type=str,
        default="depth",
        help=""
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="",
        default="ours"
    )
    parser.add_argument('--prompt_list', type=str, default='data/imnetr-ti2i.yaml')
    args = parser.parse_args()
    outdir = f'eval/{args.baseline}_{args.modality}.txt'
    gt_folder = f"datasets/ti2i/GT"
    condition_folder=f"datasets/ti2i/{args.modality}"
    pred_folder = f"eval/{args.baseline}/{args.modality}"
    main(gt_folder, condition_folder,pred_folder, outdir,args.prompt_list)