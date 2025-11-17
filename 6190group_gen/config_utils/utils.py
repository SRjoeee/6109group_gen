import argparse
from itertools import product
import os
import yaml
import re
def load_all_yaml_entries(yaml_path):
    """
    Loads a YAML file containing multiple documents (separated by ---).
    Returns a list of dicts, each with 'init_img' and 'target_prompts'.
    """
    with open(yaml_path, "r") as f:
        entries = list(yaml.safe_load_all(f))
    return entries


def extract_all_prompts(yaml_path):
    """
    Extracts all target prompts from a multi-entry YAML file.
    Returns a flat list of all prompts.
    """
    entries = load_all_yaml_entries(yaml_path)
    prompts = []
    for entry in entries:
        prompts.extend(entry.get("target_prompts", []))
    return prompts

TEMPLATES = ['a photo of ', 'an image of ', "a picture of "]

def get_real_prompts( object_name):
    return [t+object_name for t in TEMPLATES]

def find_prompts_by_image(yaml_path, image_path):
    """
    Finds the corresponding prompts for an image by matching the last subfolder
    and filename, e.g. 'penguin/sketch_8.jpg'.

    Args:
        yaml_path: path to the big YAML file.
        image_path: path or partial path to image (can be absolute or relative).
    Returns:
        list of target prompts (or [] if not found)
    """
    entries = load_all_yaml_entries(yaml_path)[0]
    # normalize key we want to match, e.g. 'penguin/sketch_8.jpg'
    parts = image_path.strip("/").split("/")
    key = "/".join(parts[-2:])  # last folder + filename

    for entry in entries:
        init_img = entry.get("init_img", "")
        if key in init_img:  # fuzzy match by last folder + filename
            return entry.get("target_prompts", [])
    return []

def find_prompts_by_objectfile(yaml_path, object_filename):
    """
    Finds the corresponding prompts for an image by matching the last subfolder
    and filename, e.g. 'penguin/sketch_8.jpg'.

    Args:
        yaml_path: path to the big YAML file.
        image_path: path or partial path to image (can be absolute or relative).
    Returns:
        list of target prompts (or [] if not found)
    """
    entries = load_all_yaml_entries(yaml_path)[0]
    # normalize key we want to match, e.g. 'penguin/sketch_8.jpg'
    object, filename = object_filename
    key= object+'/'+filename
    for entry in entries:
        init_img = entry.get("init_img", "")
        if key in init_img:  # fuzzy match by last folder + filename
            return entry.get("target_prompts", [])
    return []

def read_hyper_grids(filepath):
    """
    Read a YAML hyperparameter grid and return a list of all combinations
    as dictionaries.

    Example output:
    [{'scale': 0.1, 'seed': 1}, {'scale': 0.1, 'seed': 2}, ...]
    """
    with open(filepath, "r") as f:
        cfg = yaml.safe_load(f)
    param_grid = cfg["params"]

    keys = list(param_grid.keys())
    values_product = product(*param_grid.values())

    # create a dict for each combination
    combos = [dict(zip(keys, values)) for values in values_product]
    return combos

def update_args(base_args, update_dict):
    """Return a copy of base_args with only matching keys replaced."""
    new_args = argparse.Namespace(**vars(base_args))  # copy
    for k, v in update_dict.items():
        if hasattr(new_args, k):
            setattr(new_args, k, v)
    return new_args

def render_namespace_name(args_dict, keys=None, sep="_", float_precision=3):
    """
    Convert selected args from a Namespace into a short, filesystem-safe string.
    Example:
        Namespace(scale=0.1, seed=3, gamma=1.0)
        -> 'scale0.1_seed3_gamma1.0'
    """
    args_dict = args_dict.copy()
    if "kld_scale" in args_dict:
        del args_dict["kld_scale"]
    # pick only relevant keys if provided, else use all keys
    if keys is None:
        keys = sorted(args_dict.keys())

    parts = []
    for k in keys:
        if k not in args_dict:
            continue
        v = args_dict[k]
        # format floats neatly
        if isinstance(v, float):
            v = round(v, float_precision)
        v_str = str(v)
        short_k = k[:6]
        v_str = re.sub(r"[^\w\.-]", "", v_str)  # remove unsafe characters
        parts.append(f"{short_k}{v_str}")
    return sep.join(parts)
