import os, subprocess

import pandas as pd
from clip_interrogator import Config, Interrogator
import csv
import os
from IPython.display import clear_output, display
from PIL import Image
from tqdm import tqdm

def image_to_prompt(ci, image, mode):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)

def run_clip(folder_path, prompt_mode):

    # Доступные модели
    caption_model_name = 'blip-large'  # @param ["blip-base", "blip-large", "git-large-coco"]
    clip_model_name = 'ViT-L-14/openai'  # @param ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]

    config = Config()
    config.clip_model_name = clip_model_name
    config.caption_model_name = caption_model_name
    ci = Interrogator(config)
    ci.config.quiet = True

    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')] if os.path.exists(
        folder_path) else []
    prompts = []
    proc_files = []

    print('Start clip captioning...')

    for idx, file_name in enumerate(tqdm(files)):
        image = Image.open(os.path.join(folder_path, file_name)).convert('RGB')
        prompt = image_to_prompt(ci, image, prompt_mode)
        prompts.append(prompt)
        proc_files.append(file_name)

        if idx % 100 == 0:
            df = pd.DataFrame()
            df[f'{prompt_mode}_clip_prompts'] = prompts
            df['ImageId'] = proc_files
            df.to_csv('./training/iMaterialist(Fashion)/caption_2.csv', index=False)

    df = pd.DataFrame()
    df[f'{prompt_mode}_clip_prompts'] = prompts
    df['ImageId'] = proc_files
    df.to_csv('./training/iMaterialist(Fashion)/caption_2.csv', index=False)


if __name__ == '__main__':
    # Установка
    # pip install clip-interrogator==0.5.4
    # pip uninstall transformers
    # pip install transformers==4.26.1

    # conda activate clip-interrogator
    # cd C:\Projects\FireBall\ControlNetMain
    # python tutorial_clip_caption.py

    # Прогон всех изображений около 6 часов на fast mode
    folder_path = "./training/iMaterialist(Fashion)/train"
    prompt_mode = 'fast'  # @param ["best","fast","classic","negative"]
    run_clip(folder_path, prompt_mode)
