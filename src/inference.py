import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
import torch

from unet.predictor import generate_mask, load_seg_model


def adaptive_resize(res_image, target_image_size=512, max_image_size=768, divisible=64):

    assert target_image_size % divisible == 0
    assert max_image_size % divisible == 0
    assert max_image_size >= target_image_size

    width, height = res_image.size
    aspect_ratio = width / height

    if height > width:
        new_width = target_image_size
        new_height = new_width / aspect_ratio
        new_height = (new_height // divisible) * divisible
        new_height = int(min(new_height, max_image_size))
    else:
        new_height = target_image_size
        new_width = new_height / aspect_ratio
        new_width = (new_width // divisible) * divisible
        new_width = int(min(new_width, max_image_size))

    return res_image.resize((new_width, new_height))


device = torch.device('cuda')

# models--stabilityai--stable-diffusion-xl-base-1.0 path
base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

# controlnet checkpoint path
controlnet_path = './fashion_training/checkpoint-45000/controlnet'

# vae model path
vae_path = 'madebyollin/sdxl-vae-fp16-fix'

# pretrained segmentation model path
segmentation_model_path = './weights/cloth_segm.pth'

# directory to write output image
output_dir = './outputs'

# Example: load image, extract segmentation mask with unet and use it as condition
control_image = load_image(
    './data/train/7fb8e9647ebb8a25d6dc9ab837a36a49.jpg',
)
prompt = 'Maya, Brazilian, Tan skin, gorgeous middle aged woman, white background, short hair, , standing, facing front. HD, ((((((((((((show whole body)))))), no shadows, cartoon , wearing a black mini skirt, full frame , 35mm WIDE ANGLE SHOT, 80s STYLES, MAKE IMAGE INTO 3D , coloring book, vector, normal chest'
is_mask_ready = False


# Example: load segmentation mask and use it as condition
# control_image = load_image('./data/train_condition_images/7fbe43fc176a8719a42f7e6bb47e4819.png')
# prompt = 'Maya, Brazilian, Tan skin, gorgeous middle aged woman, white background, short hair, full face, standing, facing front. HD, ((((((((((((show whole body)))))), no shadows, cartoon , wearing a black mini skirt, full frame , 35mm WIDE ANGLE SHOT, 80s STYLES, MAKE IMAGE INTO 3D , coloring book, vector, normal chest'
# is_mask_ready = True
# controlnet_input_size = 512


if not is_mask_ready:
    segmentation_model = load_seg_model(segmentation_model_path, device=device)
    detected_map_original = generate_mask(
        control_image, segmentation_model, device=device
    )
    detected_map = np.stack(
        [detected_map_original, detected_map_original, detected_map_original], axis=-1
    )

    source = (detected_map.astype(np.float32) / 3.0) * 255
    source = np.clip(source, 0, 255)
    control_image = Image.fromarray(source.astype('uint8'), 'RGB')


control_image = adaptive_resize(control_image, target_image_size=512, max_image_size=768, divisible=64)

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()

# memory optimization.
pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=40, generator=generator, image=control_image
).images[0]


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

cnt_image = get_concat_h(control_image, image).save(os.path.join(output_dir, 'output.png'))
plt.imshow(cnt_image)
plt.show()
