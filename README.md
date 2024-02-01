# Fashion ControlNet SDXL #

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/dragynir/fashion_controlnet)


This repo contains training code, inference code and pre-trained model for 
image generation pipeline based on [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) conditioned on [Clothes Segmentation](https://github.com/levindabhi/cloth-segmentation) using U2NET.


| Input image                                | Mask                                                    | Generated                                                    |
|--------------------------------------------|---------------------------------------------------------|--------------------------------------------------------------|
| <img src="assets/input_1.jpg" width="512"> | <img src="assets/compare/mask/image_1.png" width="512"> | <img src="assets/compare/condition/image_1.png" width="512"> |

Prompt: `a woman wearing a white top and jeans with a high waist and a high rise heming hem is standing in front of a white background, Christian Hilfgott Brand, waist up, a photocopy, neoclassicism, ultra quality, sharp focus
`

# Inference

You can find inference script in [inference.py](src/inference.py). Run it with prepared mask or with image to extract mask first.

Also, check out huggingface space app [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/dragynir/fashion_controlnet)

You can clone hugging face space, or download it into your machine with gpu. Install requirements.txt and run locally.

# Weights

Trained ControlNet weights for fashion is available on huggingface [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/dragynir/fashion_controlnet/tree/main/weights) (5 GB)

# Technical details

* **Condition** : Clothes Segmentation mask, see details in [UNET repo](https://github.com/levindabhi/cloth-segmentation)

* **Image Dataset** : ControlNet was trained on 45k images [iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) dataset. 
For condition i use 3 categories (upper body, lower body and full body).  Inspect [dataset.py](examples/controlnet/dataset.py) for better understanding.

* **Image Caption** : Captions were created with [clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator)
The CLIP Interrogator is a prompt engineering tool that combines OpenAI's CLIP and Salesforce's BLIP to optimize text prompts to match a given image.
You can create your own caption with [clip_caption.py](examples/controlnet/clip_caption.py)

* **Control Net**:  Training script is based on diffusers script [train_controlnet_sdxl.py](https://github.com/huggingface/diffusers/tree/main/examples/controlnet)
You can find changed configuration parameters at [config.py](src/config.py).

# Major Changes in Training/Inference

1) SDXL's VAE is known to suffer from numerical instability issues.
So i used [this one](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) as recomended in diffusers README_sdxl.md.
With default one i have got NaNs in fp16 training.


2) Added `compute_adaptive_hw` function that prepare `crop_top_left` and `original_image_size` to pass into Unet.
This helps to match controlnet with original controlnet training, see Micro-Conditioning in original [SDXL paper](https://arxiv.org/pdf/2307.01952.pdf).


3) Adaptive image resizing to keep original image aspect ratio during inference.



# Training

### Installation

- Python >= 3.9
- `pip install -r requirements.txt`


### Data Preparation

1) Download and extract [imaterialist-fashion-2019](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) dataset to [data/](data) folder.
2) Run [clip_caption.py](examples/controlnet/clip_caption.py) to extract prompts from images or use precomputed [file: TODO upload caption.csv to hugginface data/ folder]().
3) Then run [dataset.py](src/dataset.py) to generate and save mask in `.png` format to disc.
4) Validation masks are already in validation [folder](data/validation).

#### The final directory structure should be

    .data/
    ├── train/                    # train images   
    │   ├── image1.jpg          
    │   ├── image2.jpg
    ├── train_condition_images/  # train images masks
    │   ├── image1.png          
    │   ├── image2.png
    ├── test/   
    ├── validation/   
    ├── train.csv
    ├── caption.csv              # caption created by clip_caption.py
    └── label_descriptions.json

### Run training

```bash
accelerate launch src/train_controlnet_sdxl.py
```


### Compare conditioned and no condition generations with the same seed.

| Mask                                                    | Condition                                                    | No condition                                                    |
|---------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------------|
| <img src="assets/compare/mask/image_1.png" width="512"> | <img src="assets/compare/condition/image_1.png" width="512"> | <img src="assets/compare/no_condition/image_1.png" width="512"> |
| <img src="assets/compare/mask/image_2.png" width="512"> | <img src="assets/compare/condition/image_2.png" width="512"> | <img src="assets/compare/no_condition/image_2.png" width="512"> |
| <img src="assets/compare/mask/image_4.png" width="512"> | <img src="assets/compare/condition/image_4.png" width="512"> | <img src="assets/compare/no_condition/image_4.png" width="512"> |



[//]: # ()
[//]: # (Mask            |                 Condition                 |                 No Condition                 )

[//]: # (:-----------------------------------------:|:-----------------------------------------:|:-----------------------------------------:)

[//]: # (![]&#40;assets/compare/mask/image_1.png&#41; | ![]&#40;assets/compare/condition/image_1.png&#41; |  ![]&#40;assets/compare/mask/image_1.png&#41;)

[//]: # (![]&#40;assets/compare/mask/image_2.png&#41; | ![]&#40;assets/compare/condition/image_2.png&#41; |  ![]&#40;assets/compare/no_condition/image_2.png&#41;)

[//]: # (![]&#40;assets/compare/mask/image_4.png&#41; | ![]&#40;assets/compare/condition/image_4.png&#41; |  ![]&#40;assets/compare/no_condition/image_4.png&#41;)


# TODO Real TODO list (training 1024, training as SDXL)
- [x] прокинуть параметры в демке (seed и т д)
- [x] добавить caption.csv в data/ - в этом репосе т к в huggingface нельзя
- [x] сделать colab ноутбук https://colab.research.google.com/drive/1-SbSkGveuC7ZSohz5nyMib9Iot_5os4c?usp=sharing
- [x] прогнать скрипт на 100-200 примеров генерации (prompt, generated with condition, generated without condition(original sdxl))
  - https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 как запускать без condition
  - добавить примеры в Readme.md
- [x] закончить пункт Run training
- [x] вставить condition/ no condition примеры
- - [x] обновить видео (пример входной: https://lexica.art/prompt/adfeb3a7-1e74-4809-a681-a8555a7b5d82)
  - [ ] сохранил пока локально - надо будет его отредачить - чтобы было короткое
- [ ] Выровнять таблицу с изображениями
- [ ] обучить более сильную модель
- [ ] обновить под нову модель, переписать скрипт последнего обучения в этой репе
- [ ] потестить колаб на открытой репе
- [ ] Проверить что все ссылки в Readme.md работают!
- [ ] Check English spelling
