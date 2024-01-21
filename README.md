# Fashion ControlNet SDXL #

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
TODO: hugginface space
TODO: colab using hugging face

This repo contains training code, inference code and pre-trained model for 
image generation pipeline based on [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) conditioned on [Clothes Segmentation](https://github.com/levindabhi/cloth-segmentation) using U2NET.

TODO: Video demo

Here clothes are parsed into 3 category: Upper body(red), Lower body(green) and Full body(blue)
condition
![Sample 000](assets/000.png)

# Technical details

* **Condition** : Clothes Segmentation mask, see details in [UNET repo](https://github.com/levindabhi/cloth-segmentation)

* **Image Dataset** : ControlNet is trained on 45k images [iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) dataset. 
For condition i use 3 categories (upper body, lower body and full body).  Inspect [dataset.py](examples/controlnet/dataset.py) for better understanding.

* **Image Caption** : Captions created with [clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator)
The CLIP Interrogator is a prompt engineering tool that combines OpenAI's CLIP and Salesforce's BLIP to optimize text prompts to match a given image.
You can create your own caption with [clip_caption.py](examples/controlnet/clip_caption.py)

* **Control Net**: 


# Inference



# Data and weights



# TODO Training

Добавить структуру датасета - как файлы для обучения лежат


Install clip
# Установка
# pip install clip-interrogator==0.5.4
# pip uninstall transformers
# pip install transformers==4.26.1

# conda activate clip-interrogator
# cd C:\Projects\FireBall\ControlNetMain
# python tutorial_clip_caption.py

