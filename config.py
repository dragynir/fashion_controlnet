from dataclasses import dataclass
from typing import List, Optional

from dataset import FashionDataset


@dataclass
class TrainingConfig:
    """Training config for SDXL controlnet."""

    pretrained_model_name_or_path: str
    pretrained_vae_model_name_or_path: str
    output_dir: str
    dataset_name: Optional[str]
    train_data_dir: str

    resume_from_checkpoint: Optional[str]

    mixed_precision: str
    resolution: int
    learning_rate: float
    max_train_steps: int
    train_batch_size: int
    gradient_accumulation_steps: int

    checkpoints_total_limit: int
    gradient_checkpointing: bool
    use_8bit_adam: bool
    enable_xformers_memory_efficient_attention: bool

    validation_image: List[str]
    validation_prompt: List[str]
    validation_steps: int
    report_to: str

    seed: int


training_config = TrainingConfig(
    pretrained_model_name_or_path='/pub/home/korostelev/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b',
    pretrained_vae_model_name_or_path='/pub/home/korostelev/.cache/huggingface/hub/sdxl-vae-fp16-fix',
    output_dir='./fashion_training',
    dataset_name=None,
    train_data_dir='/pub/home/korostelev/data/diffusion/train',
    mixed_precision='fp16',
    resolution=512,
    resume_from_checkpoint=None,  # 'latest',
    learning_rate=1e-5,
    max_train_steps=45000,
    train_batch_size=1,
    gradient_accumulation_steps=4,
    checkpoints_total_limit=1,
    gradient_checkpointing=True,
    use_8bit_adam=True,
    enable_xformers_memory_efficient_attention=False,
    validation_image=['./fashion_validation/test1.jpg'],
    validation_prompt=['woman in black and white big scarf, standing'],
    validation_steps=100,
    report_to='tensorboard',
    seed=42,
)

current_dataset = FashionDataset(
    image_dir='/pub/home/korostelev/data/diffusion/train',
    df_path='/pub/home/korostelev/data/diffusion/train.csv',
    attributes_path='/pub/home/korostelev/data/diffusion/label_descriptions.json',
    caption_path='/pub/home/korostelev/data/diffusion/caption.csv',
    resolution=512,
    max_images=None,
    condition_from_disk=True,
)

# CUDA_VISIBLE_DEVICES=1 nohup python train_controlnet_sdxl.py &
# CUDA_VISIBLE_DEVICES=0 accelerate launch train_controlnet_sdxl.py
# tensorboard --logdir=./training_logs/logs --port 7860
