from src.unet.network import U2NET

import os
from PIL import Image
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('----No checkpoints at given path----')
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print('----checkpoints loaded from path: {}----'.format(checkpoint_path))
    return model


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, float)
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert 'Please set proper channels! Normlization implemented only for 1, 3 and 18'


def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)


def generate_mask(input_image, net, palette=None, device='cpu'):

    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    # Save final cloth segmentations
    mask = output_arr[0].astype(np.uint8)

    if not palette:
        return mask

    mask_image_palette = Image.fromarray(mask, mode='P')
    mask_image_palette.putpalette(palette)
    mask_image_palette = mask_image_palette.resize(img_size, Image.BICUBIC)
    return mask, mask_image_palette


def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


def main(args):

    device = 'cuda:0' if args.cuda else 'cpu'

    model = load_seg_model(args.checkpoint_path, device=device)

    palette = get_palette(4)

    img = Image.open(args.image).convert('RGB')

    mask, mask_image_palette = generate_mask(
        img, net=model, palette=palette, device=device
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Help to set arguments for Cloth Segmentation.'
    )
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument(
        '--cuda', action='store_true', help='Enable CUDA (default: False)'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='../models/cloth_segm.pth',
        help='Path to the checkpoint file',
    )
    args = parser.parse_args()

    args.image = (
        '/pub/home/korostelev/data/diffusion/test/804a460e4bd0d666d51e84adc70f5490.jpg',
    )
    main(args)
