from typing import Any, Dict, Optional, Tuple
import datasets

import json
import os

import numpy as np
import torch
from PIL import Image

import pandas as pd
from tqdm import tqdm


class ConditionImageDataset:
    """Dataset for huggingface dataset."""

    def __init__(
        self,
        dataset_path: str,
    ) -> None:
        self.dataset_path = dataset_path
        self.images_dir = dataset_path
        self.conditioning_images_dir = dataset_path
        self.metadata_path = os.path.join(dataset_path, 'train.jsonl')
        self.metadata = pd.read_json(self.metadata_path, lines=True)

    def generator(self) -> Any:
        """Generator for huggingface from_generator call."""

        for _, row in self.metadata.iterrows():

            text = row['text']

            image_path = row['image']
            image_path = os.path.join(self.images_dir, image_path)
            image = open(image_path, 'rb').read()

            conditioning_image_path = row['conditioning_image']
            conditioning_image_path = os.path.join(
                self.conditioning_images_dir, row['conditioning_image']
            )
            conditioning_image = open(conditioning_image_path, 'rb').read()

            yield {
                'text': text,
                'image': {
                    'path': image_path,
                    'bytes': image,
                },
                'conditioning_image': {
                    'path': conditioning_image_path,
                    'bytes': conditioning_image,
                },
            }

    def generate_dataset(self) -> datasets.Dataset:
        """Prepare huggingface dataset."""
        return datasets.Dataset.from_generator(
            self.generator,
            features=datasets.Features(
                {
                    'image': datasets.Image(),
                    'conditioning_image': datasets.Image(),
                    'text': datasets.Value('string'),
                },
            ),
        )

    def __len__(self) -> int:
        """Return length of dataset."""
        raise len(self.metadata)


class FashionDataset:
    """Image dataset for Controlnet training."""

    def __init__(
        self,
        image_dir: str,
        df_path: str,
        attributes_path: str,
        caption_path: str,
        max_images: Optional[int],
        condition_from_disk: bool,
    ):
        self.image_dir = image_dir
        self.df_path = df_path
        self.attributes_path = attributes_path
        self.caption_path = caption_path

        self.metadata = self.prepare_dataset(
            self.df_path, self.caption_path, self.attributes_path
        )

        if max_images:
            self.metadata = self.metadata.head(max_images)

        self.condition_images_dir = 'train_condition_images'
        self.condition_from_disk = condition_from_disk

    def __len__(self):
        """Return length of dataset."""
        return len(self.metadata)

    def generate_dataset(self) -> datasets.Dataset:
        """Prepare huggingface dataset."""
        return datasets.Dataset.from_generator(
            self.generator,
            features=datasets.Features(
                {
                    'image': datasets.Image(),
                    'conditioning_image': datasets.Image(),
                    'text': datasets.Value('string'),
                },
            ),
        )

    def generator(self) -> Dict[str, Any]:
        """Create single datapoint."""
        for _, row in self.metadata.iterrows():

            text = self.get_prompt(row)

            target = self.get_target(row)

            if self.condition_from_disk:
                condition = self.get_saved_control_mask(row)
            else:
                condition = self.create_control_mask(row)

            yield {
                'text': text,
                'image': target,
                'conditioning_image': condition,
            }

    def save_condition_images(self) -> None:
        """Save control masks to disk."""
        condition_dir = self.image_dir.replace('train', self.condition_images_dir)
        print(f'Saving condition images to {condition_dir}')

        os.makedirs(condition_dir, exist_ok=True)
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            condition = self.create_control_mask(row)
            image_path = os.path.join(condition_dir, row['ImageId']).replace(
                '.jpg', '.png'
            )
            condition.save(image_path)

    def get_saved_control_mask(self, item: pd.Series) -> Dict[str, Any]:
        """Load control mask from disk."""
        condition_dir = self.image_dir.replace('train', self.condition_images_dir)
        image_path = os.path.join(condition_dir, item['ImageId']).replace(
            '.jpg', '.png'
        )
        image_bytes = open(image_path, 'rb').read()
        return {
            'path': image_path,
            'bytes': image_bytes,
        }

    def create_control_mask(self, item: pd.Series) -> Image:
        """Create control mask from rle."""
        import matplotlib.pyplot as plt
        width, height = item['Width'], item['Height']

        mask = np.zeros(
            (len(item['EncodedPixels']), height, width), dtype=np.uint8
        )

        labels = []
        for m, (annotation, label) in enumerate(
            zip(item['EncodedPixels'], item['CategoryId'])
        ):
            sub_mask = self.rle_decode(annotation, (width, height))
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), height, width), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        final_label = np.zeros((height, width), dtype=np.uint8)
        first_channel = np.zeros((height, width), dtype=np.uint8)
        second_channel = np.zeros((height, width), dtype=np.uint8)
        third_channel = np.zeros((height, width), dtype=np.uint8)

        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in upperbody:
                first_channel += new_masks[i]
            elif labels[i] in lowerbody:
                second_channel += new_masks[i]
            elif labels[i] in wholebody:
                third_channel += new_masks[i]

        first_channel = (first_channel > 0).astype('uint8')
        second_channel = (second_channel > 0).astype('uint8')
        third_channel = (third_channel > 0).astype('uint8')

        final_label = first_channel + second_channel * 2 + third_channel * 3
        conflict_mask = (final_label <= 3).astype('uint8')
        source = (conflict_mask) * final_label + (1 - conflict_mask) * 1
        # оставляю маску как в https://github.com/levindabhi/cloth-segmentation/tree/main
        # чтобы можно было юзать предобученную сегму

        source = np.stack([
            (source == 1).astype(int) * 255,
            (source == 2).astype(int) * 255,
            (source == 3).astype(int) * 255,
        ], axis=-1)

        source = np.clip(source, 0, 255)

        source = Image.fromarray(source.astype('uint8'), 'RGB')

        return source

    def get_target(self, item: pd.Series) -> Dict[str, Any]:
        """Create target image (output)."""
        image_path = os.path.join(self.image_dir, item['ImageId'])
        image_bytes = open(image_path, 'rb').read()
        return {
            'path': image_path,
            'bytes': image_bytes,
        }

    def get_prompt(self, item: pd.Series) -> str:
        """Construct prompt from metadata."""
        return item['fast_clip_prompts']

    def rle_decode(self, mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
        """Decode mask from annotation.

        mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
        shape: (height,width) of array to return
        Returns numpy array according to the shape, 1 - mask, 0 - background
        """
        shape = (shape[1], shape[0])
        s = mask_rle.split()
        # gets starts & lengths 1d arrays
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        # gets ends 1d array
        ends = starts + lengths
        # creates blank mask image 1d array
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        # sets mark pixles
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        # reshape as a 2d mask image
        return img.reshape(shape).T  # Needed to align to RLE direction

    def prepare_dataset(
        self, df_path: str, caption_path: str, attributes_path: str,
    ) -> pd.DataFrame:
        """Create dataset from raw data."""

        label_description = open(attributes_path).read()
        image_info = json.loads(label_description)

        categories = pd.DataFrame(image_info['categories'])
        attributes = pd.DataFrame(image_info['attributes'])

        train_df = pd.read_csv(df_path)
        caption_df = pd.read_csv(caption_path)

        # find records with attributes
        train_df['hasAttributes'] = train_df.ClassId.apply(lambda x: x.find('_') > 0)

        # get main category
        train_df['CategoryId'] = train_df.ClassId.apply(
            lambda x: x.split('_')[0]
        ).astype(int)

        train_df = train_df.merge(categories, left_on='CategoryId', right_on='id')

        size_df = train_df.groupby('ImageId')[['Height', 'Width']].mean().reset_index()
        size_df = size_df.astype({'Height': 'int', 'Width': 'int'})

        image_df = (
            train_df.groupby('ImageId')[
                ['EncodedPixels', 'CategoryId', 'name', 'supercategory']
            ]
            .agg(lambda x: list(x))
            .reset_index()
        )
        image_df = image_df.merge(size_df, on='ImageId', how='left')

        # extract all available attributes and create separate table
        cat_attributes = []
        for i in train_df[train_df.hasAttributes].index:
            item = train_df.loc[i]
            xs = item.ClassId.split('_')
            for a in xs[1:]:
                cat_attributes.append(
                    {
                        'ImageId': item.ImageId,
                        'category': int(xs[0]),
                        'attribute': int(a),
                    }
                )
        cat_attributes = pd.DataFrame(cat_attributes)

        cat_attributes = cat_attributes.merge(
            categories, left_on='category', right_on='id'
        ).merge(
            attributes, left_on='attribute', right_on='id', suffixes=('', '_attribute')
        )

        cat_image_df = (
            cat_attributes.groupby('ImageId')[['name', 'name_attribute']]
            .agg(lambda x: list(x))
            .reset_index()
        )
        named_attributes = []
        for _, row in cat_image_df.iterrows():
            named_attributes.append(
                {k: v for k, v in zip(row['name'], row['name_attribute'])}
            )
        cat_image_df['named_attributes'] = named_attributes

        image_df = image_df.merge(
            cat_image_df[['ImageId', 'named_attributes']], on='ImageId', how='left'
        )
        image_df = image_df.merge(caption_df, on='ImageId', how='left')

        return image_df


if __name__ == '__main__':

    dataset = FashionDataset(
        image_dir='./data/train',
        df_path='./data/train.csv',
        attributes_path='./data/label_descriptions.json',
        caption_path='./data/caption.csv',
        max_images=None,
        condition_from_disk=False,
    )
    dataset.save_condition_images()
