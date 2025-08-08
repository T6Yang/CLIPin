from math import ceil
import os
from pathlib import Path
import json
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode, CenterCrop, RandomHorizontalFlip, ColorJitter, RandomResizedCrop
import random

import torch
from clip.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class CocoCaptions(Dataset):
    def __init__(self, root, annFile, use_augment=False, resolution=224):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        if self.use_augment:
            transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                RandomResizedCrop(resolution, scale=(0.8, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
                RandomHorizontalFlip(0.5),
                ColorJitter(0.1),
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            transform = Compose([
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop((resolution, resolution)),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        texts = [ann['caption'] for ann in anns]
        if len(texts) == 1:
            text1 = tokenize(texts[0])[0]
            text2 = text1
        else:
            txt_idxs = random.sample(range(0, len(texts)), 2)
            text1 = tokenize(texts[txt_idxs[0]])[0]
            # text2 = tokenize(texts[txt_idxs[1]])[0]
            text2 = text1
        eos_indice1 = len(torch.where(text1 > 0)[0])
        eos_indice2 = len(torch.where(text2 > 0)[0])

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2, (text1, text2), (eos_indice1, eos_indice2), index

    def __len__(self):
        return len(self.ids)


def pad_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


def get_data(dataset_name, split, batch_size, num_workers=0):
    if dataset_name == "COCO":
        if split == "train":
            dataset = CocoCaptions(
                root="/path/to/your/coco2014",
                annFile="/path/to/your/coco2014",
                use_augment=True,
                resolution=224
            )
        elif split == "val":
            dataset = CocoCaptions(
                root="/path/to/your/coco2014",
                annFile="/path/to/your/coco2014",
                use_augment=False,
                resolution=224
            )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
        drop_last=False
    )

    return dataloader
