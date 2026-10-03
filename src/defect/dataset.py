import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .constants import MEAN, STD
from .preprocess import pil_to_array, resize_pil


def build_train_aug(a: dict):
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(a["rotation"], interpolation=T.InterpolationMode.BILINEAR),
        T.RandomAffine(0, translate=(a["translate"], a["translate"]),
                       scale=(1 - a["scale"], 1 + a["scale"])),
        T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
    ])


class CastingDataset(Dataset):
    def __init__(self, csv_path, size, train=False, aug=None, mean=MEAN, std=STD):
        df = pd.read_csv(csv_path)
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.size, self.mean, self.std = size, mean, std
        self.tf = build_train_aug(aug) if (train and aug) else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = resize_pil(im, self.size)
        if self.tf is not None:
            img = self.tf(img)
        x = torch.from_numpy(pil_to_array(img, self.mean, self.std))
        return x, torch.tensor(float(self.labels[i]))
