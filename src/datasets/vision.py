import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from PIL import Image
from sklearn.model_selection import train_test_split

import shared

logger = shared.fetch_main_logger()

from .utils import BaseDataset


def preprocessUTK(data_path, split):

    # -------------------------------- Download the dataset --------------------------------
    # if not os.path.exists(os.path.join(data_path, "utkface-new.zip")):
    os.makedirs(os.path.join(data_path), exist_ok=True)

    # Define the path for the Kaggle directory
    kaggle_dir = os.path.expanduser("~/.kaggle")

    # Create the Kaggle directory if it doesn't exist
    os.makedirs(kaggle_dir, exist_ok=True)

    # Create the kaggle.json file with your credentials
    kaggle_credentials = {
        "username": os.environ["KAGGLE_USERNAME"],
        "key": os.environ["KAGGLE_KEY"],
    }

    # Write the credentials to kaggle.json
    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        json.dump(kaggle_credentials, f)

    # Set permissions for kaggle.json (Linux/Mac)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

    # Command to download the dataset. Skip if the zip file already exists
    print("Downloading the dataset...")

    current_dir = os.getcwd()
    os.chdir(data_path)
    command = ["kaggle", "datasets", "download", "-d", "jangedoo/utkface-new"]
    subprocess.run(command)

    # Unzip the dataset
    command = ["unzip", "utkface-new.zip"]
    subprocess.run(command)

    # Return to the original directory
    os.chdir(current_dir)

    # Set the data path to the UTKFace directory
    image_path = os.path.join(data_path, "utkface_aligned_cropped/UTKFace/")

    # ----------------------------- Preprocess the dataset --------------------------------
    print("Preprocessing the dataset...")
    # Following the Kaggle pipeline
    images = []
    ages = []
    for i in os.listdir(image_path):
        splits = i.split("_")
        if "idx" not in splits[1]:
            ages.append(int(splits[0]))
            with Image.open(image_path + i).convert("RGB") as image:
                images.append(np.asarray(image.resize((200, 200), Image.LANCZOS), dtype=np.float32))

    images = pd.Series(list(images), name="Images")
    ages = pd.Series(list(ages), name="Ages")

    df = pd.concat([images, ages], axis=1)
    del images, ages

    print("Dataset shape:", df.shape)

    # kaggle preprocessing pipeline, subsample kids under 4
    under4s = df[df["Ages"] <= 4].sample(frac=0.3)

    df = df[df["Ages"] > 4]
    df = pd.concat([df, under4s], ignore_index=True)
    del under4s

    print("Dataset shape after subsampling under 4s:", df.shape)

    images = np.stack(df["Images"].values)
    images = np.transpose(images, (0, 3, 1, 2))
    ages = df["Ages"].values.astype("float32")
    del df

    train_idx, test_idx = train_test_split(np.arange(len(ages)), test_size=0.2)

    train_images = images[train_idx]
    train_ages = ages[train_idx]
    # subsample to overfit
    train_images = train_images[: len(train_images) // 4]
    train_ages = train_ages[: len(train_ages) // 4]

    test_images = images[test_idx]
    test_ages = ages[test_idx]

    age_mean = 35.43
    age_std = 18.77

    train_ages = (train_ages - age_mean) / age_std
    test_ages = (test_ages - age_mean) / age_std

    print(f"Preprocessing done. Final dataset shape for split {split}: {images.shape}")

    # Save the preprocessed dataset
    os.makedirs(os.path.join(data_path, "cleanUTKFace/"), exist_ok=True)
    np.save(os.path.join(data_path, f"cleanUTKFace/train_images.npy"), train_images)
    np.save(os.path.join(data_path, f"cleanUTKFace/train_ages.npy"), train_ages)
    np.save(os.path.join(data_path, f"cleanUTKFace/val_images.npy"), test_images)
    np.save(os.path.join(data_path, f"cleanUTKFace/val_ages.npy"), test_ages)

    if split == "train":
        return train_images, train_ages
    else:
        return test_images, test_ages


class CIFAR(BaseDataset):
    train_transforms: tv.transforms.Compose
    val_transforms: tv.transforms.Compose
    input_shape = (3, 32, 32)
    output_size: int

    def __init__(self, data_path: str, split: str):
        is_train = split == "train"
        transforms = self.train_transforms if is_train else self.val_transforms

        if data_path is None:
            data_path = os.path.join(os.environ["SLURM_TMPDIR"], f"cifar{self.output_size}")

        tv_dataset_class = getattr(tv.datasets, f"CIFAR{self.output_size}")
        self.dataset = tv_dataset_class(root=data_path, train=is_train, transform=transforms, download=True)
        logger.info(f"CIFAR{self.output_size} dataset {split} split contains {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)


class CIFAR10(CIFAR):
    train_transforms = tv.transforms.Compose(
        [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    val_transforms = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    output_size = 10


class UTKFace(BaseDataset):
    train_transforms = tv.transforms.Compose(
        [tv.transforms.Normalize((150.09727557, 114.61889488, 97.81777823), (66.16906156, 58.67345005, 57.35997734))]
    )
    val_transforms = train_transforms
    input_shape = (3, 200, 200)
    output_size = 1

    def __init__(self, data_path, split):

        self.split = split

        try:
            new_path = os.path.join(data_path, "cleanUTKFace/")
            X = np.load(os.path.join(new_path, f"{split}_images.npy"))
            y = np.load(os.path.join(new_path, f"{split}_ages.npy"))
        except:
            X, y = preprocessUTK(data_path=data_path, split=split)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        x = self.train_transforms(x) if self.split == "train" else self.val_transforms(x)

        y = self.y[index].unsqueeze(0)

        return x, y
