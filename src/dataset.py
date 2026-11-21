"""
dataset.py - PyTorch Dataset & DataLoader utilities for FaceCraft.

Handles image loading, center cropping to square aspect ratio, resizing to 64x64,
pixel normalization to [-1, 1], and caption tokenization.
"""

import os
from typing import Optional, Callable, Tuple, List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def get_default_transforms(image_size: int = 64):
    """
    Standard preprocessing pipeline:
    1. Center crop to square (178x178 for CelebA)
    2. Resize to (image_size, image_size), e.g. 64x64
    3. Convert to Tensor in [0, 1]
    4. Normalize to [-1, 1] for Diffusion & GAN stability
    """
    return T.Compose([
        T.CenterCrop(178),
        T.Resize((image_size, image_size), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class FaceCraftDataset(Dataset):
    """
    Unified Dataset for FaceCraft.
    Pairs 64x64 face image tensors with tokenized text captions.
    """
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        images_dir: str,
        tokenizer: Optional[object] = None,
        transform: Optional[Callable] = None,
        max_seq_len: int = 32
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform if transform is not None else get_default_transforms()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.metadata.iloc[idx]
        image_name = str(row["image_id"])
        caption = str(row["caption"])

        # Locate image
        img_path = os.path.join(self.images_dir, image_name)
        if not os.path.exists(img_path):
            # Check without extension or with .jpg / .png
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = os.path.join(self.images_dir, f"{os.path.splitext(image_name)[0]}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            # Fallback placeholder image if file is missing
            image = Image.new("RGB", (64, 64), color=(128, 128, 128))

        image_tensor = self.transform(image)

        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(caption, max_length=self.max_seq_len)
            tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            tokens = torch.zeros(self.max_seq_len, dtype=torch.long)

        return image_tensor, tokens, caption


def create_dataloaders(
    metadata_df: pd.DataFrame,
    images_dir: str,
    tokenizer: Optional[object] = None,
    batch_size: int = 32,
    image_size: int = 64,
    num_workers: int = 0
):
    """
    Splits metadata into train/val/test and creates PyTorch DataLoaders.
    """
    transform = get_default_transforms(image_size=image_size)

    # Check if 'split' column exists (0: train, 1: val, 2: test)
    if "split" in metadata_df.columns:
        train_df = metadata_df[metadata_df["split"].isin([0, "train", "0"])]
        val_df = metadata_df[metadata_df["split"].isin([1, "val", "1"])]
        test_df = metadata_df[metadata_df["split"].isin([2, "test", "2"])]
    else:
        # 80/10/10 random split
        train_df = metadata_df.sample(frac=0.8, random_state=42)
        remaining = metadata_df.drop(train_df.index)
        val_df = remaining.sample(frac=0.5, random_state=42)
        test_df = remaining.drop(val_df.index)

    train_ds = FaceCraftDataset(train_df, images_dir, tokenizer, transform)
    val_ds = FaceCraftDataset(val_df, images_dir, tokenizer, transform)
    test_ds = FaceCraftDataset(test_df, images_dir, tokenizer, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
