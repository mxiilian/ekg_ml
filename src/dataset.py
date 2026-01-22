import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class EKGDataset(Dataset):
    """Dataset for loading EKG image pairs (input images 0002-0015 -> target image 0001)"""

    def __init__(self, data_dir, limit, transform=None):
        """
        Args:
            data_dir: Path to the data directory containing subdirectories with images
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

        # Scan all subdirectories for image pairs
        self._prepare_dataset(limit)

    def _prepare_dataset(self, limit):
        """Prepare list of (input_image, target_image) pairs"""
        # Iterate through all subdirectories in train folder
        for subject_dir in list(
            sorted(
                self.data_dir.iterdir(),
                key=lambda x: (0, int(x.name)) if x.name.isdigit() else (1, x.name),
            )
        )[:limit]:
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            # Find the target image (0001)
            target_path = subject_dir / f"{subject_id}-0001.png"
            if not target_path.exists():
                continue

            # Find all input images (0002-0012)
            for img_num in range(2, 13):
                img_path = subject_dir / f"{subject_id}-{img_num:04d}.png"
                if img_path.exists():
                    self.samples.append((str(img_path), str(target_path)))

        print(
            f"Loaded {len(self.samples)} image pairs from {len(set([s[1] for s in self.samples]))} subjects"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_path = self.samples[idx]

        # Load images as grayscale
        input_img = Image.open(input_path).convert("L")
        target_img = Image.open(target_path).convert("L")

        # Apply transforms
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img
