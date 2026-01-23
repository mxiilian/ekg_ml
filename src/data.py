import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def remove_grid_lines(img, threshold=60):
    """
    Remove grid lines from EKG image using thresholding.
    Grid lines are typically lighter (higher pixel values) than the EKG signal.

    Args:
        img: PIL Image in grayscale
        threshold: Pixel values above this threshold will be set to white (255)

    Returns:
        PIL Image with grid lines removed
    """
    img_array = np.array(img)
    img_array[img_array > threshold] = 255
    return Image.fromarray(img_array)


def resize_with_padding(img, target_size=(1024, 1024), fill_color=255):
    """
    Resize image to target size while maintaining aspect ratio and adding padding.

    Args:
        img: PIL Image
        target_size: Tuple of (width, height) for the output size
        fill_color: Color to use for padding (255 for white, 0 for black)

    Returns:
        PIL Image resized to target_size with padding
    """
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)

    # Use LANCZOS resampling (Pillow 9.0+ uses Image.Resampling.LANCZOS)
    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS
    img_resized = img.resize((new_width, new_height), resample_method)
    new_img = Image.new("L", target_size, fill_color)

    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img


def preprocess_dataset(
    data_dir, limit=None, remove_grid=True, grid_threshold=60, target_size=(1024, 1024)
):
    """
    Preprocess dataset by converting images to grayscale, removing grid lines,
    and resizing with padding.

    Args:
        data_dir: Path to the raw data directory
        limit: Maximum number of subjects to process (None for all)
        remove_grid: Whether to remove grid lines from target images
        grid_threshold: Threshold for grid line removal
        target_size: Output image size (width, height)

    Returns:
        Path to the preprocessed grayscale directory
    """
    data_dir = Path(data_dir)
    gray_dir = data_dir.parent / "gray"
    gray_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Warning: Data directory {data_dir} does not exist, using {gray_dir}")
        return gray_dir

    subject_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda x: (0, int(x.name)) if x.name.isdigit() else (1, x.name),
    )

    if limit is not None:
        subject_dirs = subject_dirs[:limit]

    total_subjects = len(subject_dirs)

    for index, subject_dir in enumerate(subject_dirs):
        subject_id = subject_dir.name
        subject_gray_dir = gray_dir / subject_id

        # Skip if already processed
        if subject_gray_dir.exists():
            continue

        subject_gray_dir.mkdir(parents=True, exist_ok=True)

        # Process target image (0001)
        target_path = subject_dir / f"{subject_id}-0001.png"
        if target_path.exists():
            img = Image.open(target_path).convert("L")
            if remove_grid:
                img = remove_grid_lines(img, threshold=grid_threshold)
            img = resize_with_padding(img, target_size=target_size)

            # Save to subject directory and root gray directory for comparison
            gray_target_path = subject_gray_dir / f"{subject_id}-0001.png"
            gray_target_compare_path = gray_dir / f"{subject_id}-0001.png"
            img.save(gray_target_path)
            img.save(gray_target_compare_path)

        # Process input images (0002-0012)
        for img_num in range(2, 13):
            img_path = subject_dir / f"{subject_id}-{img_num:04d}.png"
            if img_path.exists():
                img = Image.open(img_path).convert("L")
                img = resize_with_padding(img, target_size=target_size)
                gray_img_path = subject_gray_dir / f"{subject_id}-{img_num:04d}.png"
                img.save(gray_img_path)

        print(f"Processed subject {subject_id}: {index + 1}/{total_subjects}")

    print(f"Grayscale images saved to {gray_dir}")
    return gray_dir


class EKGDataset(Dataset):
    """Dataset for loading EKG image pairs (input images 0002-0012 -> target image 0001)"""

    def __init__(self, data_dir, limit=None, transform=None):
        """
        Args:
            data_dir: Path to the data directory containing subdirectories with images
            limit: Maximum number of subjects to load (None for all)
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

        self._prepare_dataset(limit)

    def _prepare_dataset(self, limit):
        """Prepare list of (input_image, target_image) pairs"""
        subject_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir()],
            key=lambda x: (0, int(x.name)) if x.name.isdigit() else (1, x.name),
        )

        if limit is not None:
            subject_dirs = subject_dirs[:limit]

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            target_path = subject_dir / f"{subject_id}-0001.png"

            if not target_path.exists():
                continue

            # Find all input images (0002-0012)
            for img_num in range(2, 13):
                img_path = subject_dir / f"{subject_id}-{img_num:04d}.png"
                if img_path.exists():
                    self.samples.append((str(img_path), str(target_path)))

        num_subjects = len(set(s[1] for s in self.samples))
        print(f"Loaded {len(self.samples)} image pairs from {num_subjects} subjects")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_path = self.samples[idx]

        input_img = Image.open(input_path).convert("L")
        target_img = Image.open(target_path).convert("L")

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img
