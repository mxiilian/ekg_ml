from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import os


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
    # Convert to numpy array
    img_array = np.array(img)

    # Create a binary mask: pixels lighter than threshold become white (grid lines)
    # pixels darker than threshold are kept (EKG signal)
    img_array[img_array > threshold] = 255

    # Convert back to PIL Image
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
    # Calculate aspect ratios
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    
    # Determine new size that fits within target while maintaining aspect ratio
    if img_ratio > target_ratio:
        # Image is wider, fit to width
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        # Image is taller, fit to height
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    
    # Resize image
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new image with target size and fill color
    new_img = Image.new('L', target_size, fill_color)
    
    # Paste resized image in center
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    
    return new_img


def preprocess_dataset(data_dir, limit, remove_grid=True, grid_threshold=60, target_size=(1024, 1024)):
    data_dir = Path(data_dir)
    gray_dir = data_dir.parent / "gray"
    gray_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        data_dir = gray_dir
        
    for index, subject_dir in enumerate(
        list(
            sorted(
                data_dir.iterdir(),
                key=lambda x: (0, int(x.name)) if x.name.isdigit() else (1, x.name),
            )
        )[:limit]
    ):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name

        # Create grayscale directory for this subject
        subject_gray_dir = gray_dir / subject_id
        if subject_gray_dir.exists():
            continue
        else:
            subject_gray_dir.mkdir(parents=True, exist_ok=True)

        # Find the target image (0001)
        target_path = subject_dir / f"{subject_id}-0001.png"
        if target_path.exists():
            # Convert target to grayscale
            img = Image.open(target_path).convert("L")
            if remove_grid:
                img = remove_grid_lines(img, threshold=grid_threshold)
            # Resize with padding to target size
            img = resize_with_padding(img, target_size=target_size)
            gray_target_path = subject_gray_dir / f"{subject_id}-0001.png"
            gray_target_compare_path = gray_dir / f"{subject_id}-0001.png"
            img.save(gray_target_path)
            img.save(gray_target_compare_path)

        # Find all input images (0002-0012)
        for img_num in range(2, 13):
            img_path = subject_dir / f"{subject_id}-{img_num:04d}.png"
            if img_path.exists():
                # Convert to grayscale and save
                img = Image.open(img_path).convert("L")
                # Resize with padding to target size
                img = resize_with_padding(img, target_size=target_size)
                gray_img_path = subject_gray_dir / f"{subject_id}-{img_num:04d}.png"
                img.save(gray_img_path)

        print(f"Processed subject {subject_id}: {index}/{limit if limit is not None else len(list(data_dir.iterdir()))}")

    print(f"Grayscale images saved to {gray_dir}")
    return gray_dir
