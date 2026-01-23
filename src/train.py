from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from model import LightweightAttentionUNet, SimpleUNet
from data import EKGDataset, preprocess_dataset
from util import CombinedLoss, WeightedL1Loss, count_parameters


# Global log file handle
_log_file = None


def log(message, end="\n"):
    """Print message to console and append to log file."""
    print(message, end=end)
    if _log_file is not None:
        _log_file.write(message + end)
        _log_file.flush()


def init_log(checkpoint_dir):
    """Initialize the markdown log file."""
    global _log_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(checkpoint_dir) / f"{timestamp}_training_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, "a")
    _log_file.write(
        f"\n\n# Training Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    _log_file.flush()
    return log_path


def close_log():
    """Close the log file."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def save_sample_predictions(
    model,
    device,
    save_dir,
    image_pairs,
    epoch=None,
    transform=None,
):
    """
    Save input/output/target visualization for multiple image pairs.

    Args:
        model: The trained model
        device: Device to run inference on
        save_dir: Directory to save images
        image_pairs: List of tuples (input_path, target_path) or just input_path strings
                     e.g. [("input1.png", "target1.png"), ("input2.png", "target2.png")]
                     or ["input1.png", "input2.png"] if no targets
        epoch: Current epoch number (optional, included in filename if provided)
        transform: Transform to apply to images (default: ToTensor + Normalize)
    """
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    # Normalize pairs to (input, target) tuples
    pairs = []
    for item in image_pairs:
        if isinstance(item, (tuple, list)):
            pairs.append((item[0], item[1] if len(item) > 1 else None))
        else:
            pairs.append((item, None))

    if not pairs:
        return

    save_path = Path(save_dir) / "samples"
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    has_targets = any(p[1] is not None for p in pairs)
    num_cols = 3 if has_targets else 2
    num_rows = len(pairs)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, (input_path, target_path) in enumerate(pairs):
            # Load and process input
            input_img = Image.open(input_path).convert("L")
            input_tensor = transform(input_img).unsqueeze(0).to(device)
            output_tensor = model(input_tensor)

            # Denormalize
            input_display = (input_tensor[0] * 0.5 + 0.5).cpu().squeeze().numpy()
            output_display = (output_tensor[0] * 0.5 + 0.5).cpu().squeeze().numpy()

            axes[i, 0].imshow(input_display, cmap="gray")
            axes[i, 0].set_title(f"Input: {Path(input_path).name}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(output_display, cmap="gray")
            axes[i, 1].set_title("Prediction")
            axes[i, 1].axis("off")

            if has_targets:
                if target_path:
                    target_img = Image.open(target_path).convert("L")
                    target_tensor = transform(target_img)
                    target_display = (target_tensor * 0.5 + 0.5).squeeze().numpy()
                    axes[i, 2].imshow(target_display, cmap="gray")
                    axes[i, 2].set_title("Ground Truth")
                else:
                    axes[i, 2].set_visible(False)
                axes[i, 2].axis("off")

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch_str = f"_epoch_{epoch+1}" if epoch is not None else ""
    filename = f"predictions{epoch_str}_{timestamp}.png"
    filepath = save_path / filename

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Saved: {filepath}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Run one training epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    train_loss = 0

    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    Run validation.

    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

    return val_loss / len(val_loader)


def train(
    data_dir,
    num_epochs=30,
    batch_size=4,
    learning_rate=1e-3,
    val_split=0.2,
    checkpoint_dir="points",
    resume_from=None,
    only_preprocess=False,
):
    """
    Main training function for the U-Net model.

    Args:
        data_dir: Path to the training data directory
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        val_split: Fraction of data to use for validation
        checkpoint_dir: Directory to save model checkpoints
        resume_from: Path to checkpoint file to resume training from
        only_preprocess: If True, only preprocess the dataset without training

    Returns:
        Trained model
    """
    # Create checkpoint directory and init log
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path = init_log(checkpoint_dir)

    try:
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Using device: {device}")

        # Log configuration
        log("\n## Configuration\n")
        log(f"- Data directory: {data_dir}")
        log(f"- Epochs: {num_epochs}")
        log(f"- Batch size: {batch_size}")
        log(f"- Learning rate: {learning_rate}")
        log(f"- Validation split: {val_split}")
        log(f"- Checkpoint directory: {checkpoint_dir}")

        # Preprocess data
        log("\n## Preprocessing\n")
        log("Preprocessing dataset...")
        gray_dir = preprocess_dataset(data_dir)

        if only_preprocess:
            log("Preprocessing complete. Exiting (--only-preprocess flag set).")
            return None

        # Load dataset
        log("\n## Dataset\n")
        log("Loading dataset...")
        dataset = EKGDataset(gray_dir)

        if len(dataset) == 0:
            log("Error: No samples found in dataset. Please check your data directory.")
            return None

        # Model initialization
        model = SimpleUNet().to(device)
        log(f"Model parameters: {count_parameters(model):,}")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        criterion = WeightedL1Loss()

        # Training/Validation split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        log(f"Train samples: {train_size}, Validation samples: {val_size}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == "cuda" else False,
        )

        start_epoch = 0
        best_val_loss = float("inf")

        # Resume from checkpoint if specified
        if resume_from is not None:
            resume_path = Path(resume_from)
            if resume_path.exists():
                log(f"\nResuming from checkpoint: {resume_from}")
                checkpoint = torch.load(resume_from, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_val_loss = checkpoint.get("val_loss", float("inf"))
                log(
                    f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}"
                )
            else:
                log(
                    f"Warning: Checkpoint file {resume_from} not found. Starting from scratch."
                )

        # Specific sample images for visualization
        sample_images = [
            ("57127905", "0012"),
            ("11842146", "0012"),
            ("88853795", "0009"),
            ("78833408", "0006"),
        ]
        sample_pairs = []
        for subject_id, img_num in sample_images:
            input_path = gray_dir / subject_id / f"{subject_id}-{img_num}.png"
            target_path = gray_dir / subject_id / f"{subject_id}-0001.png"
            if input_path.exists() and target_path.exists():
                sample_pairs.append((str(input_path), str(target_path)))

        # Training loop
        log("\n## Training\n")
        log("| Epoch | Train Loss | Val Loss | Best |")
        log("|-------|------------|----------|------|")

        for epoch in range(start_epoch, num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step()

            # Check if best model
            is_best = val_loss < best_val_loss
            best_marker = "*" if is_best else ""

            log(
                f"| {epoch+1}/{num_epochs} | {train_loss:.6f} | {val_loss:.6f} | {best_marker} |"
            )

            # Save checkpoint if best model
            if is_best:
                best_val_loss = val_loss
                best_model_path = checkpoint_path / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    best_model_path,
                )

                # Save sample predictions
                if sample_pairs:
                    save_sample_predictions(
                        model=model,
                        device=device,
                        save_dir=checkpoint_dir,
                        image_pairs=sample_pairs,
                        epoch=epoch,
                    )

        log(f"\n## Summary\n")
        log(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        log(f"Log saved to: {log_path}")
        return model

    finally:
        close_log()
