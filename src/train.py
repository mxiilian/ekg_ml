import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision

from unet import UNet
from dataset import EKGDataset
import preprocess


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch_idx, (input_imgs, target_imgs) in enumerate(
        tqdm(dataloader, desc="Training")
    ):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_imgs)
        loss = criterion(outputs, target_imgs)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_imgs, target_imgs in tqdm(dataloader, desc="Validation"):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            outputs = model(input_imgs)
            loss = criterion(outputs, target_imgs)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_sample_predictions(model, dataloader, device, save_dir, epoch, num_samples=4):
    """Save sample predictions for visualization"""
    model.eval()
    save_path = Path(save_dir) / "samples"
    save_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Get first batch
        input_imgs, target_imgs = next(iter(dataloader))
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)

        # Generate predictions
        outputs = model(input_imgs)

        # Take only num_samples
        input_imgs = input_imgs[:num_samples]
        outputs = outputs[:num_samples]
        target_imgs = target_imgs[:num_samples]

        # Denormalize images (assuming normalization with mean=0.5, std=0.5)
        def denormalize(tensor):
            return tensor * 0.5 + 0.5

        input_imgs = denormalize(input_imgs).cpu()
        outputs = denormalize(outputs).cpu()
        target_imgs = denormalize(target_imgs).cpu()

        # Create figure with subplots
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Input
            axes[i, 0].imshow(input_imgs[i].squeeze().numpy(), cmap="gray")
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            # Output (prediction)
            axes[i, 1].imshow(outputs[i].squeeze().numpy(), cmap="gray")
            axes[i, 1].set_title("Output (Prediction)")
            axes[i, 1].axis("off")

            # Ground truth
            axes[i, 2].imshow(target_imgs[i].squeeze().numpy(), cmap="gray")
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(
            save_path / f"predictions_epoch_{epoch+1}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(
            f"Saved sample predictions to {save_path / f'predictions_epoch_{epoch+1}.png'}"
        )


def train_unet(
    data_dir,
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-4,
    val_split=0.1,
    checkpoint_dir="checkpoints",
    resume_from=None,
    device=None,
    only_preprocess=False,
):
    """
    Main training function for UNet

    Args:
        data_dir: Path to data directory
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        val_split: Fraction of data to use for validation
        checkpoint_dir: Directory to save model checkpoints
        resume_from: Path to checkpoint file to resume training from
        device: Device to train on (cuda/cpu)
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    limit=100

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Preprocess dataset
    gray_dir = preprocess.preprocess_dataset(data_dir, limit=limit)

    # Load dataset
    print("Loading dataset...")
    dataset = EKGDataset(gray_dir, limit=limit)

    if only_preprocess == True:
        return

    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    # Create dataloaders
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

    # Initialize model
    print("Initializing model...")
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from:
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            print(
                f"Resumed from epoch {checkpoint['epoch'] + 1}, best val loss: {best_val_loss:.6f}"
            )
        else:
            print(
                f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch."
            )

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.6f}")

        # Save sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_sample_predictions(model, val_loader, device, checkpoint_dir, epoch)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(checkpoint_dir) / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            save_sample_predictions(model, val_loader, device, checkpoint_dir, epoch)
            print(f"Saved best model to {checkpoint_path}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    return model
