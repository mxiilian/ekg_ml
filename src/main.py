#!/usr/bin/env python3
"""
Simple pipeline to train a UNet model for transforming EKG images
Input: Images 0002-0015 from each subject directory
Output: Image 0001 (target image)
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from train import train_unet


def main():
    parser = argparse.ArgumentParser(
        description="Train UNet to transform EKG images (0002-0015) to target image (0001)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/train",
        help="Path to the training data directory",
    )

    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )

    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training (default: 8)"
    )

    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints)",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (e.g., checkpoints/best_model.pth)",
    )

    parser.add_argument(
        "--only-preprocess",
        type=bool,
        default=False,
        help="Only preprocess the dataset without training",
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("UNet Training Pipeline for EKG Image Transformation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Validation split: {args.val_split}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
    print("=" * 60)

    # Train the model
    train_unet(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        only_preprocess=args.only_preprocess,
    )


if __name__ == "__main__":
    main()
