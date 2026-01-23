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

from train import train


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
        "--epochs", type=int, default=30, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("UNet Training Pipeline for EKG Image Transformation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Train the model
    train(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
