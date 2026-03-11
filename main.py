import argparse
import torch
import os
import sys
from src.baseline.vit_trainer import train_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="ViT Baseline Training")

    # Experiment Settings
    parser.add_argument('--project_name', type=str, default='ViT_Baseline')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Dataset & Model
    # Changed default to tiny_imagenet
    parser.add_argument('--dataset', type=str, default='tiny_imagenet', choices=['cifar100', 'tiny_imagenet'])
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--model_name', type=str, default='vit_tiny_patch16_224', help='timm model name')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    # I/O
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training')

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Setup directories
    os.makedirs(args.save_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 40)
    print(f"Starting Project: {args.project_name}")
    print(f"Device   : {device}")
    print(f"Model    : {args.model_name}")
    print(f"Dataset  : {args.dataset}")
    print("=" * 40)

    # Start Training
    train_baseline(args, device)


if __name__ == '__main__':
    main()