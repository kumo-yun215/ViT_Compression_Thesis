import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import time
from tqdm import tqdm
from src.utils.data_setup import download_and_prepare_tiny_imagenet


def get_dataloader(args):
    """
    Prepares DataLoaders for CIFAR-100 or Tiny-ImageNet.
    Note: Standard ViTs expect 224x224 input. Tiny-ImageNet is 64x64.
    We resize images to 224x224 to use standard pre-trained weights.
    """
    print(f"[INFO] Preparing dataset: {args.dataset}...")

    # Standard ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Upsample to match ViT input
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if args.dataset == 'cifar100':
        train_set = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
        args.num_classes = 100

    elif args.dataset == 'tiny_imagenet':
        # Custom helper to download and format Tiny-ImageNet
        dataset_root = download_and_prepare_tiny_imagenet(args.data_dir)

        train_dir = os.path.join(dataset_root, 'train')
        val_dir = os.path.join(dataset_root, 'val')

        train_set = datasets.ImageFolder(root=train_dir, transform=transform_train)
        test_set = datasets.ImageFolder(root=val_dir, transform=transform_test)
        args.num_classes = 200

    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def train_baseline(args, device):
    # 1. Prepare Data
    train_loader, test_loader = get_dataloader(args)

    # 2. Load Model
    print(f"[INFO] Loading model: {args.model_name} (Pretrained={args.pretrained})...")
    model = timm.create_model(
        args.model_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes
    )
    model = model.to(device)

    # 3. Resume Checkpoint Logic
    if args.resume and os.path.isfile(args.resume):
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
    elif args.resume:
        print(f"[WARNING] Checkpoint {args.resume} not found. Starting from scratch.")

    # 4. Log Model Statistics (Thesis Requirement)
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / 1024 / 1024
    print(f"[STATS] Parameters: {param_count / 1e6:.2f} M")
    print(f"[STATS] Theoretical Size: {model_size_mb:.2f} MB")

    # 5. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 6. Training Loop
    best_acc = 0.0
    print(f"[INFO] Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]")

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=running_loss / len(train_loader), acc=100. * correct / total)

        scheduler.step()

        # Validation
        val_acc = validate(model, test_loader, device)
        print(f" -> Validation Accuracy: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f"{args.model_name}_{args.dataset}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f" -> Model saved: {save_path}")

    print("=" * 40)
    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total