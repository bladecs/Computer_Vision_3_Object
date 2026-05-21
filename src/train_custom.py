import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_cnn import CustomVisionCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def save_training_plot(history: list[dict], test_metrics: dict | None, output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=180)
    ax_loss.plot(epochs, [item["train_loss"] for item in history], marker="o", label="Training")
    ax_loss.plot(epochs, [item["val_loss"] for item in history], marker="o", label="Validation")
    if test_metrics is not None:
        ax_loss.axhline(test_metrics["loss"], color="#27ae60", linestyle="--", label="Test")
    ax_loss.set_title("Loss During Training")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.grid(axis="y", linestyle="--", alpha=0.35)
    ax_loss.legend(frameon=False)

    ax_acc.plot(epochs, [item["train_acc"] for item in history], marker="o", label="Training")
    ax_acc.plot(epochs, [item["val_acc"] for item in history], marker="o", label="Validation")
    if test_metrics is not None:
        ax_acc.axhline(test_metrics["acc"], color="#27ae60", linestyle="--", label="Test")
    ax_acc.set_title("Accuracy During Training")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.02)
    ax_acc.grid(axis="y", linestyle="--", alpha=0.35)
    ax_acc.legend(frameon=False)

    fig.suptitle("Training Result with Test Performance", fontsize=14, weight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom CNN untuk klasifikasi objek conveyor")
    parser.add_argument("--data-root", type=Path, default=Path("data/split"), help="Folder split dataset")
    parser.add_argument("--epochs", type=int, default=40, help="Jumlah epoch")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--image-size", type=int, default=128, help="Ukuran input gambar")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=Path, default=Path("checkpoints/custom_cnn_all.pth"), help="Path simpan checkpoint")
    parser.add_argument("--plot-path", type=Path, default=None, help="Path to save the training/test plot")
    args = parser.parse_args()

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    test_dir = args.data_root / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Folder train/val tidak ditemukan. Jalankan split_dataset.py dulu.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=eval_tf)
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=eval_tf) if test_dir.exists() else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if test_ds is not None
        else None
    )

    num_classes = len(train_ds.classes)
    model = CustomVisionCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    print(f"Classes ({num_classes}): {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds) if test_ds is not None else 0}")

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            idx_to_class = {idx: name for name, idx in train_ds.class_to_idx.items()}
            checkpoint = {
                "model_state": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "idx_to_class": idx_to_class,
                "image_size": args.image_size,
                "mean": mean,
                "std": std,
                "best_val_acc": best_val_acc,
            }
            torch.save(checkpoint, args.save_path)
            print(f"  -> best checkpoint saved to {args.save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement {args.patience} epochs)")
            break

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    if args.save_path.exists():
        ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_metrics = None
    if test_loader is not None:
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)
        test_metrics = {"loss": test_loss, "acc": test_acc}
        print(f"Test  | loss={test_loss:.4f} acc={test_acc:.4f}")

    metrics_path = args.save_path.with_suffix(".metrics.json")
    plot_path = args.plot_path or args.save_path.with_suffix(".training_plot.png")
    save_training_plot(history, test_metrics, plot_path)
    metrics = {
        "best_val_acc": best_val_acc,
        "history": history,
        "test": test_metrics,
        "training_plot": str(plot_path.resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Training/test plot saved to {plot_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
