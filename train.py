"""
train.py - Pipeline huấn luyện chuẩn PyTorch cho bài toán phân loại X-quang VinBigData.
Hỗ trợ lưu trọng số tốt nhất (best_model.pth), theo dõi hàm mất mát và độ chính xác qua từng epoch.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from data_loader import get_dataloaders
from models import get_model


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Huấn luyện mô hình trong 1 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="  [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Đánh giá mô hình trên tập validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="  [Val]  ", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(
    model_name: str = "simple",
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    data_dir: str = None,
    save_dir: Path = config.CHECKPOINT_DIR,
    device: torch.device = config.DEVICE
) -> Dict[str, List[float]]:
    """
    Quy trình huấn luyện hoàn chỉnh với cơ chế lưu checkpoint best_model.pth.
    """
    print("=" * 70)
    print(f"BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH: {model_name.upper()}")
    print(f"Thiết bị tính toán: {device} | Epochs: {epochs} | Batch size: {batch_size} | LR: {learning_rate}")
    print("=" * 70)

    # 1. Nạp dữ liệu
    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size
    )

    # 2. Khởi tạo mô hình
    model = get_model(model_name, num_classes=len(class_names)).to(device)

    # 3. Hàm mất mát & Tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Lưu vết lịch sử huấn luyện
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_val_acc = 0.0
    best_checkpoint_path = save_dir / f"{model_name}_best.pth"
    last_checkpoint_path = save_dir / f"{model_name}_last.pth"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%",
            end=""
        )

        # Lưu checkpoint mô hình có kết quả tốt nhất trên tập Validation
        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": len(class_names),
                "class_names": class_names,
            }, best_checkpoint_path)
            print(" -> [ĐÃ LƯU BEST MODEL]")
        else:
            print()

    # Lưu checkpoint cuối cùng
    torch.save({
        "epoch": epochs,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "num_classes": len(class_names),
    }, last_checkpoint_path)

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"Huấn luyện hoàn tất trong {total_time:.2f}s!")
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}% (Đã lưu tại: {best_checkpoint_path})")
    print("-" * 70)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại X-quang VinBigData")
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "complex", "transfer"],
                        help="Loại mô hình: 'simple', 'complex', hoặc 'transfer'")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn thư mục dữ liệu")

    args = parser.parse_args()
    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir
    )
