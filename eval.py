"""
eval.py - Script đánh giá và kiểm thử mô hình trên tập Test.
Tính toán Loss, Accuracy, Precision, Recall, F1-Score và vẽ Confusion Matrix.
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from data_loader import get_dataloaders
from models import get_model


def evaluate_model(
    model_name: str = "simple",
    checkpoint_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    save_plot: bool = True,
    device: torch.device = config.DEVICE
):
    """
    Tải checkpoint tốt nhất và đánh giá chi tiết mô hình trên tập Test.
    """
    # 1. Xác định đường dẫn checkpoint
    if checkpoint_path is None:
        ckpt_file = config.CHECKPOINT_DIR / f"{model_name}_best.pth"
    else:
        ckpt_file = Path(checkpoint_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại: {ckpt_file}. "
            f"Vui lòng huấn luyện mô hình trước bằng lệnh: python train.py --model {model_name}"
        )

    print("=" * 70)
    print(f"BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH: {model_name.upper()}")
    print(f"Checkpoint: {ckpt_file}")
    print("=" * 70)

    # 2. Nạp dữ liệu kiểm thử
    _, _, test_loader, class_names = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size
    )

    # 3. Nạp trọng số mô hình
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", len(class_names))
    model = get_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # 4. Dự đoán trên toàn bộ tập Test
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[Testing]"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_loss = total_loss / len(all_labels)
    test_accuracy = (all_preds == all_labels).mean()

    # 5. Xuất báo cáo đánh giá chi tiết
    print("\n" + "=" * 70)
    print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f"  - Test Loss    : {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 70)

    # Báo cáo Classification Report (Precision, Recall, F1)
    present_classes = np.unique(np.concatenate([all_labels, all_preds]))
    target_names = [class_names[i] for i in present_classes]

    report = classification_report(
        all_labels,
        all_preds,
        labels=present_classes,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    print("\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT) ---")
    print(report)

    # 6. Tính Ma trận nhầm lẫn (Confusion Matrix) & Lưu biểu đồ
    cm = confusion_matrix(all_labels, all_preds, labels=present_classes)

    if save_plot:
        plot_path = config.OUTPUT_DIR / f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - Model: {model_name} (Acc: {test_accuracy*100:.1f}%)")
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, ha="right", fontsize=8)
        plt.yticks(tick_marks, target_names, fontsize=8)

        # Ghi số lượng vào từng ô
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        plt.ylabel("Nhãn thực tế (True Label)")
        plt.xlabel("Nhãn dự đoán (Predicted Label)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[eval] Đã lưu biểu đồ Confusion Matrix tại: {plot_path}")

    return {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình phân loại X-quang VinBigData")
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "complex", "transfer"],
                        help="Loại mô hình: 'simple', 'complex', hoặc 'transfer'")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn file .pth (tùy chọn)")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn thư mục dữ liệu")
    parser.add_argument("--no_plot", action="store_true", help="Không lưu ảnh Confusion Matrix")

    args = parser.parse_args()
    evaluate_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        save_plot=not args.no_plot
    )
