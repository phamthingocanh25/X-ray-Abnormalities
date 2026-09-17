"""
data_loader.py - Tiền xử lý, tăng cường dữ liệu và chuẩn bị PyTorch DataLoader.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import os
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config


# ---------------------------------------------------------------------------
# 1. Định nghĩa Pipeline Biến đổi & Tăng cường Dữ liệu (Transforms)
# ---------------------------------------------------------------------------
def get_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE):
    """
    Tạo các phép biến đổi ảnh cho các pha Train, Val và Test.
    Train: Áp dụng Data Augmentation để chống quá khớp (overfitting).
    Val/Test: Chỉ Resize và Chuẩn hóa (Normalize) theo ImageNet.
    """
    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),                    # Lật ngang ngẫu nhiên
        T.RandomRotation(degrees=10),                      # Xoay nhẹ góc tối đa 10 độ
        T.ColorJitter(brightness=0.1, contrast=0.1),       # Điều chỉnh độ sáng/tương phản
        T.ToTensor(),                                      # Chuyển về Tensor [0, 1]
        T.Normalize(mean=[0.485, 0.456, 0.406],            # Chuẩn hóa theo thống kê ImageNet
                    std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# 2. PyTorch Dataset cho VinBigData
# ---------------------------------------------------------------------------
class VinBigDataDataset(Dataset):
    """
    Dataset tải ảnh X-quang từ danh sách đường dẫn và nhãn phân loại.
    """
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        """
        Args:
            samples: Danh sách tuple (đường_dẫn_ảnh, class_id).
            transform: Chuỗi các phép biến đổi torchvision.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# 3. Trình khởi tạo dữ liệu mẫu mô phỏng (Synthetic/Demo Dataset Generator)
# Đảm bảo pipeline luôn sẵn sàng chạy thử nghiệm ngay mà không cần tải trước 50GB.
# ---------------------------------------------------------------------------
def generate_demo_dataset(base_dir: Path, num_samples_per_split: int = 15) -> Path:
    """
    Tự động tạo tập ảnh demo mô phỏng cấu trúc X-quang lồng ngực cho 15 nhãn.
    """
    demo_root = base_dir / "demo_data"
    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = demo_root / split
        if split_dir.exists() and any(split_dir.iterdir()):
            continue  # Đã tồn tại dữ liệu demo

        split_dir.mkdir(parents=True, exist_ok=True)
        count = num_samples_per_split if split == "train" else max(4, num_samples_per_split // 2)

        for class_id, class_name in enumerate(config.CLASS_NAMES):
            class_folder = split_dir / f"{class_id:02d}_{class_name.replace(' ', '_')}"
            class_folder.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                # Tạo ảnh giả lập hình lồng ngực x-ray xám với các hình elip mô phỏng phổi
                img = Image.new("L", (224, 224), color=20)
                draw = ImageDraw.Draw(img)
                # Vẽ hai vùng phổi sáng mờ
                draw.ellipse([30, 40, 100, 180], fill=120)
                draw.ellipse([124, 40, 194, 180], fill=120)
                # Thêm đặc trưng giả lập cho từng nhãn
                if class_id != 14:  # Bất thường
                    pos_x = 40 + (class_id * 10) % 120
                    pos_y = 60 + (class_id * 7) % 100
                    draw.ellipse([pos_x, pos_y, pos_x + 20, pos_y + 20], fill=220)

                img.save(class_folder / f"sample_{i:03d}.png")

    return demo_root


# ---------------------------------------------------------------------------
# 4. Quét thư mục ảnh và xây dựng DataLoader
# ---------------------------------------------------------------------------
def load_split_samples(split_dir: Path) -> List[Tuple[str, int]]:
    """
    Quét thư mục chứa các thư mục con theo lớp để lấy (đường_dẫn_ảnh, label_id).
    """
    samples = []
    if not split_dir.exists():
        return samples

    # Tìm các thư mục con
    subdirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    for dir_idx, folder in enumerate(subdirs):
        # Trích xuất class_id từ tiền tố thư mục nếu có (vd: 00_Aortic_enlargement -> 0)
        folder_name = folder.name
        try:
            class_id = int(folder_name.split("_")[0])
        except ValueError:
            class_id = dir_idx

        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_file in folder.glob(ext):
                samples.append((str(img_file), class_id))

    return samples


def get_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Tạo bộ ba DataLoader: Train, Val, Test.
    Nếu data_dir không được truyền hoặc trống, tự động kích hoạt demo dataset.
    """
    train_transform, eval_transform = get_transforms(image_size)

    # Xác định thư mục dữ liệu
    target_dir = Path(data_dir) if data_dir else config.DATA_DIR

    train_samples = load_split_samples(target_dir / "train")
    val_samples = load_split_samples(target_dir / "val")
    test_samples = load_split_samples(target_dir / "test")

    # Nếu không có dữ liệu thực tế, kích hoạt chế độ tạo demo dataset
    if len(train_samples) == 0:
        print("[data_loader] Không tìm thấy dữ liệu trong data/. Tự động khởi tạo bộ dữ liệu mô phỏng (Demo Dataset)...")
        demo_dir = generate_demo_dataset(config.BASE_DIR)
        train_samples = load_split_samples(demo_dir / "train")
        val_samples = load_split_samples(demo_dir / "val")
        test_samples = load_split_samples(demo_dir / "test")

    print(f"[data_loader] Đã tải: {len(train_samples)} mẫu Train, {len(val_samples)} mẫu Val, {len(test_samples)} mẫu Test.")

    train_dataset = VinBigDataDataset(train_samples, transform=train_transform)
    val_dataset = VinBigDataDataset(val_samples, transform=eval_transform)
    test_dataset = VinBigDataDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, config.CLASS_NAMES


if __name__ == "__main__":
    # Chạy kiểm thử nạp dữ liệu nhanh
    train_ld, val_ld, test_ld, classes = get_dataloaders(batch_size=4)
    images, labels = next(iter(train_ld))
    print(f"[Kiểm thử data_loader] Batch shape: {images.shape}, Labels shape: {labels.shape}")
    print("[data_loader] Kiểm thử thành công!")
