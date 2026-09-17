"""
model1_simple.py - Kiến trúc Simple CNN viết thủ công từ đầu (From Scratch).
Tuân thủ nghiêm ngặt nguyên tắc đan xen luân phiên giữa tầng Conv2d và tầng Pooling.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Kiến trúc Simple CNN viết từ đầu (không sử dụng pre-trained backbone).
    Đặc điểm cốt lõi: Đan xen luân phiên tuyệt đối giữa Conv2d và Pooling.
    
    Cấu trúc luồng:
        Input: (B, 3, 224, 224)
        Stage 1: Conv2d (3 -> 32)   -> BatchNorm -> ReLU -> MaxPool2d (112x112)
        Stage 2: Conv2d (32 -> 64)  -> BatchNorm -> ReLU -> MaxPool2d (56x56)
        Stage 3: Conv2d (64 -> 128) -> BatchNorm -> ReLU -> MaxPool2d (28x28)
        Stage 4: Conv2d (128 -> 256)-> BatchNorm -> ReLU -> MaxPool2d (14x14)
        Global Average Pooling (1x1)
        Classifier: Linear(256, 128) -> ReLU -> Dropout -> Linear(128, num_classes)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 15, dropout: float = 0.3):
        super().__init__()
        
        # -------------------------------------------------------------------
        # Khối trích xuất đặc trưng: Đan xen luân phiên nghiêm ngặt Conv & Pool
        # -------------------------------------------------------------------
        self.features = nn.Sequential(
            # --- Cặp 1: Conv2d -> MaxPool2d ---
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # --- Cặp 2: Conv2d -> MaxPool2d ---
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # --- Cặp 3: Conv2d -> MaxPool2d ---
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # --- Cặp 4: Conv2d -> MaxPool2d ---
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # -------------------------------------------------------------------
        # Tầng gộp toàn cục (Global Adaptive Pooling) đưa về kích thước 1x1
        # -------------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # -------------------------------------------------------------------
        # Bộ phân loại (Classifier Head)
        # -------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bước 1: Lan truyền qua các tầng luân phiên Conv-Pool
        x = self.features(x)
        # Bước 2: Gộp đặc trưng toàn cục
        x = self.global_pool(x)
        # Bước 3: Phân loại nhãn
        logits = self.classifier(x)
        return logits


def get_model1_simple(num_classes: int = 15) -> SimpleCNN:
    """Hàm khởi tạo Model 1 (Simple CNN)"""
    return SimpleCNN(num_classes=num_classes)


if __name__ == "__main__":
    # Kiểm tra kích thước tensor đầu ra
    model = get_model1_simple(num_classes=15)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"[Model 1 Simple] Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
    assert output.shape == (2, 15), "Output shape không khớp!"
    print("[Model 1 Simple] Kiểm thử thành công!")
