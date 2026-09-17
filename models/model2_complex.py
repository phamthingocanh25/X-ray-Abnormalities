"""
model2_complex.py - Kiến trúc Complex CNN viết thủ công từ đầu (From Scratch).
Kết hợp đồng thời đường đi tuần tự (Sequential) và các nhánh song song (Multi-path / Parallel Branches) cùng Residual Shortcut.
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Khối cơ sở chuẩn hóa: Tích chập -> Chuẩn hóa Batch -> Kích hoạt ReLU"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiPathParallelBlock(nn.Module):
    """
    Khối song song đa nhánh (Multi-path / Parallel Branches):
    Bao gồm 4 nhánh tính toán song song kết hợp cùng đường tắt tuần tự (Residual Shortcut):
        - Nhánh 1 (Branch 1): 1x1 Conv (Biến đổi đặc trưng điểm)
        - Nhánh 2 (Branch 2): 1x1 Conv -> 3x3 Conv (Đặc trưng không gian cục bộ)
        - Nhánh 3 (Branch 3): 1x1 Conv -> Hai tầng 3x3 Conv liên tiếp (tương đương 5x5 Conv với trường thụ cảm rộng)
        - Nhánh 4 (Branch 4): 3x3 MaxPool -> 1x1 Conv (Đặc trưng gộp bất biến không gian)
    Các nhánh song song được ghép (concatenate) lại và cộng với Residual Shortcut.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels phải chia hết cho 4 để chia đều cho 4 nhánh song song"
        branch_channels = out_channels // 4

        # Nhánh 1: 1x1 Conv
        self.branch1 = ConvBNReLU(in_channels, branch_channels, kernel_size=1)

        # Nhánh 2: 1x1 Conv -> 3x3 Conv
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, branch_channels, kernel_size=1),
            ConvBNReLU(branch_channels, branch_channels, kernel_size=3, padding=1)
        )

        # Nhánh 3: 1x1 Conv -> 3x3 Conv -> 3x3 Conv (Mô phỏng 5x5)
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, branch_channels, kernel_size=1),
            ConvBNReLU(branch_channels, branch_channels, kernel_size=3, padding=1),
            ConvBNReLU(branch_channels, branch_channels, kernel_size=3, padding=1)
        )

        # Nhánh 4: MaxPool -> 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, branch_channels, kernel_size=1)
        )

        # Đường dẫn tắt tuần tự (Residual Shortcut connection)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Thực thi 4 nhánh song song
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        # Ghép (concatenate) kết quả từ 4 nhánh song song theo chiều channel
        parallel_out = torch.cat([out1, out2, out3, out4], dim=1)

        # Kết hợp với đường tắt tuần tự (Residual connection)
        res = self.shortcut(x)
        return self.relu(parallel_out + res)


class ComplexCNN(nn.Module):
    """
    Kiến trúc Complex CNN viết hoàn toàn từ đầu:
    Kết hợp đường đi tuần tự (Sequential Stem) qua các chặng song song đa nhánh (Multi-path Stages).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 15, dropout: float = 0.4):
        super().__init__()

        # --- 1. Sequential Stem (Tiền xử lý tuần tự ban đầu) ---
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224x224 -> 112x112
        )

        # --- 2. Giai đoạn đa nhánh song song 1 (Stage 1) ---
        self.stage1 = nn.Sequential(
            MultiPathParallelBlock(64, 128),
            MultiPathParallelBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112 -> 56x56
        )

        # --- 3. Giai đoạn đa nhánh song song 2 (Stage 2) ---
        self.stage2 = nn.Sequential(
            MultiPathParallelBlock(128, 256),
            MultiPathParallelBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56 -> 28x28
        )

        # --- 4. Giai đoạn đa nhánh song song 3 (Stage 3) ---
        self.stage3 = nn.Sequential(
            MultiPathParallelBlock(256, 512),
            MultiPathParallelBlock(512, 512),
        )

        # --- 5. Global Pooling & Phân loại đa lớp ---
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Đường đi tuần tự qua Stem
        x = self.stem(x)
        # Lan truyền qua các khối đa nhánh song song
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # Gộp toàn cục và phân loại
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


def get_model2_complex(num_classes: int = 15) -> ComplexCNN:
    """Hàm khởi tạo Model 2 (Complex CNN)"""
    return ComplexCNN(num_classes=num_classes)


if __name__ == "__main__":
    model = get_model2_complex(num_classes=15)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"[Model 2 Complex] Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
    assert output.shape == (2, 15), "Output shape không khớp!"
    print("[Model 2 Complex] Kiểm thử thành công!")
