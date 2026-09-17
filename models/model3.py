"""
model3.py - Mô hình Base Network áp dụng Transfer Learning và Fine-tuning.
Sử dụng pre-trained backbone mạnh mẽ từ torchvision (ResNet, MobileNet, EfficientNet).
"""

from typing import Literal
import torch
import torch.nn as nn
import torchvision.models as models


class TransferLearningModel(nn.Module):
    """
    Mô hình Transfer Learning dựa trên pre-trained base network.
    Hỗ trợ cả chế độ:
        - Feature Extraction: Đóng băng toàn bộ backbone (freeze_base=True), chỉ huấn luyện classifier head.
        - Fine-Tuning: Mở khóa các tầng backbone để huấn luyện tinh chỉnh (freeze_base=False).
    """
    def __init__(
        self,
        num_classes: int = 15,
        backbone_name: Literal["resnet18", "resnet50", "mobilenet_v3", "efficientnet_b0"] = "resnet18",
        pretrained: bool = True,
        freeze_base: bool = False,
        dropout: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # -------------------------------------------------------------------
        # 1. Khởi tạo Pre-trained Backbone
        # -------------------------------------------------------------------
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            # Thay thế tầng phân loại cuối cùng
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        elif backbone_name == "mobilenet_v3":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes)
            )

        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Không hỗ trợ backbone: {backbone_name}")

        # -------------------------------------------------------------------
        # 2. Xử lý đóng băng tham số nếu chọn Feature Extraction
        # -------------------------------------------------------------------
        if freeze_base:
            self._freeze_backbone_layers()

    def _freeze_backbone_layers(self):
        """Đóng băng các tầng trích xuất đặc trưng của backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Mở khóa riêng cho classification head mới
        if hasattr(self.backbone, "fc"):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.backbone, "classifier"):
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model3_transfer(
    num_classes: int = 15,
    backbone_name: str = "resnet18",
    pretrained: bool = True,
    freeze_base: bool = False
) -> TransferLearningModel:
    """Hàm khởi tạo Model 3 (Transfer Learning / Fine-tuning)"""
    return TransferLearningModel(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=pretrained,
        freeze_base=freeze_base
    )


if __name__ == "__main__":
    model = get_model3_transfer(num_classes=15, backbone_name="resnet18", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"[Model 3 Transfer] Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
    assert output.shape == (2, 15), "Output shape không khớp!"
    print("[Model 3 Transfer] Kiểm thử thành công!")
