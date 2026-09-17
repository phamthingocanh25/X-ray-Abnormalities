"""
Package models - Cung cấp các kiến trúc mô hình học sâu cho bài toán VinBigData.
"""

from .model1_simple import SimpleCNN, get_model1_simple
from .model2_complex import ComplexCNN, get_model2_complex
from .model3 import TransferLearningModel, get_model3_transfer


def get_model(model_name: str, num_classes: int = 15, **kwargs):
    """
    Factory function khởi tạo mô hình theo tên định danh.
    Args:
        model_name: "simple" / "model1", "complex" / "model2", "transfer" / "model3"
        num_classes: Số lượng lớp đầu ra (mặc định 15)
    """
    model_name = model_name.lower().strip()
    if model_name in ("simple", "model1", "model1_simple"):
        return get_model1_simple(num_classes=num_classes)
    elif model_name in ("complex", "model2", "model2_complex"):
        return get_model2_complex(num_classes=num_classes)
    elif model_name in ("transfer", "model3", "base"):
        return get_model3_transfer(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Không nhận diện được mô hình '{model_name}'. Chọn: 'simple', 'complex', hoặc 'transfer'.")
