"""
model2_complex.py - Wrapper tại thư mục gốc trỏ tới models/model2_complex.py.
"""
from models.model2_complex import ComplexCNN, MultiPathParallelBlock, get_model2_complex

if __name__ == "__main__":
    import torch
    model = get_model2_complex(num_classes=15)
    dummy = torch.randn(2, 3, 224, 224)
    print(f"[Root model2_complex] Output: {model(dummy).shape}")
