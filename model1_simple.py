"""
model1_simple.py - Wrapper tại thư mục gốc trỏ tới models/model1_simple.py.
"""
from models.model1_simple import SimpleCNN, get_model1_simple

if __name__ == "__main__":
    import torch
    model = get_model1_simple(num_classes=15)
    dummy = torch.randn(2, 3, 224, 224)
    print(f"[Root model1_simple] Output: {model(dummy).shape}")
