"""
model3.py - Wrapper tại thư mục gốc trỏ tới models/model3.py.
"""
from models.model3 import TransferLearningModel, get_model3_transfer

if __name__ == "__main__":
    import torch
    model = get_model3_transfer(num_classes=15, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    print(f"[Root model3] Output: {model(dummy).shape}")
