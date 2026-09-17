"""
config.py - Cấu hình hệ thống và siêu tham số cho dự án VinBigData Chest X-ray.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Đường dẫn dự án
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Danh sách 15 lớp bệnh lý chuẩn của VinBigData Chest X-ray
# (14 lớp bất thường từ 0-13 và lớp 14 là No finding - Bình thường)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Aortic enlargement",  # 0
    "Atelectasis",         # 1
    "Calcification",       # 2
    "Cardiomegaly",        # 3
    "Consolidation",       # 4
    "ILD",                 # 5
    "Infiltration",        # 6
    "Lung Opacity",        # 7
    "Nodule/Mass",         # 8
    "Other lesion",        # 9
    "Pleural effusion",    # 10
    "Pleural thickening",  # 11
    "Pneumothorax",        # 12
    "Pulmonary fibrosis",  # 13
    "No finding",          # 14 (Bình thường)
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Siêu tham số Huấn luyện (Training Hyperparameters)
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_WORKERS = 0  # 0 trên Windows để đảm bảo ổn định đa luồng
DEFAULT_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Thiết bị tính toán: Ưu tiên CUDA GPU nếu có, ngược lại CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
