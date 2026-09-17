"""
main.py - Điểm khởi chạy chính (Pipeline Runner) cho toàn bộ quy trình:
Dữ liệu -> Huấn luyện -> Đánh giá (End-to-End Deep Learning Pipeline).
"""

import argparse
import sys
from pathlib import Path
from typing import List

import config
from data_loader import get_dataloaders
from eval import evaluate_model
from train import train


def run_pipeline(
    model_names: List[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mode: str = "all",
    data_dir: str = None
):
    """
    Điều phối luồng thực thi từ nạp dữ liệu, huấn luyện đến đánh giá cho một hoặc nhiều mô hình.
    """
    print("*" * 80)
    print("      DỰ ÁN PHÂN LOẠI BẤT THƯỜNG X-QUANG PHỔI (VINBIGDATA CHEST X-RAY)      ")
    print("*" * 80)
    print(f"Danh sách mô hình thực thi: {model_names}")
    print(f"Chế độ (Mode): {mode.upper()} | Số Epochs: {epochs} | Batch size: {batch_size}")
    print("*" * 80)

    # Bước 1: Khởi tạo/Kiểm tra dữ liệu trước
    print("\n>>> BƯỚC 1: KIỂM TRA VÀ NẠP DỮ LIỆU...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size
    )
    print(f"[OK] Đã sẵn sàng dữ liệu với {len(class_names)} lớp nhãn phân loại.\n")

    summary_results = []

    # Bước 2 & 3: Lặp qua từng mô hình để Train và Eval
    for model_name in model_names:
        print("\n" + "#" * 80)
        print(f"### TIẾN TRÌNH CHO MÔ HÌNH: {model_name.upper()} ###")
        print("#" * 80)

        # Pha Train
        if mode in ("train", "all"):
            print(f"\n>>> BƯỚC 2: HUẤN LUYỆN MÔ HÌNH [{model_name}]...")
            train(
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                data_dir=data_dir
            )

        # Pha Eval
        if mode in ("eval", "all"):
            print(f"\n>>> BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH [{model_name}] TRÊN TẬP TEST...")
            eval_metrics = evaluate_model(
                model_name=model_name,
                data_dir=data_dir,
                batch_size=batch_size
            )
            summary_results.append({
                "model": model_name,
                "loss": eval_metrics["loss"],
                "accuracy": eval_metrics["accuracy"]
            })

    # Bảng tổng kết so sánh nếu chạy từ 2 mô hình trở lên
    if summary_results and len(summary_results) > 1:
        print("\n" + "=" * 80)
        print("BẢNG TỔNG KẾT VÀ SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH TRÊN TẬP TEST")
        print("=" * 80)
        print(f"{'Mô hình':<20} | {'Test Loss':<15} | {'Test Accuracy':<15}")
        print("-" * 80)
        for res in summary_results:
            print(f"{res['model']:<20} | {res['loss']:<15.4f} | {res['accuracy']*100:<14.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy pipeline phân loại X-quang VinBigData")
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=["simple", "complex", "transfer", "all"],
        help="Chọn mô hình: 'simple' (Model 1), 'complex' (Model 2), 'transfer' (Model 3), hoặc 'all' (Cả 3)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "train", "eval"],
        help="Chế độ thực thi: 'all' (Train + Eval), 'train' (Chỉ Train), 'eval' (Chỉ Eval)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs huấn luyện (mặc định: 3 cho test nhanh)")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Tốc độ học (Learning rate)")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn tới thư mục dữ liệu")

    args = parser.parse_args()

    # Xử lý danh sách mô hình
    if args.model == "all":
        selected_models = ["simple", "complex", "transfer"]
    else:
        selected_models = [args.model]

    run_pipeline(
        model_names=selected_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mode=args.mode,
        data_dir=args.data_dir
    )
