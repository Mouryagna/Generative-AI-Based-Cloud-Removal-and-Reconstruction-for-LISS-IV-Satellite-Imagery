
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.train import SatellitePatchDataset
from src.model import DiffCRModel
from src.diffusion import Diffusion
from src.sampler import DiffusionSampler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(r"E:\Rasengan\Cloud-Removal")
MANIFEST = ROOT / "data" / "diffcr_patches" / "manifest.csv"
CHECKPOINT = ROOT / "checkpoints" / "best_model.pth"
OUTPUT_DIR = ROOT / "evaluation"

BATCH_SIZE = 4
NUM_WORKERS = 0
SAMPLING_STEPS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_metrics(prediction, target):
    prediction = ((prediction.float() + 1.0) / 2.0).clamp(0, 1)
    target = ((target.float() + 1.0) / 2.0).clamp(0, 1)

    mse = torch.mean((prediction - target) ** 2).item()
    mae = torch.mean(torch.abs(prediction - target)).item()

    rmse = float(np.sqrt(mse))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))

    return mae, rmse, psnr


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def main():
    print("Device:", DEVICE)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Train the model before evaluating it."
        )

    dataset = SatellitePatchDataset(MANIFEST, "test")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = DiffCRModel().to(DEVICE)
    diffusion = Diffusion(device=DEVICE)
    sampler = DiffusionSampler(diffusion)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_patches = 0

    print(f"Test patches: {len(dataset):,}")
    print(f"Sampling steps: {SAMPLING_STEPS}")
    print("Starting evaluation...")

    with torch.inference_mode():
        for batch_idx, (cloudy, clear) in enumerate(loader):
            cloudy = cloudy.to(DEVICE, non_blocking=True)
            clear = clear.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):
                prediction = sampler.sample(
                    model,
                    cloudy,
                    sampling_steps=SAMPLING_STEPS,
                )

            if prediction.shape != clear.shape:
                raise ValueError(
                    f"Prediction shape {prediction.shape} "
                    f"does not match target shape {clear.shape}"
                )

            prediction_01 = ((prediction.float() + 1) / 2).clamp(0, 1)
            clear_01 = ((clear.float() + 1) / 2).clamp(0, 1)

            difference = prediction_01 - clear_01
            batch_mae = difference.abs().mean().item()
            batch_mse = difference.square().mean().item()

            batch_size = cloudy.size(0)

            total_mae += batch_mae * batch_size
            total_mse += batch_mse * batch_size

            batch_psnr = (
                float("inf")
                if batch_mse == 0
                else 10 * np.log10(1.0 / batch_mse)
            )
            total_psnr += batch_psnr * batch_size
            total_patches += batch_size

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Processed {total_patches:,}/"
                    f"{len(dataset):,} patches"
                )

    mean_mae = total_mae / total_patches
    mean_mse = total_mse / total_patches
    mean_rmse = float(np.sqrt(mean_mse))
    mean_psnr = total_psnr / total_patches

    results = {
        "test_patches": total_patches,
        "sampling_steps": SAMPLING_STEPS,
        "MAE": mean_mae,
        "RMSE": mean_rmse,
        "PSNR_dB": mean_psnr,
    }

    results_path = OUTPUT_DIR / "metrics.csv"

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print("\nEvaluation results")
    for key, value in results.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    print(f"\nMetrics saved to: {results_path}")


if __name__ == "__main__":
    main()
