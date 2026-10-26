
import csv
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from src.model import DiffCRModel
from src.diffusion import Diffusion


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(r"E:\Rasengan\Cloud-Removal")
MANIFEST = ROOT / "data" / "diffcr_patches" / "manifest.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
SEED = 42

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

def find_column(columns, candidates, description):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Cannot identify {description} column in manifest. "
        f"Available columns: {list(columns)}"
    )


class SatellitePatchDataset(Dataset):
    def __init__(self, manifest_path, split):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent.parent.parent

        with open(self.manifest_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = reader.fieldnames or []

        cloudy_col = find_column(
            columns,
            ["cloudy_path", "cloudy", "input_path", "cloudy_file"],
            "cloudy image path",
        )
        clear_col = find_column(
            columns,
            ["clear_path", "clear", "target_path", "clear_file", "gt_path"],
            "clear image path",
        )
        split_col = find_column(
            columns,
            ["split", "subset", "partition"],
            "split",
        )

        split_aliases = {
            "train": {"train", "training"},
            "val": {"val", "valid", "validation"},
            "test": {"test", "testing"},
        }

        accepted_splits = split_aliases.get(split.lower(), {split.lower()})

        self.samples = [
            (row[cloudy_col], row[clear_col])
            for row in rows
            if str(row[split_col]).strip().lower() in accepted_splits
        ]

        if not self.samples:
            raise ValueError(
                f"No samples found for split={split!r}. "
                "Check the manifest's split values."
            )

        print(f"{split}: {len(self.samples):,} patches")

    def resolve_path(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root / path

    def read_image(self, relative_path):
        path = self.resolve_path(relative_path)

        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)

        # Fixed scaling preserves relative brightness between patches.
        # Assumes source reflectance values are in the 0..1023 range.
        image = np.clip(image, 0, 1023) / 1023.0
        image = image * 2.0 - 1.0

        return torch.from_numpy(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cloudy_path, clear_path = self.samples[index]

        cloudy = self.read_image(cloudy_path)
        clear = self.read_image(clear_path)

        if cloudy.shape != clear.shape:
            raise ValueError(
                f"Pair shape mismatch: {cloudy.shape} vs {clear.shape}"
            )

        if cloudy.shape != (3, 256, 256):
            raise ValueError(
                f"Expected (3, 256, 256), got {cloudy.shape}"
            )

        return cloudy, clear


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_dataset = SatellitePatchDataset(MANIFEST, "train")
val_dataset = SatellitePatchDataset(MANIFEST, "val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)


# --------------------------------------------------
# Model, diffusion, optimizer
# --------------------------------------------------

model = DiffCRModel().to(device)
diffusion = Diffusion(device=device)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda"),
)

best_val_loss = float("inf")
start_epoch = 0


# --------------------------------------------------
# Resume training if a checkpoint exists
# --------------------------------------------------

resume_path = CHECKPOINT_DIR / "last_checkpoint.pth"

if resume_path.exists():
    checkpoint = torch.load(
        resume_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"Resuming from epoch {start_epoch + 1}")


# --------------------------------------------------
# Training and validation
# --------------------------------------------------
def main():

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for cloudy, clear in train_loader:
            cloudy = cloudy.to(device, non_blocking=True)
            clear = clear.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                loss, _ = diffusion.training_loss(model, cloudy, clear)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * cloudy.size(0)

        train_loss = train_loss_sum / len(train_dataset)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for cloudy, clear in val_loader:
                cloudy = cloudy.to(device, non_blocking=True)
                clear = clear.to(device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    loss, _ = diffusion.training_loss(model, cloudy, clear)

                val_loss_sum += loss.item() * cloudy.size(0)

        val_loss = val_loss_sum / len(val_dataset)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train L1: {train_loss:.6f} | "
            f"Val L1: {val_loss:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": min(best_val_loss, val_loss),
        }

        torch.save(checkpoint, resume_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint["best_val_loss"] = best_val_loss

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best_model.pth",
            )

            print("Saved new best model.")
if __name__ == "__main__":
    main()
    print("Training complete.")
