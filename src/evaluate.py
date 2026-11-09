import numpy as np
import torch
from skimage.metrics import structural_similarity


def evaluate_metrics(prediction, target):
    prediction = ((prediction.float() + 1) / 2).clamp(0, 1)
    target = ((target.float() + 1) / 2).clamp(0, 1)

    pred = prediction.detach().cpu().numpy()
    truth = target.detach().cpu().numpy()

    mae_scores = []
    psnr_scores = []
    ssim_scores = []
    sam_scores = []

    for i in range(len(pred)):
        p = pred[i].transpose(1, 2, 0)
        t = truth[i].transpose(1, 2, 0)

        mae = np.mean(np.abs(p - t))
        mse = np.mean((p - t) ** 2)
        psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)

        ssim = structural_similarity(
            t, p, channel_axis=-1, data_range=1.0
        )

        p_spectra = p.reshape(-1, 3)
        t_spectra = t.reshape(-1, 3)

        p_norm = np.linalg.norm(p_spectra, axis=1)
        t_norm = np.linalg.norm(t_spectra, axis=1)
        valid = (p_norm > 1e-8) & (t_norm > 1e-8)

        if valid.any():
            cosine = np.sum(
                p_spectra[valid] * t_spectra[valid], axis=1
            ) / (p_norm[valid] * t_norm[valid])

            sam = np.degrees(
                np.arccos(np.clip(cosine, -1.0, 1.0))
            ).mean()
        else:
            sam = np.nan

        mae_scores.append(mae)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        sam_scores.append(sam)

    return {
        "MAE": float(np.mean(mae_scores)),
        "PSNR": float(np.mean(psnr_scores)),
        "SSIM": float(np.mean(ssim_scores)),
        "SAM": float(np.nanmean(sam_scores))
    }