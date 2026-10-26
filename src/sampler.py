#The sampler handles inference: generating a reconstructed clear image from a cloudy image.

import torch


class DiffusionSampler:
    def __init__(self, diffusion):
        self.diffusion = diffusion

    @torch.no_grad()
    def sample(self, model, cloudy, sampling_steps=20, initial_noise=None):
        """
        Reconstruct a clear image conditioned on a cloudy image.

        cloudy:       (B, C, H, W)
        sampling_steps: Number of reverse diffusion steps
        returns:      Predicted clear image (B, C, H, W)
        """
        if sampling_steps < 1 or sampling_steps > self.diffusion.timesteps:
            raise ValueError(
                "sampling_steps must be between 1 and diffusion.timesteps"
            )

        device = cloudy.device
        batch_size, channels, height, width = cloudy.shape

        if initial_noise is None:
            x = torch.randn(
                batch_size, channels, height, width,
                device=device, dtype=cloudy.dtype
            )
        else:
            if initial_noise.shape != cloudy.shape:
                raise ValueError(
                    "initial_noise must have the same shape as cloudy"
                )
            x = initial_noise.to(device=device, dtype=cloudy.dtype)

        # Select timesteps from the final noisy state down to zero.
        times = torch.linspace(
            self.diffusion.timesteps - 1,
            0,
            sampling_steps,
            device=device,
        ).long().unique_consecutive().tolist()

        was_training = model.training
        model.eval()

        try:
            for i, t in enumerate(times):
                timesteps = torch.full(
                    (batch_size,), t, device=device, dtype=torch.long
                )

                # Predict the clean image x_0 from the current noisy image.
                model_input = torch.cat([cloudy, x], dim=1)
                predicted_x0 = model(model_input, timesteps)

                # DDIM-style deterministic update to the next selected time.
                if i == len(times) - 1:
                    x = predicted_x0
                    continue

                next_t = times[i + 1]

                alpha_bar_t = self.diffusion.alpha_bars[t].to(device)
                alpha_bar_next = self.diffusion.alpha_bars[next_t].to(device)

                # Estimate the noise component using the x_0 prediction.
                predicted_noise = (
                    x - torch.sqrt(alpha_bar_t) * predicted_x0
                ) / torch.sqrt(1.0 - alpha_bar_t).clamp_min(1e-8)

                # Move to the next, less noisy timestep.
                x = (
                    torch.sqrt(alpha_bar_next) * predicted_x0
                    + torch.sqrt(1.0 - alpha_bar_next) * predicted_noise
                )

        finally:
            model.train(was_training)

        return x
