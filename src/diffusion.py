
import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=None,
    ):
        self.timesteps = timesteps
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.betas = torch.linspace(
            beta_start, beta_end, timesteps
        ).to(self.device)

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0 - self.alpha_bars
        )

    def _extract(self, values, timesteps, target_shape):
        values = values.to(timesteps.device)
        result = values.gather(0, timesteps)

        return result.reshape(
            timesteps.shape[0],
            *([1] * (len(target_shape) - 1))
        )

    def q_sample(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = self._extract(
            self.sqrt_alpha_bars, timesteps, x_start.shape
        )

        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            x_start.shape,
        )

        return (
            sqrt_alpha_bar * x_start
            + sqrt_one_minus_alpha_bar * noise
        )

    def training_loss(self, model, cloudy, clear):
        if cloudy.shape != clear.shape:
            raise ValueError(
                f"cloudy and clear must have the same shape; "
                f"got {cloudy.shape} and {clear.shape}"
            )

        batch_size = clear.shape[0]

        timesteps = torch.randint(
            0,
            self.timesteps,
            (batch_size,),
            device=clear.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(clear)

        noisy_clear = self.q_sample(
            clear, timesteps, noise=noise
        )

        model_input = torch.cat(
            [cloudy, noisy_clear], dim=1
        )

        predicted_clear = model(model_input, timesteps)

        loss = F.l1_loss(predicted_clear, clear)

        return loss, {
            "timesteps": timesteps,
            "noisy_clear": noisy_clear,
            "predicted_clear": predicted_clear,
        }
