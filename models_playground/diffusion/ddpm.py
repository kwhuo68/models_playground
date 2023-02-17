from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model (DDPM)"""

    def __init__(self, beta_start, beta_end, timesteps=1000):
        """
        DDPM

        :param beta_start: starting beta value
        :param beta_end: ending beta value
        :param timesteps: number of timesteps
        """
        super().__init__()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        self.betas = self.linear_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_inv_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def linear_beta_schedule(self):
        """
        Get beta schedule with linearly spacing
        """
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

    def noise_images(self, x, t):
        """
        Add noise to images

        :param x: input tensor of shape (b, c, h, w)
        :param t: current time step
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_cumprod[t])[
            :, None, None, None
        ]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def extract(self, a, t, x_shape):
        """
        Extract from tensor at specific timestep

        :param a: tensor to extract from
        :param t: timestep to extract
        :param x_shape: shape of x
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0)

        :param x_0: starting x
        :param t: current timestep
        :param noise: noise to use
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.extract(self.alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Sample from p(x_t | x_0)

        :param: model: model to sample from
        :param x: input tensor
        :param t: current timestep
        :param t_index: index of current timestep
        """
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_inv_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Run p_sample in a loop

        :param: model: model to sample from
        :param shape: shape of image to sample
        """

        b = shape[0]
        img = torch.randn(shape)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop timestep",
            total=self.timesteps,
        ):
            img = self.p_sample(model, img, torch.full((b,), i, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        """
        Run sample via p_sample_loop

        :param model: model
        :param image_size: size of image to sample
        :param batch_size: batch size
        :param channels: number of channels
        """
        return self.p_sample_loop(
            model, shape=(batch_size, channels, image_size, image_size)
        )

    def p_losses(self, denoise_model, x_0, t, noise=None, loss_type="l1"):
        """
        Compute p-losses

        :param denoise_model: denoising model
        :param x_0: starting x
        :param t: current timestep
        :param noise: noise
        :param loss_type: loss type
        """

        if noise is None:
            noise = torch.randn_like(x_0)

        x_noisy = self.q_sample(x_0=x_0, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, img, *args, **kwargs):
        """
        Apply forward pass for DDPM

        :param img: image
        :param args: args
        :param kwargs: kwargs
        """
        b, c, h, w = img.shape
        img_size = self.image_size
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,)).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
