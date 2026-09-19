from math import exp

import torch
import torch.nn.functional as F
from torch import nn

from .physics import torch_dx, torch_dy, torch_dz


def _gaussian(window_size, sigma):
    values = torch.tensor(
        [
            exp(-((index - window_size // 2) ** 2) / (2 * sigma**2))
            for index in range(window_size)
        ]
    )
    return values / values.sum()


def _create_window(window_size, channels):
    window_1d = _gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t())
    window_3d = window_1d.mm(window_2d.reshape(1, -1)).reshape(
        window_size, window_size, window_size
    )
    return (
        window_3d.float()
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(channels, 1, window_size, window_size, window_size)
        .contiguous()
    )


class SSIM3D(nn.Module):
    def __init__(self, window_size=9):
        super().__init__()
        self.window_size = window_size
        self.channels = 1
        self.register_buffer(
            "window", _create_window(window_size, self.channels), persistent=False
        )

    def forward(self, prediction, target):
        channels = prediction.shape[1]
        if (
            channels != self.channels
            or self.window.dtype != prediction.dtype
            or self.window.device != prediction.device
        ):
            self.window = _create_window(self.window_size, channels).to(prediction)
            self.channels = channels

        padding = self.window_size // 2
        mean_prediction = F.conv3d(
            prediction, self.window, padding=padding, groups=channels
        )
        mean_target = F.conv3d(target, self.window, padding=padding, groups=channels)
        mean_prediction_sq = mean_prediction.square()
        mean_target_sq = mean_target.square()
        mean_product = mean_prediction * mean_target

        variance_prediction = (
            F.conv3d(prediction.square(), self.window, padding=padding, groups=channels)
            - mean_prediction_sq
        )
        variance_target = (
            F.conv3d(target.square(), self.window, padding=padding, groups=channels)
            - mean_target_sq
        )
        covariance = (
            F.conv3d(prediction * target, self.window, padding=padding, groups=channels)
            - mean_product
        )

        c1, c2 = 0.1**2, 0.3**2
        score = ((2 * mean_product + c1) * (2 * covariance + c2)) / (
            (mean_prediction_sq + mean_target_sq + c1)
            * (variance_prediction + variance_target + c2)
        )
        return score.mean()


class GradLoss(nn.Module):
    def forward(self, prediction, target):
        spacing = torch.ones(prediction.shape[0], device=prediction.device)
        loss = prediction.new_zeros(())
        for channel in range(4):
            pred = prediction[:, channel : channel + 1]
            truth = target[:, channel : channel + 1]
            loss = loss + F.mse_loss(torch_dx(pred, spacing), torch_dx(truth, spacing))
            loss = loss + F.mse_loss(torch_dy(pred, spacing), torch_dy(truth, spacing))
            loss = loss + F.mse_loss(torch_dz(pred, spacing), torch_dz(truth, spacing))
        return loss
