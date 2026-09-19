from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from common.data import DX_MIN
from common.metrics import GradLoss, SSIM3D
from common.physics import divergence_sgs, nrmse, remove_edges, sgs


class TrainingTask(pl.LightningModule):
    def __init__(
        self,
        model,
        train_mean,
        train_std,
        val_mean,
        val_std,
        learning_rate=1e-4,
        loss_type="mse_grad",
        grad_loss_weight=0.99,
        lr_step_size=150,
    ):
        super().__init__()
        self.model = model
        self.upscale = model.upscale
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.grad_loss_weight = grad_loss_weight
        self.lr_step_size = lr_step_size
        self.grad_loss = GradLoss()
        self.ssim = SSIM3D()
        self.register_buffer(
            "train_mean",
            train_mean[None, :, None, None, None],
            persistent=False,
        )
        self.register_buffer(
            "train_std",
            train_std[None, :, None, None, None],
            persistent=False,
        )
        self.register_buffer(
            "val_mean",
            val_mean[None, :, None, None, None],
            persistent=False,
        )
        self.register_buffer(
            "val_std",
            val_std[None, :, None, None, None],
            persistent=False,
        )

    def training_step(self, batch, batch_idx):
        x, target, _, _, _ = batch
        prediction = self.model(x)
        if self.loss_type == "mse":
            loss = F.mse_loss(prediction, target)
        elif self.loss_type == "l1":
            loss = F.l1_loss(prediction, target)
        elif self.loss_type == "mse_grad":
            mse = (1.0 - self.grad_loss_weight) * F.mse_loss(prediction, target)
            gradient = self.grad_loss_weight * self.grad_loss(prediction, target)
            loss = mse + gradient
            self.log(
                "train_mse_weighted",
                mse,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_grad_weighted",
                gradient,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, target, dx, dy, dz = batch
        with torch.autocast(device_type=x.device.type, enabled=False):
            prediction = self.model(x.float()).float()
            target = target.float()
            mean = self.val_mean.to(dx)
            std = self.val_std.to(dx)
            prediction_physical = prediction * std + mean
            target_physical = target * std + mean

            dx = dx.float() * DX_MIN * self.upscale
            dy = dy.float() * DX_MIN * self.upscale
            dz = dz.float() * DX_MIN * self.upscale
            stress_prediction = sgs(prediction_physical, self.upscale)
            stress_target = sgs(target_physical, self.upscale)
            div_prediction = divergence_sgs(stress_prediction, dx, dy, dz)
            div_target = divergence_sgs(stress_target, dx, dy, dz)

            field_scores = [
                self.ssim(
                    remove_edges(prediction_physical[:, channel : channel + 1]),
                    remove_edges(target_physical[:, channel : channel + 1]),
                )
                for channel in range(4)
            ]
            div_scores = [
                self.ssim(remove_edges(pred), remove_edges(truth))
                for pred, truth in zip(div_prediction, div_target)
            ]
            avg_field_ssim = torch.stack(field_scores).mean()
            avg_div_ssim = torch.stack(div_scores).mean()

        self.log(
            "val_avg_ssim",
            avg_field_ssim,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        self.log(
            "val_div_sgs_ssim",
            avg_div_ssim,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_train_epoch_end(self):
        learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", learning_rate, sync_dist=False)

    def forward(self, x):
        return self.model(x)


class EvaluationTask(pl.LightningModule):
    FIELD_NAMES = ("rho", "ux", "uy", "uz")

    def __init__(
        self,
        model,
        mean,
        std,
        output_dir,
        split,
        plot_spectrum=True,
        spectrum_channel="ux",
    ):
        super().__init__()
        self.model = model
        self.upscale = model.upscale
        self.output_dir = Path(output_dir)
        self.split = split
        self.plot_spectrum = plot_spectrum
        self.spectrum_channel = spectrum_channel
        self.ssim = SSIM3D()
        self.register_buffer(
            "mean", mean[None, :, None, None, None], persistent=False
        )
        self.register_buffer(
            "std", std[None, :, None, None, None], persistent=False
        )
        self._spectrum_prediction = None
        self._spectrum_target = None
        self._spectrum_input = None
        self._spectrum_count = 0

    @staticmethod
    def _field_snr(prediction, target, eps=1e-12):
        nmse = nrmse(prediction, target, eps=eps).square()
        return -10.0 * torch.log10(torch.clamp(nmse, min=eps))

    @staticmethod
    def _power_spectrum(volume, dim=4):
        volume = volume - volume.mean(dim=dim, keepdim=True)
        frequency = torch.fft.rfft(volume, dim=dim)
        power = frequency.real.square() + frequency.imag.square()
        reduce_dims = tuple(index for index in range(power.ndim) if index != dim)
        return power.mean(dim=reduce_dims) / volume.shape[dim]

    def _accumulate_spectrum(self, prediction, target, input_upsampled):
        channel = self.FIELD_NAMES.index(self.spectrum_channel)
        pred_power = self._power_spectrum(prediction[:, channel : channel + 1]).detach()
        target_power = self._power_spectrum(target[:, channel : channel + 1]).detach()
        input_power = self._power_spectrum(
            input_upsampled[:, channel : channel + 1]
        ).detach()
        if self._spectrum_prediction is None:
            self._spectrum_prediction = torch.zeros_like(pred_power)
            self._spectrum_target = torch.zeros_like(target_power)
            self._spectrum_input = torch.zeros_like(input_power)
        self._spectrum_prediction += pred_power
        self._spectrum_target += target_power
        self._spectrum_input += input_power
        self._spectrum_count += 1

    def test_step(self, batch, batch_idx):
        x, target, dx, dy, dz = batch
        prediction = self.model(x)
        mean = self.mean.to(dx)
        std = self.std.to(dx)
        prediction_physical = prediction * std + mean
        target_physical = target * std + mean

        if self.plot_spectrum:
            input_physical = x * std + mean
            input_upsampled = F.interpolate(
                input_physical,
                scale_factor=self.upscale,
                mode="trilinear",
                align_corners=False,
            )
            self._accumulate_spectrum(
                prediction_physical, target_physical, input_upsampled
            )

        dx = dx * DX_MIN * self.upscale
        dy = dy * DX_MIN * self.upscale
        dz = dz * DX_MIN * self.upscale
        stress_prediction = sgs(prediction_physical, self.upscale)
        stress_target = sgs(target_physical, self.upscale)
        div_prediction = divergence_sgs(stress_prediction, dx, dy, dz)
        div_target = divergence_sgs(stress_target, dx, dy, dz)

        prediction_cropped = remove_edges(prediction_physical)
        target_cropped = remove_edges(target_physical)
        for index, name in enumerate(self.FIELD_NAMES):
            pred = prediction_cropped[:, index : index + 1]
            truth = target_cropped[:, index : index + 1]
            self.log(
                f"test_ssim_{name}",
                self.ssim(pred, truth),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
            self.log(
                f"test_snr_{name}",
                self._field_snr(pred, truth),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )

        for index, (pred, truth) in enumerate(zip(div_prediction, div_target), 1):
            pred = remove_edges(pred)
            truth = remove_edges(truth)
            self.log(
                f"test_ssim_divsgs{index}",
                self.ssim(pred, truth),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
            self.log(
                f"test_snr_divsgs{index}",
                self._field_snr(pred, truth),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )

    def on_test_epoch_end(self):
        if (
            not self.plot_spectrum
            or self._spectrum_target is None
            or self._spectrum_count == 0
            or not self.trainer.is_global_zero
        ):
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = (self._spectrum_target / self._spectrum_count).float().cpu()[1:]
        prediction = (
            self._spectrum_prediction / self._spectrum_count
        ).float().cpu()[1:]
        low_resolution = (
            self._spectrum_input / self._spectrum_count
        ).float().cpu()[1:]
        wave_number = torch.arange(1, target.numel() + 1, dtype=torch.float32)

        figure, axis = plt.subplots(figsize=(7, 5))
        axis.loglog(wave_number, target.clamp_min(1e-20), label="HR target")
        axis.loglog(wave_number, prediction.clamp_min(1e-20), label="prediction")
        axis.loglog(wave_number, low_resolution.clamp_min(1e-20), label="LR input")
        axis.set_xlabel("wave number")
        axis.set_ylabel("power")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            self.output_dir / f"{self.split}_spectrum_{self.spectrum_channel}.png",
            dpi=180,
        )
        plt.close(figure)

    def forward(self, x):
        return self.model(x)
