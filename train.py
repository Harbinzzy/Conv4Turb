import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from common.data import ScaleTransform, TurbulenceDataset, get_stats, read_metadata
from models import MODEL_NAMES, create_model
from models.tasks import TrainingTask


def parse_devices(value):
    if value == "auto":
        return "auto"
    try:
        return [int(device.strip()) for device in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "devices must be 'auto' or comma-separated GPU indices, for example 0,1"
        ) from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a Conv4Turb model on 3D turbulence super-resolution data."
    )
    parser.add_argument("--model-type", choices=MODEL_NAMES, default="edsr")
    parser.add_argument("--approx-param", default="0.5M")
    parser.add_argument("--upscale", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loss-type", choices=("mse", "l1", "mse_grad"), default="mse_grad")
    parser.add_argument("--grad-loss-weight", type=float, default=0.99)
    parser.add_argument("--lr-step-size", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--val-num-workers", type=int, default=0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--train-meta", default="/data1t/blastnet/train_data_summary.csv"
    )
    parser.add_argument(
        "--val-meta", default="/data1t/blastnet/test_data_summary.csv"
    )
    parser.add_argument("--data-path", default="/data1t/blastnet")

    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--save-top-k", type=int, default=3)

    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=parse_devices, default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser


def configure_reproducibility(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    pl.seed_everything(seed, workers=True)


def main():
    args = build_parser().parse_args()
    if not 0.0 <= args.grad_loss_weight <= 1.0:
        raise ValueError("--grad-loss-weight must be between 0 and 1")
    if args.save_period < 1:
        raise ValueError("--save-period must be at least 1")
    if args.lr_step_size < 1:
        raise ValueError("--lr-step-size must be at least 1")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")

    configure_reproducibility(args.seed)
    train_mean, train_std = get_stats("train")
    val_mean, val_std = get_stats("test")
    train_dataset = TurbulenceDataset(
        read_metadata(args.train_meta),
        args.data_path,
        "train",
        args.upscale,
        ScaleTransform(train_mean, train_std),
    )
    val_dataset = TurbulenceDataset(
        read_metadata(args.val_meta),
        args.data_path,
        "test",
        args.upscale,
        ScaleTransform(val_mean, val_std),
    )
    pin_memory = torch.cuda.is_available() and args.accelerator != "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.val_num_workers > 0,
    )

    architecture = create_model(args.model_type, args.approx_param, args.upscale)
    task = TrainingTask(
        architecture,
        train_mean,
        train_std,
        val_mean,
        val_std,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        grad_loss_weight=args.grad_loss_weight,
        lr_step_size=args.lr_step_size,
    )

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=(
            f"{args.model_type}-{args.approx_param}-{args.upscale}x"
            "-{epoch:04d}-{val_avg_ssim:.6f}"
        ),
        monitor="val_avg_ssim",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        every_n_epochs=args.save_period,
        save_on_train_epoch_end=False,
    )
    trainer = pl.Trainer(
        logger=False,
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        deterministic="warn",
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.save_period,
        log_every_n_steps=1,
        enable_progress_bar=True,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(
        task,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(args.resume) if args.resume is not None else None,
    )


if __name__ == "__main__":
    main()
