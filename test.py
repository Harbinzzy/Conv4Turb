import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from common.data import ScaleTransform, TurbulenceDataset, get_stats, read_metadata
from models import MODEL_NAMES, create_model
from models.tasks import EvaluationTask
from train import configure_reproducibility, parse_devices


SPLITS = ("test", "paramvar", "forcedhit")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a Conv4Turb checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=MODEL_NAMES, default="edsr")
    parser.add_argument("--approx-param", default="0.5M")
    parser.add_argument("--upscale", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--test-data", choices=("all", *SPLITS), default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data-path", default="/data1t/blastnet")
    parser.add_argument(
        "--test-meta", default="/data1t/blastnet/test_data_summary.csv"
    )
    parser.add_argument(
        "--paramvar-meta", default="/data1t/blastnet/paramvar_data_summary.csv"
    )
    parser.add_argument(
        "--forcedhit-meta", default="/data1t/blastnet/forcedhit_data_summary.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval"))

    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--devices", type=parse_devices, default=[0])
    parser.add_argument("--precision", default="32-true")
    parser.add_argument(
        "--spectrum-channel", choices=("rho", "ux", "uy", "uz"), default="ux"
    )
    parser.add_argument(
        "--no-spectrum",
        dest="plot_spectrum",
        action="store_false",
        help="disable power-spectrum plots",
    )
    parser.set_defaults(plot_spectrum=True)
    return parser


def load_architecture_weights(model, checkpoint_path):
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no state dictionary: {checkpoint_path}")

    if any(key.startswith("model.") for key in state_dict):
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    elif any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)


def main():
    args = build_parser().parse_args()
    configure_reproducibility(args.seed)
    architecture = create_model(args.model_type, args.approx_param, args.upscale)
    load_architecture_weights(architecture, args.checkpoint)
    architecture.eval()

    selected_splits = SPLITS if args.test_data == "all" else (args.test_data,)
    metadata_paths = {
        "test": args.test_meta,
        "paramvar": args.paramvar_meta,
        "forcedhit": args.forcedhit_meta,
    }
    pin_memory = torch.cuda.is_available() and args.accelerator != "cpu"
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        deterministic=True,
        enable_progress_bar=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for split in selected_splits:
        mean, std = get_stats(split)
        dataset = TurbulenceDataset(
            read_metadata(metadata_paths[split]),
            args.data_path,
            split,
            args.upscale,
            ScaleTransform(mean, std),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        task = EvaluationTask(
            architecture,
            mean,
            std,
            args.output_dir,
            split,
            plot_spectrum=args.plot_spectrum,
            spectrum_channel=args.spectrum_channel,
        )
        result = trainer.test(task, dataloaders=loader, verbose=True)
        all_results[split] = result[0] if result else {}

    result_path = args.output_dir / (
        f"{args.model_type}_{args.approx_param}_{args.upscale}x_metrics.json"
    )
    with result_path.open("w") as file:
        json.dump(all_results, file, indent=2)
    print(f"Metrics written to {result_path}")


if __name__ == "__main__":
    main()
