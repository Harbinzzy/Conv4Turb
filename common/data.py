import csv
from pathlib import Path

import numpy as np
import torch


DX_MIN = 3.906250185536919e-06
FIELD_NAMES = (
    "RHO_kgm-3_id",
    "UX_ms-1_id",
    "UY_ms-1_id",
    "UZ_ms-1_id",
)

DATASET_STATS = {
    "train": (
        torch.tensor([0.24, 28.0, 28.0, 28.0]),
        torch.tensor([0.068, 48.0, 48.0, 48.0]),
    ),
    "test": (
        torch.tensor([0.24, 29.0, 29.0, 29.0]),
        torch.tensor([0.068, 48.0, 48.0, 48.0]),
    ),
    "paramvar": (
        torch.tensor([0.23, 34.0, 34.0, 34.0]),
        torch.tensor([0.059, 55.0, 55.0, 55.0]),
    ),
    "forcedhit": (
        torch.tensor([11.0, -0.051, -0.051, -0.051]),
        torch.tensor([4.6, 1.4, 1.4, 1.4]),
    ),
}


def get_stats(split):
    try:
        mean, std = DATASET_STATS[split]
    except KeyError as exc:
        choices = ", ".join(DATASET_STATS)
        raise ValueError(f"Unknown data split {split!r}; choose from {choices}") from exc
    return mean.clone(), std.clone()


def read_metadata(path):
    path = Path(path).expanduser()
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Metadata file has no header: {path}")
        rows = {column: [] for column in reader.fieldnames}
        for row in reader:
            for column in reader.fieldnames:
                rows[column].append(row[column])

    required = {"hash", "dx [m]"}
    missing = required.difference(rows)
    if missing:
        raise ValueError(
            f"Metadata file {path} is missing columns: {', '.join(sorted(missing))}"
        )
    return rows


def _spacing(metadata, key, index, fallback):
    values = metadata.get(key)
    if values is None or values[index] == "":
        return fallback
    return torch.tensor(np.float32(values[index]))


def load_sample(index, metadata, data_path, split, upscale):
    sample_hash = metadata["hash"][index]
    data_path = Path(data_path).expanduser()
    lr_side = 128 // upscale

    low_resolution = []
    high_resolution = []
    for field in FIELD_NAMES:
        lr_path = data_path / f"LR_{upscale}x" / split / f"{field}{sample_hash}.dat"
        hr_path = data_path / "HR" / split / f"{field}{sample_hash}.dat"
        low_resolution.append(
            np.memmap(lr_path, dtype=np.float32, mode="r").reshape(
                lr_side, lr_side, lr_side
            )
        )
        high_resolution.append(
            np.memmap(hr_path, dtype=np.float32, mode="r").reshape(128, 128, 128)
        )

    x = torch.from_numpy(np.stack(low_resolution, axis=0))
    y = torch.from_numpy(np.stack(high_resolution, axis=0))
    dx = torch.tensor(np.float32(metadata["dx [m]"][index]))
    dy = _spacing(metadata, "dy [m]", index, dx)
    dz = _spacing(metadata, "dz [m]", index, dx)
    return x, y, dx, dy, dz


def _rotate_vector_volume(volume, k, axes):
    spatial_dims = (axes[0] + 1, axes[1] + 1)
    for _ in range(k):
        volume = torch.rot90(volume, k=1, dims=spatial_dims)
        volume[[spatial_dims[0], spatial_dims[1]]] = volume[
            [spatial_dims[1], spatial_dims[0]]
        ]
        volume[spatial_dims[0]] = -volume[spatial_dims[0]]
    return volume


def _rotate_spacing(spacing, k, axes):
    for _ in range(k):
        spacing[[axes[0], axes[1]]] = spacing[[axes[1], axes[0]]]
    return spacing


def random_rotate(x, y, dx, dy, dz):
    spacing = torch.stack((dx, dy, dz))
    k = np.random.randint(0, 4)
    axes = np.random.choice((0, 1, 2), 2, replace=False)
    x = _rotate_vector_volume(x, k, axes)
    y = _rotate_vector_volume(y, k, axes)
    dx, dy, dz = _rotate_spacing(spacing, k, axes)
    return x, y, dx, dy, dz


def _flip_vector_volume(volume, axis):
    channel_and_spatial_dim = axis + 1
    volume = torch.flip(volume, dims=(channel_and_spatial_dim,))
    volume[channel_and_spatial_dim] = -volume[channel_and_spatial_dim]
    return volume


def random_flip(x, y):
    for axis in range(3):
        if np.random.rand() > 0.5:
            x = _flip_vector_volume(x, axis)
            y = _flip_vector_volume(y, axis)
    return x, y


class ScaleTransform:
    def __init__(self, mean, std):
        self.mean = mean[:, None, None, None]
        self.std = std[:, None, None, None]

    def __call__(self, sample):
        return (sample - self.mean) / self.std


class TurbulenceDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, data_path, split, upscale, transform=None):
        if 128 % upscale != 0:
            raise ValueError(f"upscale must divide 128, got {upscale}")
        self.metadata = metadata
        self.data_path = data_path
        self.split = split
        self.upscale = upscale
        self.transform = transform

    def __len__(self):
        return len(self.metadata["hash"])

    def __getitem__(self, index):
        x, y, dx, dy, dz = load_sample(
            index, self.metadata, self.data_path, self.split, self.upscale
        )
        dx, dy, dz = dx / DX_MIN, dy / DX_MIN, dz / DX_MIN

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        if self.split == "train":
            x, y, dx, dy, dz = random_rotate(x, y, dx, dy, dz)
            x, y = random_flip(x, y)
        return x, y, dx, dy, dz
