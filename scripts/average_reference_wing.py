#!/usr/bin/env python3
"""Build an average reference wing using SliceLearn's normalization.

The normalization deliberately mirrors ``slicelearn_code.py``:

1. the shared dataset-metadata masking rule identifies foreground and fills holes;
2. the mask is centred on an S x S canvas;
3. centroid, covariance, and principal axes define a similarity transform;
4. the shape is scaled to TARGET_SIZE and resampled by nearest neighbour;
5. the same x < ROOT_CROP_X wing-root region is removed.

The pixelwise mean of the normalized masks is saved as an 8-bit probability
image. A majority-vote mask (mean >= threshold) is saved as the binary average
reference wing. The script also creates an overlay of every normalized
training-wing outline with the reference outline on top.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from slicelearn_dataset import (
    AVERAGE_REFERENCE_ROOT,
    DATASET_SPLITS,
    SLIDES_UNDAMAGED,
    load_segmentation_mask,
    split_filenames,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SLIDES_UNDAMAGED
DEFAULT_OUTPUT = AVERAGE_REFERENCE_ROOT

S = 1001
TARGET_SIZE = 0.4
ROOT_CROP_X = -0.8
FLIP = (-1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing original-coordinate wing segmentations.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Column in dataset_splits.csv to select (default: train; use all for every PNG).",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=DATASET_SPLITS,
        help="Dataset split CSV used unless --split all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory in which results will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Mean occupancy threshold for the binary reference (default: 0.5).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the overlay without opening an interactive plot window.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def pad_wing(path: Path, size: int = S) -> np.ndarray:
    """Load and pad a wing exactly as in SliceLearn."""
    binary = load_segmentation_mask(path)

    height, width = binary.shape
    if height > size or width > size:
        raise ValueError(
            f"{path.name}: image ({height}, {width}) exceeds canvas S={size}"
        )

    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left
    return np.pad(binary, ((top, bottom), (left, right)), mode="constant")


def make_square_grid(size: int = S) -> np.ndarray:
    values = np.linspace(-1, 1, size)
    x_grid, y_grid = np.meshgrid(values, values)
    return np.stack([x_grid, y_grid], axis=-1)


def move_image_to_grid(grid: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Attach image values to the grid, converting to mathematical y order."""
    return np.concatenate(
        [grid, image[::-1, :, None].astype(np.uint8)],
        axis=-1,
    )


def compute_alignment(
    shape: np.ndarray,
    target_size: float = TARGET_SIZE,
    flip: tuple[int, int] = FLIP,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return SliceLearn's scale, rotation, and translation."""
    if len(shape) == 0:
        raise ValueError("cannot normalize an empty wing mask")

    centroid = shape.mean(axis=0)
    centred = shape - centroid
    covariance = (centred.T @ centred) / len(shape)
    shape_size = float(np.sqrt(np.trace(covariance)))
    if shape_size == 0:
        raise ValueError("cannot normalize a zero-size wing mask")

    scale = target_size / shape_size
    _, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = eigenvectors[:, [1, 0]]
    eigenvectors[:, 0] *= flip[0]
    eigenvectors[:, 1] *= flip[1]
    rotation = eigenvectors.T
    translation = -scale * (rotation @ centroid)
    return scale, rotation, translation


def transform_on_set_grid(
    grid: np.ndarray,
    image_grid: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply the same nearest-neighbour inverse warp as SliceLearn."""
    output_coordinates = grid.reshape(-1, 2)
    source_queries = ((output_coordinates - translation) @ rotation) / scale

    # SliceLearn uses a KDTree nearest-neighbour query here. Because its source
    # coordinates are a regular grid, rounding directly to the nearest row and
    # column gives the same nearest-neighbour resampling without rebuilding and
    # querying a million-point tree for every wing.
    size = grid.shape[0]
    columns = np.rint((source_queries[:, 0] + 1) * (size - 1) / 2)
    rows = np.rint((source_queries[:, 1] + 1) * (size - 1) / 2)
    columns = np.clip(columns, 0, size - 1).astype(np.intp)
    rows = np.clip(rows, 0, size - 1).astype(np.intp)
    return image_grid[rows, columns, 2].reshape(grid.shape[:2]).astype(np.uint8)


def transform_values_on_set_grid(
    grid: np.ndarray,
    values: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour warp scalar or multichannel values on the same grid."""
    output_coordinates = grid.reshape(-1, 2)
    source_queries = ((output_coordinates - translation) @ rotation) / scale
    size = grid.shape[0]
    columns = np.rint((source_queries[:, 0] + 1) * (size - 1) / 2)
    rows = np.rint((source_queries[:, 1] + 1) * (size - 1) / 2)
    columns = np.clip(columns, 0, size - 1).astype(np.intp)
    rows = np.clip(rows, 0, size - 1).astype(np.intp)
    output_shape = grid.shape[:2] + values.shape[2:]
    return values[rows, columns].reshape(output_shape)


def normalize_wing(path: Path, grid: np.ndarray) -> np.ndarray:
    """Normalize one mask and apply SliceLearn's root crop."""
    padded = pad_wing(path, grid.shape[0])
    image_grid = move_image_to_grid(grid, padded)
    shape = image_grid[image_grid[:, :, 2] > 0][:, :2]
    scale, rotation, translation = compute_alignment(shape)
    normalized = transform_on_set_grid(
        grid, image_grid, scale, rotation, translation
    )
    normalized[grid[:, :, 0] < ROOT_CROP_X] = 0
    return normalized


def normalize_wing_segmentation(path: Path, grid: np.ndarray) -> np.ndarray:
    """Normalize an RGB segmentation with its mask-derived shape transform."""
    binary = load_segmentation_mask(path)
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    size = grid.shape[0]
    height, width = binary.shape
    if height > size or width > size:
        raise ValueError(
            f"{path.name}: image ({height}, {width}) exceeds canvas S={size}"
        )
    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left
    padded_mask = np.pad(binary, ((top, bottom), (left, right)))
    padded_rgb = np.pad(
        rgb,
        ((top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    image_grid = move_image_to_grid(grid, padded_mask)
    shape = image_grid[image_grid[:, :, 2] > 0][:, :2]
    scale, rotation, translation = compute_alignment(shape)
    normalized = transform_values_on_set_grid(
        grid,
        padded_rgb[::-1],
        scale,
        rotation,
        translation,
    ).astype(np.uint8)
    normalized[grid[:, :, 0] < ROOT_CROP_X] = 255
    return normalized


def dice_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(bool)
    second = second.astype(bool)
    intersection = np.count_nonzero(first & second)
    return 2 * intersection / max(
        np.count_nonzero(first) + np.count_nonzero(second),
        1,
    )


def save_probability_image(mean_mask: np.ndarray, path: Path) -> None:
    """Save occupancy with white meaning high average wing occupancy."""
    image = np.round(np.clip(mean_mask, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image[::-1, :]).save(path)


def save_binary_wing(mask: np.ndarray, path: Path) -> None:
    """Save a black-wing-on-white-background mask like the source images."""
    image = (1 - mask.astype(np.uint8)) * 255
    Image.fromarray(image[::-1, :]).save(path)


def plot_overlay(
    grid: np.ndarray,
    normalized_masks: list[np.ndarray],
    reference: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    """Plot all training outlines in grey and the average reference in red."""
    x_grid, y_grid = grid[:, :, 0], grid[:, :, 1]
    figure, axis = plt.subplots(figsize=(10, 7))

    for mask in normalized_masks:
        axis.contour(
            x_grid,
            y_grid,
            mask,
            levels=[0.5],
            colors="0.55",
            linewidths=0.6,
            alpha=0.22,
        )

    axis.contourf(
        x_grid,
        y_grid,
        reference,
        levels=[0.5, 1.5],
        colors=["tab:red"],
        alpha=0.15,
    )
    axis.contour(
        x_grid,
        y_grid,
        reference,
        levels=[0.5],
        colors="tab:red",
        linewidths=2.5,
    )
    axis.axhline(0, color="black", linewidth=0.5, alpha=0.35)
    axis.axvline(0, color="black", linewidth=0.5, alpha=0.35)
    axis.set(
        xlim=(-1, 1),
        ylim=(-1, 1),
        xlabel="x",
        ylabel="y",
        title=(
            f"Average reference wing (red) over "
            f"{len(normalized_masks)} normalized training wings"
        ),
    )
    axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.split == "all":
        files = sorted(input_dir.glob("*.png"))
    else:
        files = [
            input_dir / filename
            for filename in split_filenames(args.split, args.split_csv.resolve())
        ]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files selected by {args.split!r} are missing "
                f"from {input_dir}; first missing file: {missing[0]}"
            )
    if not files:
        raise FileNotFoundError(f"No PNG training masks found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    grid = make_square_grid()
    normalized_masks: list[np.ndarray] = []

    for path in tqdm(files, desc="Normalizing training wings", unit="wing"):
        normalized_masks.append(normalize_wing(path, grid))

    mean_mask = np.mean(np.stack(normalized_masks, axis=0), axis=0)
    reference = (mean_mask >= args.threshold).astype(np.uint8)

    probability_path = output_dir / "average_reference_probability.png"
    reference_path = output_dir / "average_reference_wing.png"
    overlay_path = output_dir / "average_reference_overlay.png"
    reference_segmentation_path = (
        output_dir / "reference_wing_segmentation.png"
    )
    reference_source_path = output_dir / "reference_wing_source.csv"

    save_probability_image(mean_mask, probability_path)
    save_binary_wing(reference, reference_path)
    plot_overlay(
        grid,
        normalized_masks,
        reference,
        overlay_path,
        show=not args.no_show,
    )
    similarities = [
        dice_similarity(mask, reference) for mask in normalized_masks
    ]
    closest_index = int(np.argmax(similarities))
    closest_path = files[closest_index]
    closest_segmentation = normalize_wing_segmentation(closest_path, grid)
    Image.fromarray(closest_segmentation[::-1]).save(
        reference_segmentation_path
    )
    with reference_source_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("filename", "split", "shape_dice_to_average"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": closest_path.name,
                "split": args.split,
                "shape_dice_to_average": similarities[closest_index],
            }
        )

    print(f"Saved mean occupancy: {probability_path}")
    print(f"Saved binary reference: {reference_path}")
    print(f"Saved overlay: {overlay_path}")
    print(
        "Closest training wing: "
        f"{closest_path.name} "
        f"(Dice={similarities[closest_index]:.6f})"
    )
    print(f"Saved reference segmentation: {reference_segmentation_path}")
    print(f"Saved reference source metadata: {reference_source_path}")


if __name__ == "__main__":
    main()
