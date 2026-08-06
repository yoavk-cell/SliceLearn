#!/usr/bin/env python3
"""Calculate filtered pixel metrics for all paired model predictions.

Only result folders containing an input, prediction, and ground truth are
included. Consequently, artificial-damage slides and live bees are analyzed,
while naturally damaged test images (which have no ground truth) are skipped.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODELS_ROOT = (
    PROJECT_ROOT / "02_data_outputs" / "01_final" / "02_models"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "02_data_outputs"
    / "01_final"
    / "03_analysis"
    / "model_pixel_metrics.csv"
)
FIELDNAMES = [
    "wing",
    "model",
    "ground_truth_pixels",
    "prediction_pixels",
    "prediction_pixels_not_in_ground_truth",
    "ground_truth_pixels_not_in_prediction",
    "absolute_error_pixels",
    "estimated_damage_pixels",
    "estimated_percent_damage_pixels",
]


@dataclass(frozen=True)
class ModelLayout:
    folder: str
    label: str
    input_suffix: str
    prediction_suffix: str
    truth_suffix: str
    input_foreground: str
    prediction_foreground: str
    truth_foreground: str
    result_subdirectories: tuple[str, ...] = ("results",)


@dataclass(frozen=True)
class PairedResult:
    wing: str
    model: str
    input_path: Path
    prediction_path: Path
    truth_path: Path
    input_foreground: str
    prediction_foreground: str
    truth_foreground: str


MODEL_LAYOUTS = (
    ModelLayout(
        folder="00_slicelearn",
        label="slicelearn",
        input_suffix="_1input.png",
        prediction_suffix="_2model.png",
        truth_suffix="_3gt.png",
        input_foreground="dark",
        prediction_foreground="dark",
        truth_foreground="dark",
        # Current SliceLearn batch inference mirrors the evaluation dataset
        # beneath ``reconstruction``. Retain ``results`` as a fallback for
        # outputs created by the older script.
        result_subdirectories=("reconstruction", "results"),
    ),
    ModelLayout(
        folder="01_transcnnhae_mask",
        label="transcnnhae_mask",
        input_suffix="_1input.png",
        prediction_suffix="_2prediction.png",
        truth_suffix="_3ground_truth.png",
        input_foreground="light",
        prediction_foreground="light",
        truth_foreground="light",
    ),
    ModelLayout(
        folder="02_transcnnhae_seg",
        label="transcnnhae_seg",
        input_suffix="_1input.png",
        prediction_suffix="_2prediction.png",
        truth_suffix="_3ground_truth.png",
        input_foreground="dark",
        prediction_foreground="dark",
        truth_foreground="dark",
    ),
    ModelLayout(
        folder="03_registration",
        label="registration",
        input_suffix="_1input.png",
        prediction_suffix="_2model.png",
        truth_suffix="_3gt.png",
        input_foreground="dark",
        prediction_foreground="dark",
        truth_foreground="dark",
    ),
    ModelLayout(
        folder="04_textured_registration",
        label="textured_registration",
        input_suffix="_1input_segmentation.png",
        prediction_suffix="_2prediction_mask.png",
        truth_suffix="_3ground_truth_segmentation.png",
        input_foreground="dark",
        prediction_foreground="dark",
        truth_foreground="dark",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate filtered pixel metrics for paired model outputs."
    )
    parser.add_argument(
        "--models-root", type=Path, default=DEFAULT_MODELS_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=[layout.label for layout in MODEL_LAYOUTS],
        help="Limit analysis to selected model families.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Limit analysis to these seed numbers.",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=5,
        help=(
            "Remove difference contours with an OpenCV contour area below "
            "this value (default: 5 pixels)."
        ),
    )
    parser.add_argument(
        "--left-ignore-fraction",
        type=float,
        default=0.15,
        help=(
            "Ignore differences in this fraction of the ground-truth "
            "foreground width from its left edge (default: 0.15)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a discovered prediction lacks its input or ground truth.",
    )
    args = parser.parse_args()
    if args.min_component_size < 0:
        parser.error("--min-component-size cannot be negative")
    if not 0 <= args.left_ignore_fraction <= 1:
        parser.error("--left-ignore-fraction must be between 0 and 1")
    return args


def seed_number(seed_dir: Path) -> Optional[int]:
    suffix = seed_dir.name.removeprefix("seed_")
    return int(suffix) if suffix.isdigit() else None


def wing_identifier(
    prediction_path: Path,
    results_root: Path,
    prediction_suffix: str,
) -> str:
    relative = prediction_path.relative_to(results_root)
    filename = relative.name.removesuffix(prediction_suffix) + ".png"
    return str(relative.with_name(filename))


def discover_results(
    models_root: Path,
    selected_types: Optional[set[str]],
    selected_seeds: Optional[set[int]],
    strict: bool,
    include_test_damaged: bool = False,
) -> tuple[list[PairedResult], list[str]]:
    paired: list[PairedResult] = []
    warnings: list[str] = []
    for layout in MODEL_LAYOUTS:
        if selected_types is not None and layout.label not in selected_types:
            continue
        family_root = models_root / layout.folder
        if not family_root.is_dir():
            warnings.append(f"Model family not found: {family_root}")
            continue
        for seed_dir in sorted(family_root.glob("seed_*")):
            seed = seed_number(seed_dir)
            if seed is None:
                continue
            if selected_seeds is not None and seed not in selected_seeds:
                continue
            results_root = next(
                (
                    seed_dir / subdirectory
                    for subdirectory in layout.result_subdirectories
                    if (seed_dir / subdirectory).is_dir()
                ),
                None,
            )
            if results_root is None:
                attempted = ", ".join(
                    str(seed_dir / subdirectory)
                    for subdirectory in layout.result_subdirectories
                )
                warnings.append(f"No results directory; tried: {attempted}")
                continue
            model_name = f"{layout.label}_seed_{seed:03d}"
            predictions = sorted(
                results_root.rglob(f"*{layout.prediction_suffix}")
            )
            matched = 0
            for prediction_path in predictions:
                stem = prediction_path.name.removesuffix(
                    layout.prediction_suffix
                )
                input_path = prediction_path.with_name(
                    stem + layout.input_suffix
                )
                truth_path = prediction_path.with_name(
                    stem + layout.truth_suffix
                )
                wing = wing_identifier(
                    prediction_path,
                    results_root,
                    layout.prediction_suffix,
                )
                wing_parts = Path(wing).parts
                is_test_damaged = (
                    len(wing_parts) >= 2
                    and wing_parts[1] == "02_test_damaged"
                )
                # Naturally damaged outputs intentionally have no GT.
                if not truth_path.is_file():
                    if not (include_test_damaged and is_test_damaged):
                        if strict and input_path.is_file():
                            raise FileNotFoundError(
                                f"Ground truth missing for {prediction_path}"
                            )
                        continue
                if not input_path.is_file():
                    message = f"Input missing for {prediction_path}"
                    if strict:
                        raise FileNotFoundError(
                            message
                        )
                    warnings.append(message)
                    continue
                paired.append(
                    PairedResult(
                        wing=wing,
                        model=model_name,
                        input_path=input_path,
                        prediction_path=prediction_path,
                        truth_path=truth_path,
                        input_foreground=layout.input_foreground,
                        prediction_foreground=layout.prediction_foreground,
                        truth_foreground=layout.truth_foreground,
                    )
                )
                matched += 1
            if not predictions:
                warnings.append(f"No predictions found for {model_name}")
            elif not matched:
                warnings.append(
                    f"No paired predictions with ground truth for {model_name}"
                )
    return paired, warnings


def read_mask(path: Path, foreground: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if foreground == "light":
        return image > 127
    if foreground == "dark":
        # Match segment_wing() in the old evaluator: threshold the textured
        # segmentation, then fill enclosed pale regions so veins/outlines
        # describe the complete wing rather than only its dark pixels.  Pad
        # with a known background border before flood filling: predictions
        # can legitimately touch any edge of their saved canvas, including
        # (0, 0), so that original image pixel is not a safe flood-fill seed.
        foreground_mask = (image < 240).astype(np.uint8) * 255
        padded = cv2.copyMakeBorder(
            foreground_mask,
            1,
            1,
            1,
            1,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        flood = padded.copy()
        height, width = padded.shape
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(padded, holes)
        return filled[1:-1, 1:-1] > 0
    raise ValueError(f"Unknown foreground polarity: {foreground}")


def ground_truth_left_slice(
    ground_truth: np.ndarray,
    fraction: float,
    include_before_ground_truth: bool = False,
) -> slice:
    """Return columns ignored at the left side of the GT foreground."""
    _, xs = np.where(ground_truth)
    if not len(xs) or fraction <= 0:
        return slice(0, 0)
    left = int(xs.min())
    right = int(xs.max()) + 1
    width = right - left
    cutoff = min(right, left + int(np.ceil(width * fraction)))
    start = 0 if include_before_ground_truth else left
    return slice(start, cutoff)


def result_left_ignore_slice(
    result: PairedResult,
    ground_truth: np.ndarray,
    fraction: float,
) -> slice:
    """Return the dataset-aware left exclusion used by metric filtering.

    Live-bee masks can contain body/root material before the ground-truth wing
    begins. For those images, ignore every column through the usual GT-relative
    cutoff. Slide images retain the historical GT-bounding-box-only behavior.
    """
    parts = Path(result.wing).parts
    is_live_bee = bool(parts) and parts[0] == "01_live_bees"
    return ground_truth_left_slice(
        ground_truth,
        fraction,
        include_before_ground_truth=is_live_bee,
    )


def retain_large_contours(
    difference: np.ndarray,
    min_area: int,
    ignored_columns: slice,
) -> np.ndarray:
    """Match the old evaluator's external-contour area filtering."""
    binary = difference.astype(np.uint8) * 255
    binary[:, ignored_columns] = 0
    if min_area <= 0:
        return binary > 0
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    retained = np.zeros_like(binary)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(
                retained, [contour], -1, 255, thickness=cv2.FILLED
            )
    retained[:, ignored_columns] = 0
    return retained > 0


def exclude_columns(mask: np.ndarray, columns: slice) -> np.ndarray:
    """Return a copy with the shared metric-exclusion columns removed."""
    filtered = mask.copy()
    filtered[:, columns] = False
    return filtered


def calculate_row(
    result: PairedResult,
    min_component_size: int,
    left_ignore_fraction: float,
) -> dict[str, object]:
    input_mask = read_mask(result.input_path, result.input_foreground)
    prediction = read_mask(
        result.prediction_path, result.prediction_foreground
    )
    ground_truth = read_mask(result.truth_path, result.truth_foreground)
    if not (
        input_mask.shape == prediction.shape == ground_truth.shape
    ):
        raise ValueError(
            f"Shape mismatch for {result.model} {result.wing}: "
            f"input={input_mask.shape}, prediction={prediction.shape}, "
            f"ground_truth={ground_truth.shape}"
        )

    ignored_columns = result_left_ignore_slice(
        result, ground_truth, left_ignore_fraction
    )
    # Apply the dataset-aware exclusion to every mask before calculating any
    # pixel count. Previously it affected only the two mismatch masks, while
    # prediction totals and estimated damage still included live-bee root/body
    # pixels on the left. This shared preprocessing is model-independent.
    input_mask = exclude_columns(input_mask, ignored_columns)
    prediction = exclude_columns(prediction, ignored_columns)
    ground_truth = exclude_columns(ground_truth, ignored_columns)
    prediction_only = retain_large_contours(
        prediction & ~ground_truth,
        min_component_size,
        ignored_columns,
    )
    ground_truth_only = retain_large_contours(
        ground_truth & ~prediction,
        min_component_size,
        ignored_columns,
    )

    ground_truth_pixels = int(np.count_nonzero(ground_truth))
    prediction_pixels = int(np.count_nonzero(prediction))
    prediction_only_pixels = int(np.count_nonzero(prediction_only))
    ground_truth_only_pixels = int(np.count_nonzero(ground_truth_only))
    estimated_damage_pixels = (
        prediction_pixels - int(np.count_nonzero(input_mask))
    )
    estimated_percent = (
        estimated_damage_pixels / ground_truth_pixels * 100
        if ground_truth_pixels
        else 0.0
    )
    return {
        "wing": result.wing,
        "model": result.model,
        "ground_truth_pixels": ground_truth_pixels,
        "prediction_pixels": prediction_pixels,
        "prediction_pixels_not_in_ground_truth": prediction_only_pixels,
        "ground_truth_pixels_not_in_prediction": ground_truth_only_pixels,
        "absolute_error_pixels": (
            prediction_only_pixels + ground_truth_only_pixels
        ),
        "estimated_damage_pixels": estimated_damage_pixels,
        "estimated_percent_damage_pixels": f"{estimated_percent:.6f}",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_config(
    output_path: Path,
    args: argparse.Namespace,
    paired_count: int,
    row_count: int,
) -> None:
    config = {
        "models_root": str(args.models_root),
        "output_csv": str(output_path),
        "model_types": args.model_types or "all",
        "seeds": args.seeds or "all",
        "min_component_size": args.min_component_size,
        "component_filter": (
            "cv2.RETR_EXTERNAL contours retained when "
            "cv2.contourArea >= min_component_size"
        ),
        "left_ignore_fraction": args.left_ignore_fraction,
        "left_ignore_reference": (
            "ground-truth foreground bounding-box width; live-bee results "
            "also ignore all columns before the ground-truth left edge"
        ),
        "left_ignore_applied_to": (
            "input, prediction, ground truth, difference counts, and "
            "estimated-damage counts for every model family"
        ),
        "estimated_damage_pixels": (
            "prediction_pixels - artificial_input_pixels"
        ),
        "estimated_percent_damage_pixels": (
            "estimated_damage_pixels / ground_truth_pixels * 100"
        ),
        "paired_results_discovered": paired_count,
        "rows_written": row_count,
    }
    with output_path.with_suffix(".yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    args.models_root = args.models_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    selected_types = (
        set(args.model_types) if args.model_types is not None else None
    )
    selected_seeds = set(args.seeds) if args.seeds is not None else None

    paired, warnings = discover_results(
        args.models_root,
        selected_types,
        selected_seeds,
        args.strict,
    )
    if not paired:
        raise FileNotFoundError(
            f"No paired model predictions found beneath {args.models_root}"
        )
    print(f"Discovered {len(paired)} paired predictions.")
    for warning in warnings:
        print(f"Warning: {warning}")

    rows = [
        calculate_row(
            result,
            args.min_component_size,
            args.left_ignore_fraction,
        )
        for result in tqdm(
            paired,
            desc="Calculating pixel metrics",
            unit="prediction",
            dynamic_ncols=True,
        )
    ]
    rows.sort(key=lambda row: (str(row["wing"]), str(row["model"])))
    write_csv(args.output, rows)
    write_config(args.output, args, len(paired), len(rows))
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Configuration: {args.output.with_suffix('.yaml')}")


if __name__ == "__main__":
    main()
