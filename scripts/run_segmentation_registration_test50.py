#!/usr/bin/env python3
"""Run textured-reference ANTs registration on test_50 and live bees.

The fixed registration image is the RGB wing segmentation converted to scalar
luminance.  The moving image is ``reference_wing_segmentation.png`` in the
same normalized frame as ``average_reference_wing.png``.  ANTs estimates a
similarity transform with masked Mattes, and that transform is applied both to
the textured reference and to the binary average-reference mask.

The runner processes:

* all artificial-damage regimes for the 50 slide wings in ``test_50``;
* all artificial-damage regimes for every live-bee wing.
* genuinely damaged slide wings in ``02_test_damaged`` without ground truth.

It writes seed-specific configuration, results, transforms, and model inputs.
Metrics are deliberately not calculated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import shutil
import sys
import threading
import time
from pathlib import Path

import ants
import numpy as np
import scipy
import yaml
from PIL import Image
from scipy.ndimage import binary_erosion
from tqdm import tqdm

from ants_average_reference_baseline import (
    DEFAULT_REFERENCE_LEFT_PADDING,
    SELECTION_WEIGHTS,
    START_ANGLES_DEGREES,
    load_reference_mask,
    load_reference_segmentation,
    register_reference,
)
from run_registration_datasets import (
    RegistrationJob,
    build_jobs,
    format_duration,
    progress_heartbeat,
)
from slicelearn_dataset import (
    DATASET_ROOT,
    load_metadata,
    load_segmentation_mask,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "02_data_outputs"
    / "01_final"
    / "02_models"
    / "04_textured_registration"
)
GROUPS = (
    "slides-artificial",
    "live-bees-artificial",
    "slides-test-damaged",
)
DEFAULT_START_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
AFFINE_PARAMETERS = {
    "type_of_transform": "Similarity",
    "metric": "mattes",
    "fixed_metric_mask": "observed_wing",
    "mask_all_stages": True,
    "random_sampling_rate": 1.0,
    "iterations": [2000, 1200, 600, 200],
    "shrink_factors": [8, 4, 2, 1],
    "smoothing_sigmas": [3, 2, 1, 0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--reference-mask", type=Path)
    parser.add_argument("--reference-segmentation", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=GROUPS,
        default=list(GROUPS),
    )
    parser.add_argument("--damage-types", nargs="+")
    parser.add_argument(
        "--reference-left-padding",
        type=float,
        default=DEFAULT_REFERENCE_LEFT_PADDING,
    )
    parser.add_argument("--canvas-padding", type=float, default=0.25)
    parser.add_argument(
        "--start-angles",
        type=float,
        nargs="+",
        default=START_ANGLES_DEGREES,
    )
    parser.add_argument(
        "--start-scales",
        type=float,
        nargs="+",
        default=DEFAULT_START_SCALES,
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save RGB input/reference and ground-truth/reference overlaps.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit base specimens per group for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute registrations with existing core outputs.",
    )
    args = parser.parse_args()
    if not 0 <= args.reference_left_padding < 0.5:
        parser.error("--reference-left-padding must be in [0, 0.5)")
    if not 0 <= args.canvas_padding < 1:
        parser.error("--canvas-padding must be in [0, 1)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if any(scale <= 0 for scale in args.start_scales):
        parser.error("--start-scales values must be positive")
    return args


def output_paths(job: RegistrationJob, results_root: Path) -> dict[str, Path]:
    directory = results_root / job.relative_output
    stem = job.input_path.stem
    paths = {
        "input_segmentation": directory / f"{stem}_1input_segmentation.png",
        "prediction_mask": directory / f"{stem}_2prediction_mask.png",
        "prediction_segmentation": (
            directory / f"{stem}_2warped_reference_segmentation.png"
        ),
        "input_overlap": directory / f"{stem}_4input_overlap.png",
    }
    if job.ground_truth_path is not None:
        paths["ground_truth_segmentation"] = (
            directory / f"{stem}_3ground_truth_segmentation.png"
        )
        paths["ground_truth_overlap"] = (
            directory / f"{stem}_5ground_truth_overlap.png"
        )
    return paths


def core_outputs_complete(
    paths: dict[str, Path],
    save_visualizations: bool,
) -> bool:
    required = [
        paths["input_segmentation"],
        paths["prediction_mask"],
        paths["prediction_segmentation"],
    ]
    if "ground_truth_segmentation" in paths:
        required.append(paths["ground_truth_segmentation"])
    if save_visualizations:
        required.append(paths["input_overlap"])
        if "ground_truth_overlap" in paths:
            required.append(paths["ground_truth_overlap"])
    return all(path.is_file() for path in required)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        ).astype(np.float32)
        / 255.0
    )


def padded_frame(
    height: int,
    width: int,
    fraction: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    vertical = round(height * fraction)
    horizontal = round(width * fraction)
    return ((vertical, vertical), (horizontal, horizontal))


def pad_mask(
    mask: np.ndarray,
    padding: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    return np.pad(mask, padding, mode="constant")


def pad_rgb(
    rgb: np.ndarray,
    padding: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    return np.pad(
        rgb,
        padding + ((0, 0),),
        mode="constant",
        constant_values=255,
    )


def warp_rgb(
    fixed_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    transforms: list[str],
) -> np.ndarray:
    fixed = ants.from_numpy(luminance(fixed_rgb))
    channels = []
    for channel in range(3):
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=ants.from_numpy(
                moving_rgb[..., channel].astype(np.float32)
            ),
            transformlist=transforms,
            interpolator="linear",
        )
        channels.append(warped.numpy())
    return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)


def segmentation_overlap(
    target_rgb: np.ndarray,
    prediction_rgb: np.ndarray,
    target_mask: np.ndarray,
    prediction_mask: np.ndarray,
) -> np.ndarray:
    target = target_mask.astype(bool)
    prediction = prediction_mask.astype(bool)
    image = np.full_like(target_rgb, 255)
    target_only = target & ~prediction
    prediction_only = prediction & ~target
    shared = target & prediction
    image[target_only] = target_rgb[target_only]
    image[prediction_only] = prediction_rgb[prediction_only]
    image[shared] = (
        0.5 * target_rgb[shared] + 0.5 * prediction_rgb[shared]
    ).astype(np.uint8)
    target_edge = target & ~binary_erosion(target)
    prediction_edge = prediction & ~binary_erosion(prediction)
    image[target_edge] = (0, 220, 0)
    image[prediction_edge] = (220, 0, 220)
    image[target_edge & prediction_edge] = (255, 255, 255)
    return image


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def save_prediction_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)


def make_config(
    args: argparse.Namespace,
    reference_mask: Path,
    reference_segmentation: Path,
    jobs: list[RegistrationJob],
    damage_types: list[str],
    seed_directory: Path,
    invocation_id: str,
) -> dict:
    metadata_path = args.datasets_root / "dataset_metadata.yaml"
    metadata = load_metadata(metadata_path)
    return {
        "invocation": {
            "id": invocation_id,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            "overwrite": args.overwrite,
            "limit_per_group": args.limit,
        },
        "run": {
            "method": "ants_textured_segmentation_registration",
            "seed": args.seed,
            "groups": args.groups,
            "group_registration_counts": {
                group: sum(job.group == group for job in jobs)
                for group in args.groups
            },
            "total_registrations": len(jobs),
            "damage_types": damage_types,
            "save_visualizations": args.save_visualizations,
            "metrics_calculated": False,
            "output_directory": str(seed_directory),
        },
        "inputs": {
            "datasets_root": str(args.datasets_root),
            "dataset_splits": str(args.datasets_root / "dataset_splits.csv"),
            "dataset_metadata": str(metadata_path),
            "reference_mask": str(reference_mask),
            "reference_segmentation": str(reference_segmentation),
        },
        "dataset": metadata,
        "reference": {
            "left_padding_fraction": args.reference_left_padding,
            "canvas_padding_fraction": args.canvas_padding,
            "placement": "uniform_scale_preserving_aspect_ratio",
        },
        "registration": {
            **AFFINE_PARAMETERS,
            "fixed_image": "input RGB segmentation luminance",
            "moving_image": "reference RGB segmentation luminance",
            "prediction": "binary average-reference mask warped by transform",
            "start_angles_degrees": list(args.start_angles),
            "start_scales": list(args.start_scales),
            "include_identity_candidate": False,
            "random_seed": args.seed,
            "selection_score_weights": SELECTION_WEIGHTS,
        },
        "software": {
            "python": platform.python_version(),
            "antspy": getattr(ants, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": Image.__version__,
            "itk_global_default_number_of_threads": os.environ.get(
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                "not set",
            ),
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    args.datasets_root = args.datasets_root.resolve()
    args.output_root = args.output_root.resolve()
    reference_mask = (
        args.reference_mask.resolve()
        if args.reference_mask is not None
        else (
            args.datasets_root
            / "02_average_reference"
            / "average_reference_wing.png"
        )
    )
    reference_segmentation = (
        args.reference_segmentation.resolve()
        if args.reference_segmentation is not None
        else (
            args.datasets_root
            / "02_average_reference"
            / "reference_wing_segmentation.png"
        )
    )
    required_inputs = (
        args.datasets_root / "dataset_splits.csv",
        args.datasets_root / "dataset_metadata.yaml",
        reference_mask,
        reference_segmentation,
    )
    missing = [path for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    jobs, damage_types = build_jobs(args)
    seed_directory = args.output_root / f"seed_{args.seed:03d}"
    results_root = seed_directory / "results"
    transforms_root = seed_directory / "transforms"
    model_directory = seed_directory / "model"
    configs_directory = seed_directory / "configs"
    for directory in (
        results_root,
        transforms_root,
        model_directory,
        configs_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    invocation_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = make_config(
        args,
        reference_mask,
        reference_segmentation,
        jobs,
        damage_types,
        seed_directory,
        invocation_id,
    )
    config_text = yaml.safe_dump(config, sort_keys=False)
    (seed_directory / "config.yaml").write_text(config_text, encoding="utf-8")
    invocation_config = configs_directory / f"run_{invocation_id}.yaml"
    suffix = 1
    while invocation_config.exists():
        invocation_config = (
            configs_directory / f"run_{invocation_id}_{suffix:02d}.yaml"
        )
        suffix += 1
    invocation_config.write_text(config_text, encoding="utf-8")
    shutil.copy2(reference_mask, model_directory / reference_mask.name)
    shutil.copy2(
        reference_segmentation,
        model_directory / reference_segmentation.name,
    )

    completed = 0
    pending: list[RegistrationJob] = []
    for job in jobs:
        paths = output_paths(job, results_root)
        if not args.overwrite and core_outputs_complete(
            paths,
            args.save_visualizations,
        ):
            completed += 1
        else:
            pending.append(job)

    tqdm.write(
        f"Requested {len(jobs)} registrations: {completed} already complete, "
        f"{len(pending)} remaining. Metrics are disabled.",
        file=sys.stderr,
    )
    progress = tqdm(
        total=len(jobs),
        initial=completed,
        desc="Textured segmentation registration",
        unit="wing",
        file=sys.stderr,
        disable=False,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    durations: list[float] = []
    run_started = time.monotonic()

    for pending_index, job in enumerate(pending):
        paths = output_paths(job, results_root)
        progress.set_postfix_str(f"running {job.label}", refresh=True)
        registration_started = time.monotonic()
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=progress_heartbeat,
            args=(
                heartbeat_stop,
                job.label,
                registration_started,
                durations,
                len(pending) - pending_index - 1,
            ),
            daemon=True,
        )
        heartbeat.start()

        try:
            input_rgb_original = np.asarray(
                Image.open(job.input_path).convert("RGB"),
                dtype=np.uint8,
            )
            observed_original = load_segmentation_mask(
                job.input_path,
                args.datasets_root / "dataset_metadata.yaml",
            )
            reference_mask_original = load_reference_mask(
                reference_mask,
                observed_original.shape,
                left_padding_fraction=args.reference_left_padding,
            )
            reference_rgb_original = load_reference_segmentation(
                reference_segmentation,
                reference_mask,
                observed_original.shape,
                left_padding_fraction=args.reference_left_padding,
            )
            padding = padded_frame(
                *observed_original.shape,
                args.canvas_padding,
            )
            observed = pad_mask(observed_original, padding)
            reference_in_frame = pad_mask(reference_mask_original, padding)
            input_rgb = pad_rgb(input_rgb_original, padding)
            reference_rgb = pad_rgb(reference_rgb_original, padding)

            transform_directory = transforms_root / job.relative_output
            transform_directory.mkdir(parents=True, exist_ok=True)
            transform_prefix = (
                transform_directory / f"{job.input_path.stem}_"
            )
            prediction, registration = register_reference(
                observed,
                reference_in_frame,
                transform_prefix,
                seed=args.seed,
                aff_metric="mattes",
                fixed_metric_mask=observed,
                include_identity_candidate=False,
                start_angles_degrees=tuple(args.start_angles),
                start_scales=tuple(args.start_scales),
                fixed_registration_image=luminance(input_rgb),
                moving_registration_image=luminance(reference_rgb),
            )
            prediction_rgb = warp_rgb(
                input_rgb,
                reference_rgb,
                registration["fwdtransforms"],
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join()

        save_rgb(input_rgb, paths["input_segmentation"])
        save_prediction_mask(prediction, paths["prediction_mask"])
        save_rgb(prediction_rgb, paths["prediction_segmentation"])
        if args.save_visualizations:
            save_rgb(
                segmentation_overlap(
                    input_rgb,
                    prediction_rgb,
                    observed,
                    prediction,
                ),
                paths["input_overlap"],
            )
        if job.ground_truth_path is not None:
            ground_truth_rgb_original = np.asarray(
                Image.open(job.ground_truth_path).convert("RGB"),
                dtype=np.uint8,
            )
            ground_truth_original = load_segmentation_mask(
                job.ground_truth_path,
                args.datasets_root / "dataset_metadata.yaml",
            )
            if ground_truth_original.shape != observed_original.shape:
                raise ValueError(
                    f"Ground-truth shape mismatch for {job.input_path.name}: "
                    f"{ground_truth_original.shape} != "
                    f"{observed_original.shape}"
                )
            ground_truth = pad_mask(ground_truth_original, padding)
            ground_truth_rgb = pad_rgb(ground_truth_rgb_original, padding)
            save_rgb(
                ground_truth_rgb,
                paths["ground_truth_segmentation"],
            )
            if args.save_visualizations:
                save_rgb(
                    segmentation_overlap(
                        ground_truth_rgb,
                        prediction_rgb,
                        ground_truth,
                        prediction,
                    ),
                    paths["ground_truth_overlap"],
                )

        duration = time.monotonic() - registration_started
        durations.append(duration)
        remaining = len(pending) - pending_index - 1
        mean_duration = float(np.mean(durations))
        eta = mean_duration * remaining
        finish = dt.datetime.now().astimezone() + dt.timedelta(seconds=eta)
        progress.update(1)
        progress.set_postfix_str(
            f"last {format_duration(duration)}, "
            f"mean {format_duration(mean_duration)}, "
            f"finish {finish:%Y-%m-%d %H:%M:%S}",
            refresh=True,
        )

    progress.close()
    print(
        f"Completed {len(jobs)} requested registrations in "
        f"{format_duration(time.monotonic() - run_started)}."
    )
    print(f"Run configuration: {seed_directory / 'config.yaml'}")
    print(f"Results: {results_root}")
    print(f"Transforms: {transforms_root}")


if __name__ == "__main__":
    main()
