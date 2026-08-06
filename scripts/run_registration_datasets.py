#!/usr/bin/env python3
"""Run average-reference ANTs registration on consolidated SliceLearn data.

The runner processes three groups:

* ``slides-artificial``: the paired ``test_50`` slide wings in nine regimes;
* ``live-bees-artificial``: every live-bee wing in nine regimes;
* ``slides-test-damaged``: genuinely damaged slide wings without ground truth.

It saves padded-frame binary inputs, full-mask predictions, ground truth where
available, optional overlays, transforms, and run configuration. Padding keeps
the transformed reference inside the fixed ANTs image instead of clipping it
to the native input dimensions. Evaluation metrics are deliberately not
calculated.
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
from dataclasses import dataclass
from pathlib import Path

import ants
import numpy as np
import scipy
import yaml
from PIL import Image
from tqdm import tqdm

from ants_average_reference_baseline import (
    DEFAULT_REFERENCE_LEFT_PADDING,
    START_ANGLES_DEGREES,
    START_SCALES,
    SELECTION_WEIGHTS,
    load_reference_mask,
    register_reference,
)
from slicelearn_dataset import (
    DATASET_ROOT,
    load_metadata,
    load_segmentation_mask,
    split_filenames,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "02_data_outputs"
    / "01_final"
    / "02_models"
    / "03_registration"
)
ALL_GROUPS = (
    "slides-artificial",
    "live-bees-artificial",
    "slides-test-damaged",
)
AFFINE_PARAMETERS = {
    "type_of_transform": "Similarity",
    "metric": "mattes",
    "random_sampling_rate": 1.0,
    "iterations": [2000, 1200, 600, 200],
    "shrink_factors": [8, 4, 2, 1],
    "smoothing_sigmas": [3, 2, 1, 0],
}
@dataclass(frozen=True)
class RegistrationJob:
    group: str
    input_path: Path
    relative_output: Path
    ground_truth_path: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.relative_output}/{self.input_path.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=ALL_GROUPS,
        default=list(ALL_GROUPS),
        help="Dataset groups to process (default: all three).",
    )
    parser.add_argument(
        "--damage-types",
        nargs="+",
        help="Artificial-damage regimes to process (default: every regime).",
    )
    parser.add_argument(
        "--reference-left-padding",
        type=float,
        default=DEFAULT_REFERENCE_LEFT_PADDING,
    )
    parser.add_argument(
        "--canvas-padding",
        type=float,
        default=0.25,
        help=(
            "Pad every side of the native image by this fraction before "
            "registration so the transformed reference is not clipped "
            "(default: 0.25)."
        ),
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save a color overlay alongside each prediction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit base specimens per group for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute registrations whose core output files already exist.",
    )
    args = parser.parse_args()
    if not 0 <= args.reference_left_padding < 0.5:
        parser.error("--reference-left-padding must be in [0, 0.5)")
    if not 0 <= args.canvas_padding < 1:
        parser.error("--canvas-padding must be in [0, 1)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def resolve_damage_types(
    roots: list[Path],
    requested: list[str] | None,
) -> list[str]:
    available_sets = [
        {path.name for path in root.iterdir() if path.is_dir()}
        for root in roots
    ]
    if not available_sets:
        return []
    common = set.intersection(*available_sets)
    selected = requested or sorted(common)
    missing = sorted(set(selected) - common)
    if missing:
        raise FileNotFoundError(
            f"Damage regimes unavailable in every selected dataset: {missing}"
        )
    return list(selected)


def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(f"Missing required images:\n{preview}{more}")


def build_jobs(args: argparse.Namespace) -> tuple[list[RegistrationJob], list[str]]:
    root = args.datasets_root
    slide_undamaged = root / "00_slides" / "00_undamaged"
    slide_artificial = root / "00_slides" / "01_artificial_damage"
    slide_test_damaged = root / "00_slides" / "02_test_damaged"
    live_undamaged = root / "01_live_bees" / "00_undamaged"
    live_artificial = root / "01_live_bees" / "01_artificial_damage"

    artificial_roots: list[Path] = []
    if "slides-artificial" in args.groups:
        artificial_roots.append(slide_artificial)
    if "live-bees-artificial" in args.groups:
        artificial_roots.append(live_artificial)
    damage_types = resolve_damage_types(artificial_roots, args.damage_types)
    jobs: list[RegistrationJob] = []

    if "slides-artificial" in args.groups:
        filenames = split_filenames("test_50", root / "dataset_splits.csv")
        if args.limit is not None:
            filenames = filenames[: args.limit]
        for damage_type in damage_types:
            for filename in filenames:
                jobs.append(
                    RegistrationJob(
                        group="slides-artificial",
                        input_path=slide_artificial / damage_type / filename,
                        ground_truth_path=slide_undamaged / filename,
                        relative_output=(
                            Path("00_slides")
                            / "01_artificial_damage"
                            / damage_type
                        ),
                    )
                )

    if "live-bees-artificial" in args.groups:
        filenames = sorted(path.name for path in live_undamaged.glob("*.png"))
        if args.limit is not None:
            filenames = filenames[: args.limit]
        for damage_type in damage_types:
            for filename in filenames:
                jobs.append(
                    RegistrationJob(
                        group="live-bees-artificial",
                        input_path=live_artificial / damage_type / filename,
                        ground_truth_path=live_undamaged / filename,
                        relative_output=(
                            Path("01_live_bees")
                            / "01_artificial_damage"
                            / damage_type
                        ),
                    )
                )

    if "slides-test-damaged" in args.groups:
        paths = sorted(slide_test_damaged.glob("*.png"))
        if args.limit is not None:
            paths = paths[: args.limit]
        for path in paths:
            jobs.append(
                RegistrationJob(
                    group="slides-test-damaged",
                    input_path=path,
                    relative_output=Path("00_slides") / "02_test_damaged",
                )
            )

    required = [job.input_path for job in jobs]
    required.extend(
        job.ground_truth_path
        for job in jobs
        if job.ground_truth_path is not None
    )
    require_files(required)
    if not jobs:
        raise ValueError("No registration jobs were selected")
    return jobs, damage_types


def output_paths(
    job: RegistrationJob,
    results_root: Path,
) -> dict[str, Path]:
    directory = results_root / job.relative_output
    stem = job.input_path.stem
    paths = {
        "input": directory / f"{stem}_1input.png",
        "prediction": directory / f"{stem}_2model.png",
        "overlay": directory / f"{stem}_4overlay.png",
        "registration_metadata": directory / f"{stem}_registration.yaml",
    }
    if job.ground_truth_path is not None:
        paths["ground_truth"] = directory / f"{stem}_3gt.png"
    return paths


def core_outputs_complete(
    paths: dict[str, Path],
    expected_metric: str,
    expected_canvas_padding: float,
) -> bool:
    required = [
        paths["input"],
        paths["prediction"],
        paths["registration_metadata"],
    ]
    if "ground_truth" in paths:
        required.append(paths["ground_truth"])
    if not all(path.is_file() for path in required):
        return False
    try:
        with paths["registration_metadata"].open(encoding="utf-8") as handle:
            registration_metadata = yaml.safe_load(handle) or {}
        if registration_metadata.get("aff_metric") != expected_metric:
            return False
        stored_fraction = registration_metadata.get("canvas_padding_fraction")
        if stored_fraction is None or not np.isclose(
            float(stored_fraction), expected_canvas_padding
        ):
            return False

        original_shape = tuple(
            int(value) for value in registration_metadata["original_shape"]
        )
        stored_padded_shape = tuple(
            int(value) for value in registration_metadata["padded_shape"]
        )
        if len(original_shape) != 2 or len(stored_padded_shape) != 2:
            return False
        expected_padding = padded_frame(
            *original_shape,
            expected_canvas_padding,
        )
        padding_metadata = registration_metadata["padding_pixels"]
        stored_padding = (
            (
                int(padding_metadata["top"]),
                int(padding_metadata["bottom"]),
            ),
            (
                int(padding_metadata["left"]),
                int(padding_metadata["right"]),
            ),
        )
        if stored_padding != expected_padding:
            return False
        expected_padded_shape = (
            original_shape[0] + sum(expected_padding[0]),
            original_shape[1] + sum(expected_padding[1]),
        )
        if stored_padded_shape != expected_padded_shape:
            return False

        saved_masks = [
            load_saved_binary(paths["input"]),
            load_saved_binary(paths["prediction"]),
        ]
        if "ground_truth" in paths:
            saved_masks.append(load_saved_binary(paths["ground_truth"]))
        if any(mask.shape != expected_padded_shape for mask in saved_masks):
            return False
        if touched_frame_edges(saved_masks[1]):
            return False
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
        return False
    return True


def padded_frame(
    height: int,
    width: int,
    fraction: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return symmetric vertical and horizontal padding for a native frame."""
    vertical = round(height * fraction)
    horizontal = round(width * fraction)
    return ((vertical, vertical), (horizontal, horizontal))


def pad_mask(
    mask: np.ndarray,
    padding: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    return np.pad(mask, padding, mode="constant")


def touched_frame_edges(mask: np.ndarray) -> tuple[str, ...]:
    """Return padded-frame edges reached by a registered prediction."""
    touched: list[str] = []
    if mask[0, :].any():
        touched.append("top")
    if mask[-1, :].any():
        touched.append("bottom")
    if mask[:, 0].any():
        touched.append("left")
    if mask[:, -1].any():
        touched.append("right")
    return tuple(touched)


def save_black_foreground(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)


def load_saved_binary(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) < 128).astype(np.uint8)


def save_overlay(
    target: np.ndarray,
    prediction: np.ndarray,
    path: Path,
) -> None:
    """Save target green, prediction magenta, and their overlap white."""
    target_bool = target.astype(bool)
    prediction_bool = prediction.astype(bool)
    image = np.zeros((*target.shape, 3), dtype=np.uint8)
    image[target_bool] = (0, 220, 0)
    image[prediction_bool] = (220, 0, 220)
    image[target_bool & prediction_bool] = (255, 255, 255)
    Image.fromarray(image).save(path)


def ensure_resume_overlay(paths: dict[str, Path]) -> None:
    if paths["overlay"].is_file():
        return
    prediction = load_saved_binary(paths["prediction"])
    target_path = paths.get("ground_truth", paths["input"])
    target = load_saved_binary(target_path)
    save_overlay(target, prediction, paths["overlay"])


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_heartbeat(
    stop_event: threading.Event,
    label: str,
    started: float,
    previous_durations: list[float],
    jobs_after_current: int,
    interval_seconds: float = 10,
) -> None:
    while not stop_event.wait(interval_seconds):
        elapsed = time.monotonic() - started
        if previous_durations:
            mean_duration = float(np.mean(previous_durations))
            eta = (
                max(mean_duration - elapsed, 0)
                + mean_duration * jobs_after_current
            )
            finish = dt.datetime.now().astimezone() + dt.timedelta(seconds=eta)
            estimate = (
                f"ETA {format_duration(eta)}, "
                f"finish about {finish:%Y-%m-%d %H:%M:%S}"
            )
        else:
            estimate = "ETA estimating from first registration"
        tqdm.write(
            f"Still running {label}: current wing {format_duration(elapsed)}; "
            f"{estimate}",
            file=sys.stderr,
        )


def make_config(
    args: argparse.Namespace,
    reference: Path,
    jobs: list[RegistrationJob],
    damage_types: list[str],
    seed_directory: Path,
    invocation_id: str,
) -> dict:
    metadata_path = args.datasets_root / "dataset_metadata.yaml"
    metadata = load_metadata(metadata_path)
    group_counts = {
        group: sum(job.group == group for job in jobs)
        for group in args.groups
    }
    return {
        "invocation": {
            "id": invocation_id,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            "overwrite": args.overwrite,
            "limit_per_group": args.limit,
        },
        "run": {
            "method": "ants_average_reference_registration",
            "seed": args.seed,
            "groups": args.groups,
            "group_registration_counts": group_counts,
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
            "reference_mask": str(reference),
        },
        "dataset": metadata,
        "reference": {
            "left_padding_fraction": args.reference_left_padding,
            "canvas_padding_fraction": args.canvas_padding,
            "placement": "uniform_scale_preserving_aspect_ratio",
        },
        "registration": {
            **AFFINE_PARAMETERS,
            "start_angles_degrees": list(START_ANGLES_DEGREES),
            "start_scales": list(START_SCALES),
            "include_identity_candidate": True,
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
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "not set"
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
    reference = (
        args.reference.resolve()
        if args.reference is not None
        else args.datasets_root / "02_average_reference" / "average_reference_wing.png"
    )
    for path in (
        args.datasets_root / "dataset_splits.csv",
        args.datasets_root / "dataset_metadata.yaml",
        reference,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

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
        reference,
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
    shutil.copy2(reference, model_directory / "average_reference_wing.png")

    completed = 0
    incompatible_existing = 0
    pending: list[RegistrationJob] = []
    for job in jobs:
        paths = output_paths(job, results_root)
        if (
            not args.overwrite
            and core_outputs_complete(
                paths,
                AFFINE_PARAMETERS["metric"],
                args.canvas_padding,
            )
        ):
            if args.save_visualizations:
                ensure_resume_overlay(paths)
            completed += 1
        else:
            core_path_keys = ["input", "prediction", "registration_metadata"]
            if "ground_truth" in paths:
                core_path_keys.append("ground_truth")
            if any(paths[key].exists() for key in core_path_keys):
                incompatible_existing += 1
            pending.append(job)

    tqdm.write(
        f"Requested {len(jobs)} registrations: {completed} already complete, "
        f"{len(pending)} remaining; {incompatible_existing} existing result "
        "sets will be replaced because they are unpadded, malformed, or use "
        "different settings. Metrics are disabled.",
        file=sys.stderr,
    )
    progress = tqdm(
        total=len(jobs),
        initial=completed,
        desc="ANTs registration",
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
            observed_original = load_segmentation_mask(
                job.input_path,
                args.datasets_root / "dataset_metadata.yaml",
            )
            reference_original = load_reference_mask(
                reference,
                observed_original.shape,
                left_padding_fraction=args.reference_left_padding,
            )
            padding = padded_frame(
                *observed_original.shape,
                args.canvas_padding,
            )
            observed = pad_mask(observed_original, padding)
            reference_in_frame = pad_mask(reference_original, padding)
            transform_directory = transforms_root / job.relative_output
            transform_directory.mkdir(parents=True, exist_ok=True)
            transform_prefix = (
                transform_directory / f"{job.input_path.stem}_"
            )
            prediction, _ = register_reference(
                observed,
                reference_in_frame,
                transform_prefix,
                seed=args.seed,
                aff_metric=AFFINE_PARAMETERS["metric"],
            )
            touched_edges = touched_frame_edges(prediction)
            if touched_edges:
                raise RuntimeError(
                    f"Registered prediction for {job.label} still touches "
                    f"the padded canvas ({', '.join(touched_edges)}). "
                    "Increase --canvas-padding and rerun this seed."
                )
        finally:
            heartbeat_stop.set()
            heartbeat.join()

        save_black_foreground(observed, paths["input"])
        save_black_foreground(prediction, paths["prediction"])
        if job.ground_truth_path is not None:
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
            save_black_foreground(ground_truth, paths["ground_truth"])
        if args.save_visualizations:
            target = (
                ground_truth
                if job.ground_truth_path is not None
                else observed
            )
            save_overlay(target, prediction, paths["overlay"])
        paths["registration_metadata"].write_text(
            yaml.safe_dump(
                {
                    "aff_metric": AFFINE_PARAMETERS["metric"],
                    "registration_image": "signed_distance",
                    "seed": args.seed,
                    "canvas_padding_fraction": args.canvas_padding,
                    "padding_pixels": {
                        "top": padding[0][0],
                        "bottom": padding[0][1],
                        "left": padding[1][0],
                        "right": padding[1][1],
                    },
                    "original_shape": list(observed_original.shape),
                    "padded_shape": list(observed.shape),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
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
