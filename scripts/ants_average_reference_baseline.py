#!/usr/bin/env python3
"""Register the average reference wing to artificially damaged wing masks.

This is a fixed-shape baseline for SliceLearn reconstruction. SliceLearn
jointly fits a latent wing shape and a similarity transform to an observed
damaged wing. Here the shape is always the average reference wing and ANTsPy
fits a similarity transform to signed-distance fields. Registration uses full
deterministic sampling, a longer coarse-to-fine schedule, and multiple initial
rotations/scales. Candidates are selected without using undamaged ground truth,
with an asymmetric observed-coverage score and a normalized-pose prior.

The damaged mask is the fixed image (observed pose); the complete average wing
is the moving image. Consequently, ANTs' warped moving image is the baseline
prediction of the complete wing in the damaged observation's coordinate frame.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import tempfile
from pathlib import Path

import ants
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from slicelearn_dataset import (
    AVERAGE_REFERENCE_WING,
    SLIDES_ARTIFICIAL_DAMAGE,
    SLIDES_UNDAMAGED,
    load_segmentation_mask,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DAMAGE_ROOT = SLIDES_ARTIFICIAL_DAMAGE
DEFAULT_UNDAMAGED_DIR = SLIDES_UNDAMAGED
DEFAULT_REFERENCE = AVERAGE_REFERENCE_WING
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "ants_average_reference_baseline"
DEFAULT_DAMAGE_TYPES = ("angle_45_50", "chunks_25", "crop_right_50")
DEFAULT_REFERENCE_LEFT_PADDING = 0.05
START_ANGLES_DEGREES = (-10.0, -5.0, 0.0, 5.0, 10.0)
START_SCALES = (0.9, 1.0, 1.1)
SELECTION_WEIGHTS = {
    "directed_distance": 1.0,
    "one_minus_observed_coverage": 1.0,
    "area_penalty": 0.10,
    "centroid_penalty": 2.0,
    "angle_penalty": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--damage-root", type=Path, default=DEFAULT_DAMAGE_ROOT)
    parser.add_argument("--undamaged-dir", type=Path, default=DEFAULT_UNDAMAGED_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="ANTs random seed (default: 0).",
    )
    parser.add_argument(
        "--reference-left-padding",
        type=float,
        default=DEFAULT_REFERENCE_LEFT_PADDING,
        help=(
            "Fraction of the fixed-image width left blank before placing the "
            "root-cropped reference (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar="DAMAGE_TYPE/FILENAME.png",
        help="Specific sample to process; repeat for multiple samples.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the comparison figure without opening it.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.reference_left_padding < 0.5:
        parser.error("--reference-left-padding must be between 0 and 0.5")
    return args


def load_original_coordinate_mask(path: Path) -> np.ndarray:
    """Load a native segmentation using the shared dataset masking metadata."""
    return load_segmentation_mask(path)


def load_reference_mask(
    path: Path,
    shape: tuple[int, int],
    left_padding_fraction: float = DEFAULT_REFERENCE_LEFT_PADDING,
) -> np.ndarray:
    """Place the cropped reference after a left margin in the fixed frame."""
    reference = np.asarray(Image.open(path).convert("L")) < 128
    coordinates = np.argwhere(reference)
    if not len(coordinates):
        raise ValueError(f"Reference mask is empty: {path}")
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    cropped = Image.fromarray(reference[top:bottom, left:right].astype(np.uint8) * 255)
    height, width = shape
    left_padding = round(width * left_padding_fraction)
    available_width = width - left_padding
    crop_width, crop_height = cropped.size
    uniform_scale = min(
        available_width / crop_width,
        height / crop_height,
    )
    resized_width = max(1, round(crop_width * uniform_scale))
    resized_height = max(1, round(crop_height * uniform_scale))
    resized = cropped.resize(
        (resized_width, resized_height), resample=Image.Resampling.NEAREST
    )
    placed = np.zeros((height, width), dtype=np.uint8)
    top_padding = (height - resized_height) // 2
    placed[
        top_padding:top_padding + resized_height,
        left_padding:left_padding + resized_width,
    ] = (np.asarray(resized) >= 128).astype(np.uint8)
    return placed


def load_reference_segmentation(
    segmentation_path: Path,
    mask_path: Path,
    shape: tuple[int, int],
    left_padding_fraction: float = DEFAULT_REFERENCE_LEFT_PADDING,
) -> np.ndarray:
    """Place an RGB reference using exactly the binary reference geometry."""
    mask = np.asarray(Image.open(mask_path).convert("L")) < 128
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError(f"Reference mask is empty: {mask_path}")
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1

    segmentation = Image.open(segmentation_path).convert("RGB")
    if segmentation.size != (mask.shape[1], mask.shape[0]):
        raise ValueError(
            "Reference segmentation and mask dimensions differ: "
            f"{segmentation.size} vs {(mask.shape[1], mask.shape[0])}"
        )
    cropped = segmentation.crop((left, top, right, bottom))
    height, width = shape
    left_padding = round(width * left_padding_fraction)
    available_width = width - left_padding
    crop_width, crop_height = cropped.size
    uniform_scale = min(
        available_width / crop_width,
        height / crop_height,
    )
    resized_width = max(1, round(crop_width * uniform_scale))
    resized_height = max(1, round(crop_height * uniform_scale))
    resized = cropped.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    placed = np.full((height, width, 3), 255, dtype=np.uint8)
    top_padding = (height - resized_height) // 2
    placed[
        top_padding:top_padding + resized_height,
        left_padding:left_padding + resized_width,
    ] = np.asarray(resized)
    return placed


def coordinate_diagnostics(
    damaged: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Confirm that the damaged pixels are contained in same-frame ground truth."""
    if damaged.shape != ground_truth.shape:
        raise ValueError(
            f"Original-coordinate shape mismatch: {damaged.shape} vs "
            f"{ground_truth.shape}"
        )
    damaged_bool = damaged.astype(bool)
    truth_bool = ground_truth.astype(bool)
    containment = (
        np.count_nonzero(damaged_bool & truth_bool)
        / max(np.count_nonzero(damaged_bool), 1)
    )
    if containment < 0.99:
        raise RuntimeError(
            f"Damaged and undamaged masks are not in the same coordinates: "
            f"containment={containment:.4f}"
        )
    return {"coordinate_damaged_in_ground_truth": containment}


def choose_default_samples(root: Path) -> list[Path]:
    """Choose different specimens spanning three representative damage modes."""
    samples: list[Path] = []
    fractions = (0.0, 0.5, 1.0)
    for damage_type, fraction in zip(DEFAULT_DAMAGE_TYPES, fractions):
        candidates = sorted((root / damage_type).glob("*.png"))
        if not candidates:
            raise FileNotFoundError(f"No PNG masks found in {root / damage_type}")
        index = round(fraction * (len(candidates) - 1))
        samples.append(candidates[index])
    return samples


def resolve_samples(root: Path, requested: list[str]) -> list[Path]:
    if not requested:
        return choose_default_samples(root)
    paths = [(root / item).resolve() for item in requested]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def signed_distance(mask: np.ndarray, clip_distance: float = 32.0) -> np.ndarray:
    """Return a clipped SDF, normalized to approximately [-1, 1]."""
    foreground = mask.astype(bool)
    sdf = distance_transform_edt(~foreground) - distance_transform_edt(foreground)
    return np.clip(sdf, -clip_distance, clip_distance).astype(np.float32) / clip_distance


def make_initial_transform(
    shape: tuple[int, int],
    angle_degrees: float,
    scale: float,
    path: Path,
) -> None:
    """Write a centered rotation and uniform-scale initial transform."""
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = scale * np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    height, width = shape
    transform = ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=2,
        matrix=matrix,
        center=((height - 1) / 2, (width - 1) / 2),
        translation=(0.0, 0.0),
    )
    ants.write_transform(transform, str(path))


def candidate_score(
    observed: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Score a fit asymmetrically, without using undamaged ground truth."""
    observed_bool = observed.astype(bool)
    prediction_bool = prediction.astype(bool)
    reference_bool = reference.astype(bool)
    observed_area = max(np.count_nonzero(observed_bool), 1)
    prediction_area_raw = np.count_nonzero(prediction_bool)
    if prediction_area_raw == 0:
        return float("inf"), {
            "selection_score": float("inf"),
            "selection_directed_distance": float("inf"),
            "selection_coverage": 0.0,
            "selection_area_penalty": float("inf"),
            "selection_centroid_penalty": float("inf"),
            "selection_angle_penalty": float("inf"),
        }
    prediction_area = prediction_area_raw
    reference_area = max(np.count_nonzero(reference_bool), 1)

    distance_to_prediction = distance_transform_edt(~prediction_bool)
    directed_distance = (
        float(distance_to_prediction[observed_bool].mean())
        / math.hypot(*observed.shape)
    )
    coverage = np.count_nonzero(observed_bool & prediction_bool) / observed_area
    area_penalty = abs(math.log(prediction_area / reference_area))

    def centroid_and_axis(mask: np.ndarray) -> tuple[np.ndarray, float]:
        coordinates = np.argwhere(mask)
        centroid = coordinates.mean(axis=0)
        centered = coordinates - centroid
        covariance = centered.T @ centered / len(coordinates)
        _, eigenvectors = np.linalg.eigh(covariance)
        major_axis = eigenvectors[:, -1]
        angle = math.atan2(major_axis[0], major_axis[1])
        return centroid, angle

    prediction_centroid, prediction_angle = centroid_and_axis(prediction_bool)
    reference_centroid, reference_angle = centroid_and_axis(reference_bool)
    centroid_penalty = (
        np.linalg.norm(prediction_centroid - reference_centroid)
        / math.hypot(*observed.shape)
    )
    angle_difference = abs(prediction_angle - reference_angle) % math.pi
    angle_difference = min(angle_difference, math.pi - angle_difference)
    angle_penalty = angle_difference / math.pi

    # Extra prediction outside the observation is not penalized: that is the
    # material this reconstruction baseline is intended to recover. Pose
    # penalties resolve the otherwise unidentifiable direction of extrapolation
    # when only a small fragment remains.
    score = (
        SELECTION_WEIGHTS["directed_distance"] * directed_distance
        + SELECTION_WEIGHTS["one_minus_observed_coverage"] * (1.0 - coverage)
        + SELECTION_WEIGHTS["area_penalty"] * area_penalty
        + SELECTION_WEIGHTS["centroid_penalty"] * centroid_penalty
        + SELECTION_WEIGHTS["angle_penalty"] * angle_penalty
    )
    return score, {
        "selection_score": score,
        "selection_directed_distance": directed_distance,
        "selection_coverage": coverage,
        "selection_area_penalty": area_penalty,
        "selection_centroid_penalty": centroid_penalty,
        "selection_angle_penalty": angle_penalty,
    }


def register_reference(
    damaged: np.ndarray,
    reference: np.ndarray,
    transform_prefix: Path,
    seed: int = 0,
    aff_metric: str = "meansquares",
    fixed_metric_mask: np.ndarray | None = None,
    include_identity_candidate: bool = True,
    start_angles_degrees: tuple[float, ...] = START_ANGLES_DEGREES,
    start_scales: tuple[float, ...] = START_SCALES,
    registration_image_mode: str = "signed_distance",
    fixed_registration_image: np.ndarray | None = None,
    moving_registration_image: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Run deterministic, multi-start similarity registration.

    ``damaged`` and ``reference`` are always binary masks used for candidate
    scoring and for the returned reconstruction.  Registration can instead be
    driven by explicit scalar-valued images via ``fixed_registration_image``
    and ``moving_registration_image``.  This keeps the image used to estimate
    the transform separate from the mask that is ultimately warped.
    """
    for stale_path in transform_prefix.parent.glob(f"{transform_prefix.name}best*"):
        if stale_path.is_file():
            stale_path.unlink()

    explicit_images = (
        fixed_registration_image is not None
        or moving_registration_image is not None
    )
    if explicit_images:
        if fixed_registration_image is None or moving_registration_image is None:
            raise ValueError(
                "fixed_registration_image and moving_registration_image "
                "must be supplied together"
            )
        if fixed_registration_image.shape != damaged.shape:
            raise ValueError(
                "fixed_registration_image shape differs from damaged mask: "
                f"{fixed_registration_image.shape} vs {damaged.shape}"
            )
        if moving_registration_image.shape != reference.shape:
            raise ValueError(
                "moving_registration_image shape differs from reference mask: "
                f"{moving_registration_image.shape} vs {reference.shape}"
            )
        fixed_array = fixed_registration_image.astype(np.float32)
        moving_array = moving_registration_image.astype(np.float32)
        effective_registration_mode = "explicit_intensity"
    elif registration_image_mode == "signed_distance":
        fixed_array = signed_distance(damaged)
        moving_array = signed_distance(reference)
        effective_registration_mode = registration_image_mode
    elif registration_image_mode == "segmentation":
        fixed_array = damaged.astype(np.float32)
        moving_array = reference.astype(np.float32)
        effective_registration_mode = registration_image_mode
    else:
        raise ValueError(
            "registration_image_mode must be 'signed_distance' or "
            f"'segmentation', not {registration_image_mode!r}"
        )

    fixed = ants.from_numpy(fixed_array)
    moving = ants.from_numpy(moving_array)
    binary_reference_image = ants.from_numpy(reference.astype(np.float32))
    ants_fixed_mask = (
        ants.from_numpy(fixed_metric_mask.astype(np.uint8))
        if fixed_metric_mask is not None
        else None
    )
    best: dict | None = None
    candidate_summaries: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="ants_wing_starts_") as temp_name:
        temp_dir = Path(temp_name)
        if include_identity_candidate:
            identity_path = temp_dir / "identity_initial.mat"
            make_initial_transform(damaged.shape, 0.0, 1.0, identity_path)
            identity_score, identity_parts = candidate_score(
                damaged, reference, reference
            )
            identity_summary = {
                "candidate_kind": "reference_identity",
                "start_angle_degrees": 0.0,
                "start_scale": 1.0,
                **identity_parts,
            }
            candidate_summaries.append(identity_summary)
            best = {
                "score": identity_score,
                "prediction": reference.copy(),
                "registration": None,
                "initial_path": identity_path,
                "summary": identity_summary,
            }

        for angle_degrees in start_angles_degrees:
            for scale in start_scales:
                label = f"a{angle_degrees:+05.1f}_s{scale:.2f}".replace(".", "p")
                initial_path = temp_dir / f"{label}_initial.mat"
                make_initial_transform(
                    damaged.shape, angle_degrees, scale, initial_path
                )

                registration = ants.registration(
                    fixed=fixed,
                    moving=moving,
                    type_of_transform="Similarity",
                    initial_transform=[str(initial_path)],
                    outprefix=str(temp_dir / f"{label}_"),
                    mask=ants_fixed_mask,
                    mask_all_stages=ants_fixed_mask is not None,
                    aff_metric=aff_metric,
                    aff_random_sampling_rate=1.0,
                    aff_iterations=(2000, 1200, 600, 200),
                    aff_shrink_factors=(8, 4, 2, 1),
                    aff_smoothing_sigmas=(3, 2, 1, 0),
                    random_seed=seed,
                )
                warped_reference = ants.apply_transforms(
                    fixed=fixed,
                    moving=binary_reference_image,
                    transformlist=registration["fwdtransforms"],
                    interpolator="nearestNeighbor",
                )
                prediction = (warped_reference.numpy() >= 0.5).astype(np.uint8)
                score, score_parts = candidate_score(
                    damaged, prediction, reference
                )
                summary = {
                    "candidate_kind": "ants_optimized",
                    "start_angle_degrees": angle_degrees,
                    "start_scale": scale,
                    **score_parts,
                }
                candidate_summaries.append(summary)

                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "prediction": prediction,
                        "registration": registration,
                        "initial_path": initial_path,
                        "summary": summary,
                    }

        if best is None:
            raise RuntimeError("All registration starts failed")

        saved_transforms: list[str] = []
        if best["registration"] is not None:
            for index, source_name in enumerate(
                best["registration"]["fwdtransforms"]
            ):
                source = Path(source_name)
                destination = Path(
                    f"{transform_prefix}best_{index}{source.suffix}"
                )
                shutil.copy2(source, destination)
                saved_transforms.append(str(destination))
        best_initial_path = Path(f"{transform_prefix}best_initial.mat")
        shutil.copy2(best["initial_path"], best_initial_path)

    result = {
        "fwdtransforms": saved_transforms,
        "initial_transform": str(best_initial_path),
        "best_candidate": best["summary"],
        "candidate_summaries": candidate_summaries,
        "registration_image_mode": effective_registration_mode,
    }
    return best["prediction"], result


def save_white_foreground(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def overlap_metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    observed_bool = observed.astype(bool)
    prediction_bool = prediction.astype(bool)
    intersection = np.count_nonzero(observed_bool & prediction_bool)
    observed_area = max(np.count_nonzero(observed_bool), 1)
    prediction_area = max(np.count_nonzero(prediction_bool), 1)
    return {
        "observed_coverage": intersection / observed_area,
        "prediction_precision": intersection / prediction_area,
        "observed_area": float(np.count_nonzero(observed_bool)),
        "prediction_area": float(np.count_nonzero(prediction_bool)),
    }


def reconstruction_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    truth = ground_truth.astype(bool)
    predicted = prediction.astype(bool)
    intersection = np.count_nonzero(truth & predicted)
    union = np.count_nonzero(truth | predicted)
    truth_area = max(np.count_nonzero(truth), 1)
    prediction_area = max(np.count_nonzero(predicted), 1)
    return {
        "ground_truth_dice": 2 * intersection / (truth_area + prediction_area),
        "ground_truth_iou": intersection / max(union, 1),
        "ground_truth_recall": intersection / truth_area,
        "ground_truth_precision": intersection / prediction_area,
        "ground_truth_area": float(np.count_nonzero(truth)),
    }


def make_comparison_figure(
    results: list[dict],
    output_path: Path,
    show: bool,
) -> None:
    figure, axes = plt.subplots(len(results), 5, figsize=(18, 3.5 * len(results)))
    axes = np.atleast_2d(axes)

    for row, result in enumerate(results):
        observed = result["observed"]
        prediction = result["prediction"]
        ground_truth = result["ground_truth"]
        overlap = observed.astype(bool) & prediction.astype(bool)

        axes[row, 0].imshow(observed, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{result['damage_type']}\nObserved damage")

        axes[row, 1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(
            "Selected average reference\n"
            f"{result['candidate_kind']}, "
            f"start={result['start_angle_degrees']:.0f}°/"
            f"{result['start_scale']:.1f}"
        )

        axes[row, 2].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(
            "Undamaged GT — original coordinates\n"
            f"damaged containment="
            f"{result['coordinate_damaged_in_ground_truth']:.3f}"
        )

        observed_overlay = np.zeros((*observed.shape, 3), dtype=np.float32)
        observed_overlay[..., 1] = observed * 0.95
        observed_overlay[..., 0] = prediction * 0.95
        observed_overlay[..., 2] = prediction * 0.95
        observed_overlay[overlap] = 1.0
        axes[row, 3].imshow(observed_overlay)
        axes[row, 3].set_title(
            "Observed vs prediction\n"
            f"observed coverage={result['observed_coverage']:.3f}"
        )

        truth_overlap = ground_truth.astype(bool) & prediction.astype(bool)
        truth_overlay = np.zeros((*ground_truth.shape, 3), dtype=np.float32)
        truth_overlay[..., 1] = ground_truth * 0.95
        truth_overlay[..., 0] = prediction * 0.95
        truth_overlay[..., 2] = prediction * 0.95
        truth_overlay[truth_overlap] = 1.0
        axes[row, 4].imshow(truth_overlay)
        axes[row, 4].set_title(
            "Ground truth vs prediction\n"
            f"Dice={result['ground_truth_dice']:.3f}, "
            f"IoU={result['ground_truth_iou']:.3f}"
        )

        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle(
        "ANTsPy similarity registration — green: target, "
        "magenta: prediction, white: overlap",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    damage_root = args.damage_root.resolve()
    undamaged_dir = args.undamaged_dir.resolve()
    reference_path = args.reference.resolve()
    output_dir = args.output_dir.resolve()

    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    samples = resolve_samples(damage_root, args.sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    all_candidate_rows: list[dict] = []

    for index, path in enumerate(samples, start=1):
        damage_type = path.parent.name
        print(f"[{index}/{len(samples)}] {damage_type}/{path.name}")
        observed = load_original_coordinate_mask(path)
        reference = load_reference_mask(
            reference_path,
            observed.shape,
            left_padding_fraction=args.reference_left_padding,
        )
        undamaged_path = undamaged_dir / path.name
        if not undamaged_path.is_file():
            raise FileNotFoundError(
                f"Undamaged counterpart not found for {path.name}: {undamaged_path}"
            )
        ground_truth = load_original_coordinate_mask(undamaged_path)
        frame_diagnostics = coordinate_diagnostics(observed, ground_truth)
        prefix = output_dir / f"{damage_type}_{path.stem}_"
        prediction, registration = register_reference(
            observed, reference, prefix, seed=args.seed
        )
        all_candidate_rows.extend(
            {
                "damage_type": damage_type,
                "filename": path.name,
                **candidate,
            }
            for candidate in registration["candidate_summaries"]
        )
        metrics = overlap_metrics(observed, prediction)
        truth_metrics = reconstruction_metrics(ground_truth, prediction)

        prediction_path = output_dir / f"{damage_type}_{path.stem}_registered.png"
        ground_truth_path = (
            output_dir / f"{damage_type}_{path.stem}_undamaged_aligned.png"
        )
        save_white_foreground(prediction, prediction_path)
        save_white_foreground(ground_truth, ground_truth_path)
        result_rows.append(
            {
                "damage_type": damage_type,
                "filename": path.name,
                "reference_left_padding": args.reference_left_padding,
                "observed": observed,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "prediction_path": str(prediction_path),
                "ground_truth_path": str(ground_truth_path),
                "forward_transforms": ";".join(registration["fwdtransforms"]),
                "initial_transform": registration["initial_transform"],
                **registration["best_candidate"],
                **frame_diagnostics,
                **metrics,
                **truth_metrics,
            }
        )

    metrics_path = output_dir / "test_registration_metrics.csv"
    csv_fields = [
        "damage_type",
        "filename",
        "reference_left_padding",
        "observed_coverage",
        "prediction_precision",
        "observed_area",
        "prediction_area",
        "ground_truth_dice",
        "ground_truth_iou",
        "ground_truth_recall",
        "ground_truth_precision",
        "ground_truth_area",
        "coordinate_damaged_in_ground_truth",
        "candidate_kind",
        "start_angle_degrees",
        "start_scale",
        "selection_score",
        "selection_directed_distance",
        "selection_coverage",
        "selection_area_penalty",
        "selection_centroid_penalty",
        "selection_angle_penalty",
        "prediction_path",
        "ground_truth_path",
        "forward_transforms",
        "initial_transform",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in csv_fields} for row in result_rows
        )

    candidates_path = output_dir / "test_registration_candidates.csv"
    candidate_fields = [
        "damage_type",
        "filename",
        "candidate_kind",
        "start_angle_degrees",
        "start_scale",
        "selection_score",
        "selection_directed_distance",
        "selection_coverage",
        "selection_area_penalty",
        "selection_centroid_penalty",
        "selection_angle_penalty",
    ]
    with candidates_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(all_candidate_rows)

    figure_path = output_dir / "test_registration_comparison.png"
    make_comparison_figure(result_rows, figure_path, show=not args.no_show)
    print(f"Saved comparison: {figure_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved candidate scores: {candidates_path}")


if __name__ == "__main__":
    main()
