#!/usr/bin/env python3
"""Copy SliceLearn image metadata and normalize live-bee physical scale.

Slide crops are already stored at the canonical prepared-slide scale. Live-bee
images are resized isotropically, using each image's Blue Line calibration, so
that all SliceLearn inputs have the same pixels-per-centimetre scale.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml
from PIL import Image
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "02_data_outputs"
    / "00_intermediate"
    / "03_slicelearn_datasets"
)
DEFAULT_SLIDE_METADATA_ROOT = (
    PROJECT_ROOT
    / "02_data_outputs"
    / "00_intermediate"
    / "00_slides"
    / "04_wing_metadata"
)
DEFAULT_LIVE_METADATA_ROOT = (
    PROJECT_ROOT
    / "02_data_outputs"
    / "00_intermediate"
    / "01_live_bees"
    / "metadata"
)
DEFAULT_SCALE_METADATA = PROJECT_ROOT / "00_raw_data" / "image_scale_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--slide-metadata-root",
        type=Path,
        default=DEFAULT_SLIDE_METADATA_ROOT,
    )
    parser.add_argument(
        "--live-metadata-root",
        type=Path,
        default=DEFAULT_LIVE_METADATA_ROOT,
    )
    parser.add_argument(
        "--scale-metadata",
        type=Path,
        default=DEFAULT_SCALE_METADATA,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f".{path.stem}_",
        suffix=".json.tmp",
    )
    try:
        json.dump(data, temporary, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        os.replace(temporary.name, path)
    except Exception:
        temporary.close()
        if os.path.exists(temporary.name):
            os.unlink(temporary.name)
        raise


def image_paths_by_stem(dataset_branch: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(dataset_branch.rglob("*.png")):
        if "metadata" in path.parts:
            continue
        grouped.setdefault(path.stem, []).append(path)
    return grouped


def prepared_slide_scale(scale_metadata_path: Path) -> tuple[float, dict]:
    scale_metadata = load_json(scale_metadata_path)
    slides = scale_metadata["datasets"]["slides"]
    pixels_per_cm = float(
        slides["current_prepared_images_pixels_per_centimeter"]
    )
    if pixels_per_cm <= 0:
        raise ValueError("Prepared-slide pixels-per-centimetre must be positive")
    return pixels_per_cm, slides


def find_slide_metadata_path(source_metadata_root: Path, stem: str) -> Path:
    direct_path = source_metadata_root / f"{stem}.json"
    if direct_path.is_file():
        return direct_path

    # Two legacy dataset files use "slidewv16" while the reviewed metadata
    # and source-slide record use "slideww16".
    legacy_alias = stem.replace("slidewv", "slideww")
    alias_path = source_metadata_root / f"{legacy_alias}.json"
    if legacy_alias != stem and alias_path.is_file():
        return alias_path
    return direct_path


def copy_slide_metadata(
    dataset_root: Path,
    source_metadata_root: Path,
    target_pixels_per_cm: float,
    slide_scale_metadata: dict,
) -> int:
    slide_root = dataset_root / "00_slides"
    destination_root = slide_root / "metadata"
    grouped_paths = image_paths_by_stem(slide_root)
    if not grouped_paths:
        raise FileNotFoundError(f"No slide PNGs found under {slide_root}")

    for stem in tqdm(
        sorted(grouped_paths),
        desc="Copying slide metadata",
        unit="wing",
        dynamic_ncols=True,
    ):
        source_path = find_slide_metadata_path(source_metadata_root, stem)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"No slide metadata for {stem}: expected {source_path}"
            )
        metadata = load_json(source_path)
        if source_path.stem != stem:
            metadata["slicelearn_metadata_mapping"] = {
                "dataset_image_stem": stem,
                "source_metadata_stem": source_path.stem,
                "reason": "legacy slidewv/slideww naming inconsistency",
            }
        metadata["spatial_calibration"] = {
            "method": "known_microscope_slide_length_prepared_crop",
            "status": "accepted",
            "pixels_per_cm": target_pixels_per_cm,
            "centimeters_per_pixel": 1 / target_pixels_per_cm,
            "reference_length_cm": float(
                slide_scale_metadata["reference_length_centimeters"]
            ),
            "reference_length_pixels": int(
                slide_scale_metadata["current_prepared_image_long_edge_pixels"]
            ),
            "resized_for_slicelearn": False,
        }
        write_json_atomic(destination_root / f"{stem}.json", metadata)
    return len(grouped_paths)


def resize_png_atomic(
    path: Path,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> str:
    with Image.open(path) as image:
        current_size = image.size
        if current_size == resized_size:
            return "already_resized"
        if current_size != original_size:
            raise ValueError(
                f"Unexpected size for {path}: {current_size}; expected "
                f"original {original_size} or normalized {resized_size}"
            )
        resized = image.resize(resized_size, Image.Resampling.LANCZOS)
        temporary = path.with_name(f".{path.stem}.resize.tmp.png")
        resized.save(temporary, format="PNG")
    temporary.replace(path)
    return "resized"


def normalize_live_bees(
    dataset_root: Path,
    source_metadata_root: Path,
    target_pixels_per_cm: float,
) -> tuple[int, int]:
    live_root = dataset_root / "01_live_bees"
    destination_root = live_root / "metadata"
    grouped_paths = image_paths_by_stem(live_root)
    if not grouped_paths:
        raise FileNotFoundError(f"No live-bee PNGs found under {live_root}")

    resized_files = 0
    for stem in tqdm(
        sorted(grouped_paths),
        desc="Scaling live-bee images",
        unit="wing",
        dynamic_ncols=True,
    ):
        source_metadata_path = source_metadata_root / f"{stem}.json"
        if not source_metadata_path.is_file():
            raise FileNotFoundError(
                f"No live-bee metadata for {stem}: "
                f"expected {source_metadata_path}"
            )
        source_metadata = load_json(source_metadata_path)
        source_calibration = source_metadata.get("spatial_calibration", {})
        source_pixels_per_cm = source_calibration.get("pixels_per_cm")
        if (
            source_calibration.get("status") != "accepted"
            or source_pixels_per_cm is None
            or float(source_pixels_per_cm) <= 0
        ):
            raise ValueError(
                f"{stem} does not have an accepted pixels/cm calibration"
            )
        source_pixels_per_cm = float(source_pixels_per_cm)
        scale_factor = target_pixels_per_cm / source_pixels_per_cm

        paths = grouped_paths[stem]
        sizes: set[tuple[int, int]] = set()
        for path in paths:
            with Image.open(path) as image:
                sizes.add(image.size)

        destination_path = destination_root / f"{stem}.json"
        prior_normalization = {}
        if destination_path.is_file():
            prior_normalization = load_json(destination_path).get(
                "physical_scale_normalization",
                {},
            )
        if prior_normalization:
            original_size = tuple(
                map(int, prior_normalization["original_size_pixels"])
            )
            resized_size = tuple(
                map(int, prior_normalization["resized_size_pixels"])
            )
            allowed_sizes = {original_size, resized_size}
            if not sizes.issubset(allowed_sizes):
                raise ValueError(
                    f"Inconsistent existing sizes for {stem}: {sorted(sizes)}; "
                    f"expected only {sorted(allowed_sizes)}"
                )
        else:
            if len(sizes) != 1:
                raise ValueError(
                    f"All variants of {stem} must initially have one size; "
                    f"found {sorted(sizes)}"
                )
            original_size = next(iter(sizes))
            resized_size = (
                max(1, round(original_size[0] * scale_factor)),
                max(1, round(original_size[1] * scale_factor)),
            )

        metadata = deepcopy(source_metadata)
        metadata["source_spatial_calibration"] = deepcopy(source_calibration)
        metadata["spatial_calibration"] = {
            "method": "isotropic_resampling_from_blue_line_calibration",
            "status": "accepted",
            "pixels_per_cm": target_pixels_per_cm,
            "centimeters_per_pixel": 1 / target_pixels_per_cm,
            "source_pixels_per_cm": source_pixels_per_cm,
            "scale_factor": scale_factor,
            "resized_for_slicelearn": True,
        }
        normalization = {
            "status": "in_progress",
            "method": "isotropic_resize",
            "resampling": "Pillow LANCZOS",
            "source_pixels_per_cm": source_pixels_per_cm,
            "target_pixels_per_cm": target_pixels_per_cm,
            "scale_factor": scale_factor,
            "original_size_pixels": list(original_size),
            "resized_size_pixels": list(resized_size),
            "number_of_dataset_variants": len(paths),
        }
        metadata["physical_scale_normalization"] = normalization
        write_json_atomic(destination_path, metadata)

        for path in paths:
            result = resize_png_atomic(path, original_size, resized_size)
            if result == "resized":
                resized_files += 1

        normalization["status"] = "completed"
        write_json_atomic(destination_path, metadata)
    return len(grouped_paths), resized_files


def update_dataset_metadata(
    dataset_root: Path,
    target_pixels_per_cm: float,
    slide_count: int,
    live_bee_count: int,
) -> None:
    path = dataset_root / "dataset_metadata.yaml"
    with path.open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    metadata["dataset"][
        "coordinate_system"
    ] = "original_orientation_isotropic_scale_normalized"
    metadata["dataset"]["resizing_applied"] = True
    metadata["physical_scale_normalization"] = {
        "target_dataset": "prepared slide images",
        "target_pixels_per_cm": target_pixels_per_cm,
        "slide_images_resized": False,
        "live_bee_images_resized": True,
        "live_bee_resize": "isotropic per-image scaling from Blue Line calibration",
        "resampling": "Pillow LANCZOS",
        "slide_metadata_records": slide_count,
        "live_bee_metadata_records": live_bee_count,
        "metadata_locations": {
            "slides": "00_slides/metadata",
            "live_bees": "01_live_bees/metadata",
        },
    }
    path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")


def normalize_dataset_physical_scale(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    slide_metadata_root: Path = DEFAULT_SLIDE_METADATA_ROOT,
    live_metadata_root: Path = DEFAULT_LIVE_METADATA_ROOT,
    scale_metadata_path: Path = DEFAULT_SCALE_METADATA,
) -> dict[str, int | float]:
    dataset_root = dataset_root.resolve()
    target_pixels_per_cm, slide_scale_metadata = prepared_slide_scale(
        scale_metadata_path.resolve()
    )
    slide_count = copy_slide_metadata(
        dataset_root,
        slide_metadata_root.resolve(),
        target_pixels_per_cm,
        slide_scale_metadata,
    )
    live_bee_count, resized_files = normalize_live_bees(
        dataset_root,
        live_metadata_root.resolve(),
        target_pixels_per_cm,
    )
    update_dataset_metadata(
        dataset_root,
        target_pixels_per_cm,
        slide_count,
        live_bee_count,
    )
    return {
        "target_pixels_per_cm": target_pixels_per_cm,
        "slide_metadata_records": slide_count,
        "live_bee_metadata_records": live_bee_count,
        "live_bee_files_resized": resized_files,
    }


def main() -> None:
    args = parse_args()
    summary = normalize_dataset_physical_scale(
        args.dataset_root,
        args.slide_metadata_root,
        args.live_metadata_root,
        args.scale_metadata,
    )
    print("Completed SliceLearn physical-scale normalization:")
    for name, value in summary.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
