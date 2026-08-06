# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import csv
import os
import math
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from PIL import Image
from skimage import measure
from skimage.io import imshow
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
#from google.colab import drive
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from slicelearn_dataset import (
    DATASET_ROOT,
    LIVE_BEES_ARTIFICIAL_DAMAGE,
    LIVE_BEES_UNDAMAGED,
    SLIDES_ARTIFICIAL_DAMAGE,
    SLIDES_UNDAMAGED,
    SLIDES_TEST_DAMAGED,
    load_segmentation_mask,
    split_filenames,
)
from slicelearn_training_controls import (
    clone_state_dict,
    make_torch_generator,
    normal_init_with_seed,
    seed_data_loader_worker,
    seed_everything,
)


def parse_cli_args():
    inference_groups = (
        "slides-test-damaged",
        "slides-artificial",
        "live-bees-artificial",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Train SliceLearn and run batch inference on the real-damaged, "
            "artificial-slide, and artificial-live-bee evaluation datasets."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument(
        "--test-split",
        default="test_50",
        help=(
            "Slide split used for artificial-damage inference "
            "(default: test_50, matching the other model evaluations)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--checkpoint-kind",
        choices=("best", "best-before-40", "final"),
        default="best",
        help=(
            "Checkpoint used for inference: the lowest combined validation "
            "loss (default), the best loss through epoch 40, or the final "
            "training epoch. All three are saved after training."
        ),
    )
    parser.add_argument("--inference-steps", type=int, default=1000)
    parser.add_argument(
        "--inference-only",
        "--skip-training",
        dest="inference_only",
        action="store_true",
        help=(
            "Skip preprocessing and training and load the existing checkpoint "
            "for the selected --seed."
        ),
    )
    parser.add_argument(
        "--inference-groups",
        nargs="+",
        choices=inference_groups,
        default=list(inference_groups),
        help="Evaluation datasets to reconstruct (default: all three).",
    )
    parser.add_argument(
        "--damage-types",
        nargs="+",
        default=None,
        help=(
            "Artificial-damage subfolders to process (default: every available "
            "damage type)."
        ),
    )
    parser.add_argument(
        "--inference-limit",
        type=int,
        default=None,
        help=(
            "Process at most this many images per dataset/damage type; useful "
            "for smoke tests."
        ),
    )
    parser.add_argument(
        "--overwrite-inference",
        action="store_true",
        help="Re-run inputs whose complete reconstruction outputs already exist.",
    )
    parser.add_argument(
        "--inference-plot-every",
        type=int,
        default=0,
        help=(
            "Save optimization diagnostics every N steps; 0 disables them "
            "during batch inference."
        ),
    )
    parser.set_defaults(inference_compile=True)
    parser.add_argument(
        "--no-inference-compile",
        dest="inference_compile",
        action="store_false",
        help=(
            "Disable the torch.compile decoder wrapper used by default during "
            "inference."
        ),
    )
    parser.add_argument(
        "--inference-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode for inference (default: default).",
    )
    parser.add_argument(
        "--inference-sampling",
        action="store_true",
        help=(
            "Use stratified coordinate sampling for the initialization sweep "
            "and early joint-optimization steps, followed by dense refinement. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--inference-sample-count",
        type=int,
        default=65536,
        help=(
            "Coordinates per stratified inference step when "
            "--inference-sampling is enabled (default: 65536)."
        ),
    )
    parser.add_argument(
        "--inference-sampling-resample-every",
        type=int,
        default=10,
        help="Reuse a stratified coordinate sample for this many steps.",
    )
    parser.add_argument(
        "--inference-sampling-boundary-band",
        type=float,
        default=0.03,
        help="SDF distance defining the exterior boundary stratum.",
    )
    parser.add_argument(
        "--inference-sampling-final-dense-steps",
        type=int,
        default=100,
        help=(
            "Full-grid refinement steps at the end of sampled inference "
            "(default: 100)."
        ),
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.inference_steps < 1:
        parser.error("--inference-steps must be at least 1")
    if args.inference_limit is not None and args.inference_limit < 1:
        parser.error("--inference-limit must be at least 1")
    if args.inference_plot_every < 0:
        parser.error("--inference-plot-every cannot be negative")
    if args.inference_sample_count < 1:
        parser.error("--inference-sample-count must be at least 1")
    if args.inference_sampling_resample_every < 1:
        parser.error("--inference-sampling-resample-every must be at least 1")
    if args.inference_sampling_boundary_band <= 0:
        parser.error("--inference-sampling-boundary-band must be positive")
    if args.inference_sampling_final_dense_steps < 1:
        parser.error(
            "--inference-sampling-final-dense-steps must be at least 1"
        )
    if (
        args.inference_sampling
        and args.inference_sampling_final_dense_steps >= args.inference_steps
    ):
        parser.error(
            "--inference-sampling-final-dense-steps must be below "
            "--inference-steps when --inference-sampling is enabled"
        )
    return args


CLI_ARGS = parse_cli_args()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def announce_phase(message):
    tqdm.write(f"\n=== {message} ===")

# =============================================================================
# ENVIRONMENT SETUP — Google Colab
# =============================================================================
# Only required if training. Mounting Drive is what gives the notebook access
# to the image dataset; skip this cell if you are loading pretrained weights.
# Running locally: skip the mount and point PROJECT_PATH at the repo root.
# =============================================================================

#drive.mount('/content/drive')
#PROJECT_PATH = "/content/drive/MyDrive/SliceLearn_Code"
#os.chdir(PROJECT_PATH)

# =============================================================================
# CONFIGURATION
# =============================================================================
S = 1051              # canvas resolution; fits every consolidated dataset image
SEED = CLI_ARGS.seed   # global random seed for reproducibility
TARGET_SIZE = 0.4     # normalized shape size after alignment
ROOT_CROP_X = -0.8    # x below which the wing root is cropped, in grid units

TRAIN_SPLIT = CLI_ARGS.train_split
VALIDATION_SPLIT = CLI_ARGS.validation_split
TEST_SPLIT = CLI_ARGS.test_split

DATA_ROOT = str(DATASET_ROOT)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = str(
    PROJECT_ROOT
    / "02_data_outputs"
    / "01_final"
    / "02_models"
    / "00_slicelearn"
    / f"seed_{SEED:03d}"
)

N_VAL_TO_USE = 20     # validation wings used downstream; lower for a faster run

seed_everything(SEED)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
diagnostics_dir = Path(OUTPUT_ROOT) / "diagnostics"
diagnostics_dir.mkdir(parents=True, exist_ok=True)
# =============================================================================

# =============================================================================
# DATASET SPLITS — select membership from the consolidated CSV
# =============================================================================
# Images remain in their source-specific dataset folders. The CSV is the sole
# source of split membership, including the paired test_50 subset.
# =============================================================================

split_columns = [
    "train",
    "train_50",
    "train_10",
    "val",
    "test",
    "test_50",
]
split_members = {name: split_filenames(name) for name in split_columns}
print(
    f"Dataset root: {DATASET_ROOT}\n"
    f"Output root: {OUTPUT_ROOT}\n"
    "Loaded dataset splits: "
    + ", ".join(f"{name}={len(files)}" for name, files in split_members.items())
)


# =============================================================================
# FUNCTIONS
# =============================================================================
# Array conventions used throughout:
#   grid   : (S, S, 3) — channels are (x, y, value); x, y in [-1, 1]
#   shape  : (N, 2)    — xy coordinates of the "on" pixels only
#   sdf    : (S, S, 3) — channels are (x, y, signed distance)
#
# S is the canvas resolution and is FIXED for a given pipeline. It is read from
# the CONFIGURATION cell above rather than passed as an argument, so that cell
# must be executed before any function here is called.
# =============================================================================

# Loads an image, binarizes it (non-white pixels = foreground), fills interior
# holes, and centers it in an SxS canvas with zero-padding.
def pad_wing(path):
    binary = load_segmentation_mask(path)
    h, w = binary.shape
    if h > S or w > S:
        raise ValueError(f"Image ({h},{w}) is larger than target size S={S}")
    pad_top    = (S - h) // 2
    pad_bottom = S - h - pad_top
    pad_left   = (S - w) // 2
    pad_right  = S - w - pad_left
    padded = np.pad(binary, ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant', constant_values=0)
    return padded


# Creates an SxS grid of (x, y) coordinates spanning [-1, 1] in both axes.
def make_square_grid():
    xs = np.linspace(-1, 1, S)
    ys = np.linspace(-1, 1, S)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X, Y], axis=-1)   # (S, S, 2)
    return coords


# Attaches a padded binary image to a coordinate grid as a third channel.
# The vertical flip reconciles array row order (top-down) with grid y (bottom-up).
def move_image_to_grid(set_grid, img):
    img_flipped = img[::-1, :]
    return np.concatenate([set_grid, img_flipped[..., None].astype(np.uint8)], axis=-1)


# Extracts the foreground pixels from a grid and returns their (x, y) coords.
def get_shape(grid):
    return grid[grid[:, :, 2] > 0][:, :2]


# Shape volume = number of foreground pixels.
def compute_volume(shape):
    return float(shape.shape[0])


# Centroid (mean position) of the foreground pixels.
def compute_centroid(shape):
    V = compute_volume(shape)
    sum_x = shape[:, 0].sum()
    sum_y = shape[:, 1].sum()
    center = np.array([sum_x / V, sum_y / V])
    return center


# 2x2 covariance matrix of the foreground pixel positions about the centroid.
def compute_covariance(shape):
    V = compute_volume(shape)
    beta = compute_centroid(shape)
    coords_shift = shape - beta
    cov_mat = (coords_shift.T @ coords_shift) / V
    return cov_mat


# Scalar size measure: sqrt of the covariance trace (total spread about centroid).
def compute_size(shape):
    return float(np.sqrt(np.trace(compute_covariance(shape))))


# Computes the similarity transform (scale lam, rotation R, translation t) that
# normalizes a shape to `target_size` and orients it along its principal axes.
# Eigenvectors from eigh come in ascending eigenvalue order, so columns are
# swapped to put the major axis first. `flip` fixes the sign ambiguity of the
# eigenvectors, i.e. which way the shape ends up facing; the default (-1, -1)
# is specific to this wing dataset and will need adjusting for other data.
def compute_alignment(shape, target_size=TARGET_SIZE, flip=(-1, -1)):
    beta = compute_centroid(shape)
    gamma = compute_size(shape)
    lam = target_size / gamma
    _, V = np.linalg.eigh(compute_covariance(shape))
    V = V[:, [1, 0]]
    V[:, 0] *= flip[0]
    V[:, 1] *= flip[1]
    R = V.T
    t = -lam * (R @ beta)
    return lam, R, t


# Applies an alignment transform by inverse warping: each output grid point is
# mapped back into source coordinates and filled by nearest-neighbor lookup.
def transform_on_set_grid(set_grid, img, lam, R, t):
    Ycoords = set_grid.reshape(-1, 2)
    Xquery = ((Ycoords - t) @ R) / lam
    src_coords = img[:, :, :2].reshape(-1, 2)
    src_vals   = img[:, :, 2].reshape(-1)
    tree = KDTree(src_coords)
    _, idx = tree.query(Xquery, k=1)
    idx = idx[:, 0]
    sampled_vals = src_vals[idx]
    out = np.concatenate([Ycoords, sampled_vals[:, None]], axis=1)
    return out.reshape(S, S, 3)


# Signed distance field of a binary shape: negative inside, positive outside,
# in grid units. `root` optionally zeroes everything left of that x value
# before the transform (removes the wing root).
def compute_sdf(grid, root=None):
    coords = grid[:, :, :2]
    vals = grid[:, :, 2].copy()
    if root is not None:
        xcoords = coords[:, :, 0]
        vals[xcoords < root] = 0
    mask = vals > 0
    d = 2.0 / (S - 1)                                    # grid spacing
    dist_out = distance_transform_edt(~mask, sampling=(d, d))
    dist_in  = distance_transform_edt(mask,  sampling=(d, d))
    sdf = dist_out - dist_in
    sdf_bin = np.concatenate([coords, sdf[..., None]], axis=-1)
    return sdf_bin


# Draws training points from an SDF field, oversampling the surface: `percent`
# of samples come from the narrow band |sdf| < epsilon, the rest uniformly from
# the whole grid. Returns (num_samples, 3) rows of (x, y, sdf).
# If the band is empty (degenerate mask), all samples are drawn uniformly.
def sample_sdf_points(
    sdf,
    epsilon=0.05,
    num_samples=5000,
    percent=0.5,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(SEED)
    coords_vals = sdf.reshape(-1, 3)
    sdf_vals = coords_vals[:, 2]
    band_mask = np.abs(sdf_vals) < epsilon
    band_points = coords_vals[band_mask]
    all_points = coords_vals
    n_band = int(num_samples * percent)
    if len(band_points) == 0:                # no surface found -> uniform only
        n_band = 0
    n_uniform = num_samples - n_band

    samples_list = []
    if n_band > 0:
        replace_band = len(band_points) < n_band
        band_idx = rng.choice(
            len(band_points),
            size=n_band,
            replace=replace_band,
        )
        samples_list.append(band_points[band_idx])
    if n_uniform > 0:
        replace_uniform = len(all_points) < n_uniform
        uniform_idx = rng.choice(
            len(all_points),
            size=n_uniform,
            replace=replace_uniform,
        )
        samples_list.append(all_points[uniform_idx])

    samples = np.concatenate(samples_list, axis=0)
    rng.shuffle(samples)
    return samples

# Preprocesses every PNG in a directory: pad to the SxS canvas, attach to the
# coordinate grid, then align (centre, scale to target_size, orient along the
# principal axes). Intermediates are discarded; only the aligned grids are
# returned, alongside the filenames in sorted order.
def preprocess_split(
    directory,
    set_grid,
    target_size=TARGET_SIZE,
    files=None,
    description="Normalizing wings",
):
    if files is None:
        files = sorted(
            f for f in os.listdir(directory) if f.lower().endswith(".png")
        )
    else:
        files = sorted(files)
    normalized_all = []
    for file in tqdm(
        files,
        desc=description,
        unit="wing",
        dynamic_ncols=True,
    ):
        path = os.path.join(directory, file)
        binary = pad_wing(path)
        coord = move_image_to_grid(set_grid, binary)
        shape = get_shape(coord)
        lam, R, t = compute_alignment(shape, target_size=target_size)
        normalized_all.append(transform_on_set_grid(set_grid, coord, lam, R, t))
    return files, normalized_all

# Pairwise distance matrix between SDFs, returning an (n, n) symmetric array.
# Distance is the root-mean-square difference between the two fields, which is
# the L2 function norm over the [-1,1]^2 domain up to a constant factor, and is
# independent of grid resolution S.
def compute_l2_sdf_matrix(
    sdf_list,
    description="Training SDF distance matrix",
):
    n = len(sdf_list)
    D = np.zeros((n, n))
    for i in tqdm(
        range(n),
        desc=description,
        unit="row",
        dynamic_ncols=True,
    ):
        Zi = sdf_list[i][:, :, 2]
        for j in range(i + 1, n):          # upper triangle only; matrix is symmetric
            Zj = sdf_list[j][:, :, 2]
            d = np.sqrt(np.mean((Zi - Zj) ** 2))
            D[i, j] = d
            D[j, i] = d
    return D


# Cross distance matrix between two sets of SDFs, returning an
# (len(sdf_list_a), len(sdf_list_b)) array. Row i holds the distances from
# element i of set A to every element of set B. Same metric as
# compute_l2_sdf_matrix.
def compute_l2_sdf_cross_matrix(
    sdf_list_a,
    sdf_list_b,
    description="Validation-to-training SDF distances",
):
    D = np.zeros((len(sdf_list_a), len(sdf_list_b)))
    for i, sdf_a in enumerate(
        tqdm(
            sdf_list_a,
            desc=description,
            unit="row",
            dynamic_ncols=True,
        )
    ):
        Za = sdf_a[:, :, 2]
        for j, sdf_b in enumerate(sdf_list_b):
            Zb = sdf_b[:, :, 2]
            D[i, j] = np.sqrt(np.mean((Za - Zb) ** 2))
    return D

# =============================================================================
# PREPROCESSING — aligned wings and their SDFs
# =============================================================================
# Phase 1: every PNG is padded to the SxS canvas, attached to the shared
#   coordinate grid, and aligned — centred, scaled to TARGET_SIZE, and oriented
#   along its principal axes. Alignment removes differences in position, scale
#   and rotation, so downstream comparisons reflect wing form rather than how
#   the specimen happened to be photographed.
#
# Phase 2: each aligned wing is converted to a signed distance field (negative
#   inside the shape, positive outside). The wing root is cropped at
#   x < ROOT_CROP_X before the transform, so the SDF describes the wing blade
#   only. The shape is recoverable at any time as sdf <= 0.
#
# Only aligned grids and their SDFs are kept; raw binaries and unaligned grids
# are intermediates and are not retained.
# Requires the split folders produced by the DATASET SPLIT cell.
# =============================================================================

if not CLI_ARGS.inference_only:
    announce_phase(f"Creating the {S} x {S} normalization grid")
    set_grid = make_square_grid()
    
    undamaged_dir = str(SLIDES_UNDAMAGED)
    
    # --- Phase 1: load and align ---
    announce_phase(
        f"Loading and normalizing {len(split_members[TRAIN_SPLIT])} "
        f"{TRAIN_SPLIT} wings"
    )
    train_files, train_normalized = preprocess_split(
        undamaged_dir,
        set_grid,
        files=split_members[TRAIN_SPLIT],
        description=f"Normalizing {TRAIN_SPLIT} wings",
    )
    announce_phase(
        f"Loading and normalizing {len(split_members[VALIDATION_SPLIT])} "
        f"{VALIDATION_SPLIT} wings"
    )
    val_files, val_normalized = preprocess_split(
        undamaged_dir,
        set_grid,
        files=split_members[VALIDATION_SPLIT],
        description=f"Normalizing {VALIDATION_SPLIT} wings",
    )
    
    # --- Phase 2: signed distance fields (root-cropped) ---
    announce_phase("Computing root-cropped training signed-distance fields")
    train_sdfs = [
        compute_sdf(grid, root=ROOT_CROP_X)
        for grid in tqdm(
            train_normalized,
            desc="Training SDFs",
            unit="wing",
            dynamic_ncols=True,
        )
    ]
    announce_phase("Computing root-cropped validation signed-distance fields")
    val_sdfs = [
        compute_sdf(grid, root=ROOT_CROP_X)
        for grid in tqdm(
            val_normalized,
            desc="Validation SDFs",
            unit="wing",
            dynamic_ncols=True,
        )
    ]
    
    print(f"train: {len(train_files)} wings | val: {len(val_files)} wings")
    
    # =============================================================================
    # SANITY CHECK — alignment
    # =============================================================================
    # Overlays the outline (SDF zero-contour) of every wing in a split. If
    # alignment succeeded the outlines form a tight bundle centred on the origin
    # with a common orientation; an outlier curve indicates a wing that was
    # mis-oriented, badly segmented, or flipped by the eigenvector sign ambiguity
    # (see `flip` in compute_alignment). Diagnostic only — nothing downstream
    # depends on this cell, so it can be skipped.
    # =============================================================================
    
    announce_phase("Building alignment diagnostic plots")
    diagnostics_dir = Path(OUTPUT_ROOT) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for sdfs, color, split in [(train_sdfs, 'red', 'train'), (val_sdfs, 'blue', 'val')]:
        fig, ax = plt.subplots(figsize=(8, 8))
        for sdf in tqdm(
            sdfs,
            desc=f"Plotting {split} outlines",
            unit="wing",
            leave=False,
            dynamic_ncols=True,
        ):
            ax.contour(sdf[:, :, 0], sdf[:, :, 1], sdf[:, :, 2],
                       levels=[0], colors=color, linewidths=1, alpha=0.5)
    
        ax.axhline(0, color='black', linewidth=0.8)   # coordinate axes for reference
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Aligned wing outlines — {split}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        figure_path = diagnostics_dir / f"normalized_outlines_{split}.png"
        fig.tight_layout()
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)
        tqdm.write(f"Saved normalized {split} outlines: {figure_path}")
    
    # =============================================================================
    # DISTANCE MATRICES — shape dissimilarity between wings
    # =============================================================================
    # D_train : (n_train, n_train) distances among the training wings.
    # D_val   : (n_val, n_train)   distances from each validation wing to every
    #                              training wing — row i is validation wing i.
    #
    # Distances are computed on the root-cropped, aligned SDFs, so they measure
    # blade shape difference only. Both matrices are saved to OUTPUT_ROOT so this
    # cell (the most expensive in the pipeline) need only be run once.
    #
    # The validation set is then truncated to the first N_VAL_TO_USE wings, which
    # allows a faster partial run without recomputing anything; set
    # N_VAL_TO_USE to the full validation size to use all of it.
    # =============================================================================
    
    announce_phase(
        f"Computing {len(train_sdfs)} x {len(train_sdfs)} training distances"
    )
    D_train = compute_l2_sdf_matrix(train_sdfs)
    announce_phase(
        f"Computing {len(val_sdfs)} x {len(train_sdfs)} "
        "validation-to-training distances"
    )
    D_val = compute_l2_sdf_cross_matrix(val_sdfs, train_sdfs)
    
    announce_phase(f"Saving distance matrices to {OUTPUT_ROOT}")
    np.save(os.path.join(OUTPUT_ROOT, "D_train.npy"), D_train)
    np.save(os.path.join(OUTPUT_ROOT, "D_val.npy"), D_val)
    
    # --- Validation subset used downstream ---
    val_files_sub      = val_files[:N_VAL_TO_USE]
    val_normalized_sub = val_normalized[:N_VAL_TO_USE]
    val_sdfs_sub       = val_sdfs[:N_VAL_TO_USE]
    D_val_sub          = D_val[:N_VAL_TO_USE, :]
    
    print(f"D_train: {D_train.shape} | D_val: {D_val.shape} "
          f"| using {len(val_sdfs_sub)} validation wings")

# =============================================================================
# MODEL COMPONENTS — dataset, SIREN layers, DeepSDF+SIREN network
# =============================================================================
# Everything in this cell belongs to the learning phase and is independent of
# the geometry utilities above. Reviewers interested only in the model need
# this cell, the hyperparameters cell, and the training loop.
# =============================================================================

# Flat tensor dataset of SDF samples. Each item is one sampled point:
#   (shape_id, coord, distance)
# where shape_id indexes the wing the point came from, coord is its (x, y)
# position, and distance is the ground-truth SDF value there. Stored as three
# contiguous tensors rather than per-point objects, so batching is a slice.
class SDFDataset(Dataset):
    def __init__(self, shape_ids, coords, distances):
        self.shape_ids = torch.as_tensor(shape_ids, dtype=torch.long)
        self.coords    = torch.as_tensor(coords,    dtype=torch.float32)
        self.distances = torch.as_tensor(distances, dtype=torch.float32)

    def __len__(self):
        return self.shape_ids.shape[0]

    def __getitem__(self, idx):
        return self.shape_ids[idx], self.coords[idx], self.distances[idx]


# Draws training points from a list of SDF fields and stacks them into three
# flat arrays. Points are sampled per wing (see sample_sdf_points), then
# concatenated with a shape_id recording which wing each point came from.
def build_sdf_samples(
    sdf_list,
    epsilon,
    num_samples,
    percent,
    rng,
    description="Sampling SDF points",
):
    shape_ids, coords, distances = [], [], []
    for shape_idx, sdf in enumerate(
        tqdm(
            sdf_list,
            desc=description,
            unit="wing",
            dynamic_ncols=True,
        )
    ):
        samples = sample_sdf_points(
            sdf,
            epsilon=epsilon,
            num_samples=num_samples,
            percent=percent,
            rng=rng,
        )
        shape_ids.append(np.full(len(samples), shape_idx, dtype=np.int64))
        coords.append(samples[:, :2])
        distances.append(samples[:, 2])
    return (np.concatenate(shape_ids),
            np.concatenate(coords).astype(np.float32),
            np.concatenate(distances).astype(np.float32))


# A single SIREN layer: linear transform followed by a scaled sine activation.
# Weight initialization depends on the layer's position — the first layer uses
# U(-1/n, 1/n), later layers U(-sqrt(6/n)/omega0, +...) — which is what keeps
# activations well-scaled through the depth of a sine network (Sitzmann et al.,
# 2020). omega0 controls the frequency the layer can represent.
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega0 = omega0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


# DeepSDF-based network with SIREN activations: maps a latent code plus a 2D coordinate to the
# predicted SDF value at that coordinate. Five sine layers followed by a plain
# linear head (no sine on the output, since SDF values are unbounded).
# NOTE: the constructor defaults below are the values recommended in the SIREN
# paper. The model actually trained here is instantiated with smaller values
# (see the hyperparameters cell) — those are the ones reported.
class SirenDeepSDF2D(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256,
                 omega0_first=30.0, omega0_hidden=60.0):
        super().__init__()
        input_dim = latent_dim + 2
        self.l1 = SineLayer(input_dim, hidden_dim, is_first=True, omega0=omega0_first)
        self.l2 = SineLayer(hidden_dim, hidden_dim, omega0=omega0_hidden)
        self.l3 = SineLayer(hidden_dim, hidden_dim, omega0=omega0_hidden)
        self.l4 = SineLayer(hidden_dim, hidden_dim, omega0=omega0_hidden)
        self.l5 = SineLayer(hidden_dim, hidden_dim, omega0=omega0_hidden)
        self.fc_out = nn.Linear(hidden_dim, 1)

        # Output head is linear, so it gets a mild uniform init rather than the
        # SIREN scheme used by the sine layers.
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_dim) * 0.5
            self.fc_out.weight.uniform_(-bound, bound)

    def forward(self, z, coords):
        x = torch.cat([z, coords], dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return self.fc_out(x)

# Normalized stress between latent-space distances and target shape distances:
#   sum((||z_i - z_j|| - D_ij)^2) / sum(D_ij^2)
# This is the quantity minimized (Eq. 2). Kruskal stress-1 is its square root
# and is what gets reported — see the training loop.
# `mask` selects which pairs count (used to drop self-pairs).
def compute_stress(z_a, z_b, D_target, den, mask=None):
    lat_dist = torch.norm(z_a.unsqueeze(1) - z_b.unsqueeze(0), dim=2)
    diff = (lat_dist - D_target).pow(2)
    num = diff[mask].sum() if mask is not None else diff.sum()
    return num / den

# Mean per-wing SDF reconstruction error: the MSE between predicted and true
# SDF values is computed separately for each wing, then averaged over wings.
# Averaging (rather than summing) keeps the value independent of how many wings
# are in the split, so train and validation curves are directly comparable.
def evaluate_sdf_loss(model, codes, ids, coords, dists, num_shapes, device):
    total = 0.0
    for shape_idx in range(num_shapes):
        sel = (ids == shape_idx)
        c = coords[sel]
        d = dists[sel]
        z = codes(torch.tensor([shape_idx], device=device)).expand(len(c), -1)
        pred = model(z, c).squeeze(1)
        total += (pred - d).pow(2).mean().item()
    return total / num_shapes

# Retrieves the latent code for one shape index, shaped (1, latent_dim).
def get_z(latent_codes, shape_idx):
    idx = torch.tensor([shape_idx], dtype=torch.long).to(device)
    return latent_codes(idx)


# Decodes a latent code into a full predicted SDF field by querying the network
# at every point of the coordinate grid. Returns an (S, S, 3) array in the same
# (x, y, value) layout as compute_sdf, so predicted and ground-truth fields can
# be compared or plotted with identical code.
def predicted_sdf(model, z, set_grid):
    model.eval()
    coords_np = set_grid.reshape(-1, 2)
    coords = torch.tensor(coords_np, dtype=torch.float32).to(device)

    if z.dim() == 1:
        z = z.unsqueeze(0)
    z = z.to(device)
    z_rep = z.repeat(coords.shape[0], 1)      # same code for every query point

    with torch.no_grad():
        pred_vals = model(z_rep, coords)

    pred_vals = pred_vals.squeeze().cpu().numpy()
    pred_sdf_flat = np.concatenate([coords_np, pred_vals[:, None]], axis=1)
    return pred_sdf_flat.reshape(S, S, 3)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Every value controlling the learning phase. These are the settings used for
# the results reported in the paper; change them here rather than inline.
# =============================================================================

# --- SDF point sampling ---
SAMPLE_EPSILON  = 0.07    # half-width of the near-surface band, in grid units
SAMPLE_NUM      = 4000    # points sampled per wing
SAMPLE_PERCENT  = 0.4     # fraction drawn from the band; remainder uniform

# --- Architecture ---
LATENT_DIM      = 16      # dimensionality of the per-wing latent code
HIDDEN_DIM      = 128     # width of each SIREN layer
OMEGA0_FIRST    = 30.0    # frequency scale, first layer
OMEGA0_HIDDEN   = 30.0    # frequency scale, hidden layers
LATENT_INIT_STD = 0.01    # std of the normal used to initialize latent codes

# --- Optimization ---
BATCH_SIZE      = 512
NUM_WORKERS     = 0       # keep 0 unless parallel loading is demonstrably needed

# --- Loss weights ---
LAMBDA_DIST = 0.5     # weight on the stress term
SIGMA       = 10      # latent prior scale; latent term is weighted by 1/SIGMA^2

# --- Optimization ---
EPOCHS      = CLI_ARGS.epochs
LR          = 1e-4
LR_MIN      = 1e-7    # cosine annealing floor
GRAD_CLIP   = 1.0     # max gradient norm

# --- Validation (test-time latent inference) ---
VAL_EVERY   = 2       # run validation every N epochs
VAL_STEPS   = 40      # optimization steps to fit validation latent codes
VAL_LR      = 1e-4
VAL_INIT_SEED_OFFSET = 10_000

# --- Checkpoint selection ---
# Training always runs for EPOCHS. Alongside the final state, retain the model
# and training latent codes from the lowest combined validation loss.
BEST_CHECKPOINT_METRIC = "val_total"
BEST_BEFORE_EPOCH = 40

# =============================================================================
# TRAINING DATA — sample points from the SDF fields
# =============================================================================
# Each wing is represented not by its full field but by a set of sampled
# points, oversampled near the surface (|sdf| < SAMPLE_EPSILON) so the network
# spends its capacity where the shape boundary actually is. Training uses all
# training wings; validation uses the subset selected earlier.
# =============================================================================

if not CLI_ARGS.inference_only:
    sampling_rng = np.random.default_rng(SEED)
    announce_phase(
        f"Sampling {SAMPLE_NUM} SDF points from each training wing"
    )
    train_ids, train_coords_s, train_dists = build_sdf_samples(
        train_sdfs,
        SAMPLE_EPSILON,
        SAMPLE_NUM,
        SAMPLE_PERCENT,
        rng=sampling_rng,
        description="Sampling training SDF points",
    )
    announce_phase(
        f"Sampling {SAMPLE_NUM} SDF points from each validation wing"
    )
    val_ids, val_coords_s, val_dists = build_sdf_samples(
        val_sdfs_sub,
        SAMPLE_EPSILON,
        SAMPLE_NUM,
        SAMPLE_PERCENT,
        rng=sampling_rng,
        description="Sampling validation SDF points",
    )
    
    train_loader = DataLoader(
        SDFDataset(train_ids, train_coords_s, train_dists),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=make_torch_generator(SEED),
        worker_init_fn=seed_data_loader_worker,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        SDFDataset(val_ids, val_coords_s, val_dists),
        batch_size=BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_data_loader_worker,
        num_workers=NUM_WORKERS,
    )
    
    print(f"Train samples: {len(train_ids)} from {len(train_sdfs)} wings")
    print(f"Val samples:   {len(val_ids)} from {len(val_sdfs_sub)} wings")
    
    # =============================================================================
    # MODEL SETUP — network, latent codes, target distances
    # =============================================================================
    # Latent codes are free parameters, one per training wing, optimized jointly
    # with the network (auto-decoder training). Validation wings get no code here;
    # theirs are inferred later with the network frozen.
    #
    # The distance matrices are moved to the device as targets for the stress term,
    # which pushes distances between latent codes to match the true L2 distances
    # between wing shapes. Only the upper triangle of D_train is used, since it is
    # symmetric with a zero diagonal. `den` and `den_val` are the normalizers that
    # make the stress loss scale-free.
    # =============================================================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Re-assert the configured seed immediately before parameter initialization.
    announce_phase(
        f"Initializing model and latent codes on {device} with seed {SEED}"
    )
    seed_everything(SEED)
    
    # --- Target distances: train-train ---
    num_train_wings = len(train_sdfs)
    D_tensor = torch.tensor(D_train, dtype=torch.float32).to(device)
    mask = torch.triu(torch.ones(num_train_wings, num_train_wings, device=device),
                      diagonal=1).bool()
    den = (D_tensor[mask] ** 2).sum().clamp(min=1e-8)
    
    # --- Target distances: val-train ---
    N_val = len(val_sdfs_sub)
    D_val_tensor = torch.tensor(D_val_sub, dtype=torch.float32).to(device)
    den_val = (D_val_tensor ** 2).sum().clamp(min=1e-8)
    
    # --- Model and trainable latent codes ---
    latent_codes = nn.Embedding(num_train_wings, LATENT_DIM).to(device)
    normal_init_with_seed(
        latent_codes.weight,
        mean=0.0,
        std=LATENT_INIT_STD,
        seed=SEED + 1,
    )
    
    model = SirenDeepSDF2D(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                           omega0_first=OMEGA0_FIRST,
                           omega0_hidden=OMEGA0_HIDDEN).to(device)
    
    # --- Loss curve trackers ---
    epoch_counter = 0
    
    train_loss_curve = []
    train_sdf_loss_curve = []
    train_stress_curve = []
    train_latent_curve = []
    
    val_loss_curve = []
    val_sdf_loss_curve = []
    val_stress_curve = []
    val_latent_curve = []
    
    # =============================================================================
    # TRAINING — auto-decoder with distance-preserving latent space
    # =============================================================================
    # The network and the training latent codes are optimized jointly. Three terms:
    #
    #   sdf     — reconstruction: predicted SDF value vs ground truth at each
    #             sampled point.
    #   latent  — prior keeping codes near the origin, weighted by 1/SIGMA^2.
    #   stress  — Kruskal stress-1 between pairwise distances in latent space and
    #             the true L2 distances between wing shapes (D_train). This is what
    #             makes the latent space metric-faithful rather than arbitrary.
    #
    # Validation runs every VAL_EVERY epochs. Validation wings have no learned
    # codes, so fresh ones are fitted from scratch with the network frozen
    # (test-time optimization), measuring how well an unseen wing can be embedded.
    # Their stress is measured against the training codes via D_val.
    #
    # Curves are appended across calls, so re-running this cell continues training
    # rather than restarting it.
    # =============================================================================
    
    # Sampled points as device tensors, indexed once for per-wing evaluation
    train_ids_t    = torch.as_tensor(train_ids, device=device)
    train_coords_t = torch.as_tensor(train_coords_s, device=device)
    train_dists_t  = torch.as_tensor(train_dists, device=device)
    val_ids_t      = torch.as_tensor(val_ids, device=device)
    val_coords_t   = torch.as_tensor(val_coords_s, device=device)
    val_dists_t    = torch.as_tensor(val_dists, device=device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(latent_codes.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN)
    best_validation_loss = float("inf")
    best_epoch = None
    best_model_state = None
    best_latent_codes_state = None
    best_before_40_validation_loss = float("inf")
    best_before_40_epoch = None
    best_before_40_model_state = None
    best_before_40_latent_codes_state = None
    
    col_ids = torch.arange(num_train_wings, device=device).unsqueeze(0)   # (1, N)
    
    announce_phase(
        f"Training for {EPOCHS} epochs "
        "(saving final, best-validation, and best-through-40 checkpoints)"
    )
    epoch_progress = tqdm(
        range(EPOCHS),
        desc=f"SliceLearn training (seed {SEED})",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_progress:
        epoch_counter += 1
    
        # ---------------- TRAINING ----------------
        model.train()
        train_batch_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch_counter} train batches",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for shape_ids, coords, distances in train_batch_progress:
            shape_ids = shape_ids.to(device)
            coords    = coords.to(device)
            distances = distances.to(device)
    
            z = latent_codes(shape_ids)
            pred = model(z, coords).squeeze(1)
            sdf_loss = (pred - distances).pow(2).mean()
    
            # Terms below act per wing, not per point, so work on unique ids only
            unique_ids = torch.unique(shape_ids)
            z_unique   = latent_codes(unique_ids)
            latent_loss = z_unique.pow(2).sum(dim=1).mean()
    
            # Stress of this batch's wings against all training wings,
            # excluding each wing's zero distance to itself
            self_mask = (unique_ids.unsqueeze(1) != col_ids)
            dist_loss = compute_stress(z_unique, latent_codes.weight,
                                       D_tensor[unique_ids], den, mask=self_mask)
    
            loss = sdf_loss + (1 / SIGMA**2) * latent_loss + LAMBDA_DIST * dist_loss
    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(latent_codes.parameters()),
                max_norm=GRAD_CLIP)
            optimizer.step()
    
        scheduler.step()
    
        # ---------------- TRAIN METRICS ----------------
        model.eval()
        with torch.no_grad():
            train_sdf = evaluate_sdf_loss(model, latent_codes, train_ids_t,
                                          train_coords_t, train_dists_t,
                                          num_train_wings, device)
            all_z = latent_codes.weight
            train_latent = all_z.pow(2).sum(dim=1).mean().item()
            train_stress_sq = compute_stress(all_z, all_z, D_tensor, den, mask=mask).item()
            train_stress = np.sqrt(train_stress_sq)      # Kruskal stress-1, reported
    
        # Total is the objective actually minimized, so it uses the un-rooted term
        train_total = (train_sdf + (1 / SIGMA**2) * train_latent
                       + LAMBDA_DIST * train_stress_sq)
    
        train_sdf_loss_curve.append(train_sdf)
        train_latent_curve.append(train_latent)
        train_stress_curve.append(train_stress)
        train_loss_curve.append(train_total)
    
        epoch_progress.set_postfix(
            train=f"{train_total:.4g}",
            refresh=True,
        )
        tqdm.write(
            f"Epoch {epoch_counter} | TRAIN | sdf {train_sdf:.6f} "
            f"| latent {train_latent:.6f} | stress {train_stress:.6f}"
        )
    
        # ---------------- VALIDATION ----------------
        if epoch_counter % VAL_EVERY == 0 or epoch_counter == EPOCHS:
            model.eval()
    
            # Fresh codes for the unseen wings; the network stays frozen
            val_latent_codes = nn.Embedding(N_val, LATENT_DIM).to(device)
            # Reuse the same seeded starting codes at every validation check so
            # improvements reflect the model, not a luckier latent initialization.
            normal_init_with_seed(
                val_latent_codes.weight,
                mean=0.0,
                std=LATENT_INIT_STD,
                seed=SEED + VAL_INIT_SEED_OFFSET,
            )
            val_optimizer = torch.optim.Adam(val_latent_codes.parameters(), lr=VAL_LR)
    
            validation_progress = tqdm(
                total=VAL_STEPS * len(val_loader),
                desc=f"Epoch {epoch_counter} validation latent fit",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for _ in range(VAL_STEPS):
                for shape_ids, coords, distances in val_loader:
                    shape_ids = shape_ids.to(device)
                    coords    = coords.to(device)
                    distances = distances.to(device)
    
                    z = val_latent_codes(shape_ids)
                    pred = model(z, coords).squeeze(1)
                    sdf_loss = (pred - distances).pow(2).mean()
    
                    unique_ids = torch.unique(shape_ids)
                    z_unique   = val_latent_codes(unique_ids)
                    latent_loss = z_unique.pow(2).sum(dim=1).mean()
    
                    # Distances measured against the frozen training codes
                    dist_loss = compute_stress(z_unique, latent_codes.weight.detach(),
                                               D_val_tensor[unique_ids], den_val)
    
                    loss = (sdf_loss + (1 / SIGMA**2) * latent_loss
                            + LAMBDA_DIST * dist_loss)
    
                    val_optimizer.zero_grad()
                    loss.backward()
                    val_optimizer.step()
                    validation_progress.update(1)
            validation_progress.close()
    
            with torch.no_grad():
                val_sdf = evaluate_sdf_loss(model, val_latent_codes, val_ids_t,
                                            val_coords_t, val_dists_t, N_val, device)
                all_z_val = val_latent_codes.weight
                val_latent = all_z_val.pow(2).sum(dim=1).mean().item()
                val_stress_sq = compute_stress(all_z_val, latent_codes.weight.detach(),
                                               D_val_tensor, den_val).item()
                val_stress = np.sqrt(val_stress_sq)
    
            val_total = (val_sdf + (1 / SIGMA**2) * val_latent
                         + LAMBDA_DIST * val_stress_sq)
    
            val_sdf_loss_curve.append(val_sdf)
            val_latent_curve.append(val_latent)
            val_stress_curve.append(val_stress)
            val_loss_curve.append(val_total)
    
            epoch_progress.set_postfix(
                train=f"{train_total:.4g}",
                val=f"{val_total:.4g}",
                best=f"{min(best_validation_loss, val_total):.4g}",
                refresh=True,
            )
            tqdm.write(
                f"Epoch {epoch_counter} | VAL   | sdf {val_sdf:.6f} "
                f"| latent {val_latent:.6f} | stress {val_stress:.6f}"
            )
    
            is_new_best = val_total < best_validation_loss
            is_new_best_before_40 = (
                epoch_counter <= BEST_BEFORE_EPOCH
                and val_total < best_before_40_validation_loss
            )
            if is_new_best or is_new_best_before_40:
                validation_model_state = clone_state_dict(model)
                validation_latent_codes_state = clone_state_dict(latent_codes)

            if is_new_best:
                best_validation_loss = float(val_total)
                best_epoch = int(epoch_counter)
                best_model_state = validation_model_state
                best_latent_codes_state = validation_latent_codes_state
                tqdm.write(
                    f"New best validation checkpoint | epoch {best_epoch} "
                    f"| {BEST_CHECKPOINT_METRIC} {best_validation_loss:.6f}"
                )
            else:
                tqdm.write(
                    f"Best validation checkpoint | epoch {best_epoch} "
                    f"| {BEST_CHECKPOINT_METRIC} {best_validation_loss:.6f}"
                )

            if is_new_best_before_40:
                best_before_40_validation_loss = float(val_total)
                best_before_40_epoch = int(epoch_counter)
                best_before_40_model_state = validation_model_state
                best_before_40_latent_codes_state = (
                    validation_latent_codes_state
                )
                tqdm.write(
                    f"New best-through-{BEST_BEFORE_EPOCH} checkpoint "
                    f"| epoch {best_before_40_epoch} "
                    f"| {BEST_CHECKPOINT_METRIC} "
                    f"{best_before_40_validation_loss:.6f}"
                )
    
    epoch_progress.close()
    
    # =============================================================================
    # SAVE CHECKPOINT
    # =============================================================================
    # Writes everything needed to reproduce or reload the trained model: network
    # weights, training latent codes, the loss curves, and the hyperparameters they
    # were produced with. Storing the hyperparameters alongside the weights means a
    # checkpoint is self-describing — the architecture can be rebuilt from the file
    # without consulting the notebook.
    #
    # The latent codes are also written as a plain .npy for analysis outside torch.
    # =============================================================================
    
    checkpoint_path = os.path.join(OUTPUT_ROOT, "sirendeepsdf_checkpoint.pt")
    best_checkpoint_path = os.path.join(
        OUTPUT_ROOT, "sirendeepsdf_checkpoint_best.pt"
    )
    best_before_40_checkpoint_path = os.path.join(
        OUTPUT_ROOT, "sirendeepsdf_checkpoint_best_before_40.pt"
    )

    if best_epoch is None or best_before_40_epoch is None:
        raise RuntimeError("No validation checkpoint was recorded during training")

    def checkpoint_payload(
        model_state,
        latent_state,
        checkpoint_kind,
        selected_epoch,
        selection_metric,
        selection_value,
    ):
        """Build matching checkpoint dictionaries for all saved states."""
        return {
            "model_state_dict": model_state,
            "latent_codes_state_dict": latent_state,
            "checkpoint_kind": checkpoint_kind,
            "selected_epoch": selected_epoch,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "epochs_trained": epoch_counter,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_before_40_epoch": best_before_40_epoch,
            "best_before_40_validation_loss": (
                best_before_40_validation_loss
            ),
            "train_files": train_files,       # row order of the latent codes
            "validation_files": val_files_sub,
            "config": {
                "S": S, "SEED": SEED,
                "DATASET_ROOT": str(DATASET_ROOT),
                "TRAIN_SPLIT": TRAIN_SPLIT,
                "VALIDATION_SPLIT": VALIDATION_SPLIT,
                "TEST_SPLIT": TEST_SPLIT,
                "TARGET_SIZE": TARGET_SIZE, "ROOT_CROP_X": ROOT_CROP_X,
                "SAMPLE_EPSILON": SAMPLE_EPSILON, "SAMPLE_NUM": SAMPLE_NUM,
                "SAMPLE_PERCENT": SAMPLE_PERCENT,
                "LATENT_DIM": LATENT_DIM, "HIDDEN_DIM": HIDDEN_DIM,
                "OMEGA0_FIRST": OMEGA0_FIRST,
                "OMEGA0_HIDDEN": OMEGA0_HIDDEN,
                "LATENT_INIT_STD": LATENT_INIT_STD,
                "BATCH_SIZE": BATCH_SIZE, "NUM_WORKERS": NUM_WORKERS,
                "EPOCHS": EPOCHS, "LR": LR, "LR_MIN": LR_MIN,
                "LAMBDA_DIST": LAMBDA_DIST,
                "SIGMA": SIGMA,
                "GRAD_CLIP": GRAD_CLIP,
                "VAL_EVERY": VAL_EVERY,
                "VAL_STEPS": VAL_STEPS,
                "VAL_LR": VAL_LR,
                "VAL_INIT_SEED_OFFSET": VAL_INIT_SEED_OFFSET,
                "BEST_CHECKPOINT_METRIC": BEST_CHECKPOINT_METRIC,
                "BEST_BEFORE_EPOCH": BEST_BEFORE_EPOCH,
            },
            "curves": {
                "train_loss": train_loss_curve,
                "train_sdf": train_sdf_loss_curve,
                "train_stress": train_stress_curve,
                "train_latent": train_latent_curve,
                "val_loss": val_loss_curve,
                "val_sdf": val_sdf_loss_curve,
                "val_stress": val_stress_curve,
                "val_latent": val_latent_curve,
            },
        }

    announce_phase(f"Saving final checkpoint to {checkpoint_path}")
    torch.save(
        checkpoint_payload(
            model.state_dict(),
            latent_codes.state_dict(),
            checkpoint_kind="final",
            selected_epoch=epoch_counter,
            selection_metric="final_epoch",
            selection_value=epoch_counter,
        ),
        checkpoint_path,
    )
    torch.save(
        checkpoint_payload(
            best_model_state,
            best_latent_codes_state,
            checkpoint_kind="best",
            selected_epoch=best_epoch,
            selection_metric=BEST_CHECKPOINT_METRIC,
            selection_value=best_validation_loss,
        ),
        best_checkpoint_path,
    )
    torch.save(
        checkpoint_payload(
            best_before_40_model_state,
            best_before_40_latent_codes_state,
            checkpoint_kind="best_before_40",
            selected_epoch=best_before_40_epoch,
            selection_metric=(
                f"{BEST_CHECKPOINT_METRIC}_through_epoch_"
                f"{BEST_BEFORE_EPOCH}"
            ),
            selection_value=best_before_40_validation_loss,
        ),
        best_before_40_checkpoint_path,
    )
    
    np.save(os.path.join(OUTPUT_ROOT, "latent_codes_train.npy"),
            latent_codes.weight.detach().cpu().numpy())
    np.save(
        os.path.join(OUTPUT_ROOT, "latent_codes_train_best.npy"),
        best_latent_codes_state["weight"].numpy(),
    )
    np.save(
        os.path.join(OUTPUT_ROOT, "latent_codes_train_best_before_40.npy"),
        best_before_40_latent_codes_state["weight"].numpy(),
    )

    print(f"Saved final checkpoint to {checkpoint_path} (epoch {epoch_counter})")
    print(
        f"Saved best checkpoint to {best_checkpoint_path} "
        f"(epoch {best_epoch}, {BEST_CHECKPOINT_METRIC} "
        f"{best_validation_loss:.6f})"
    )
    print(
        f"Saved best-through-{BEST_BEFORE_EPOCH} checkpoint to "
        f"{best_before_40_checkpoint_path} "
        f"(epoch {best_before_40_epoch}, {BEST_CHECKPOINT_METRIC} "
        f"{best_before_40_validation_loss:.6f})"
    )
    
    # =============================================================================
    # SANITY CHECK — reconstruction quality after training
    # =============================================================================
    # Two checks on the trained model, both on training wings (i.e. the fit itself,
    # not generalization):
    #
    #   1. Latent stress. Kruskal stress-1 between distances in latent space and
    #      the true L2 shape distances. 0 means the embedding reproduces the shape
    #      metric exactly; values below ~0.05 are usually considered a good fit,
    #      ~0.1 acceptable. Compare against the stress at initialization to confirm
    #      training actually improved the embedding.
    #
    #   2. Reconstruction. For a few chosen wings, the decoded zero-contour is
    #      overlaid on the ground-truth shape. A good model traces the outline
    #      closely; systematic gaps mean the network is underfitting, while a
    #      ragged or broken contour means the latent code failed to converge.
    #
    # Diagnostic only — nothing downstream depends on this cell.
    # =============================================================================
    
    SANITY_SHAPE_IDS = [0, 50, 134]     # training wing indices to inspect
    
    # --- 1. Latent stress on the training set ---
    with torch.no_grad():
        final_stress = np.sqrt(compute_stress(latent_codes.weight, latent_codes.weight,
                                      D_tensor, den, mask=mask).item())
    
    print(f"Training latent stress (Kruskal stress-1): {final_stress:.6f}")
    print(f"Trained for {epoch_counter} epochs on {num_train_wings} wings")
    
    # --- 2. Predicted vs ground-truth contours ---
    fig, axes = plt.subplots(1, len(SANITY_SHAPE_IDS),
                             figsize=(5 * len(SANITY_SHAPE_IDS), 5))
    axes = np.atleast_1d(axes)
    
    for ax, shape_idx in zip(axes, SANITY_SHAPE_IDS):
        true_sdf = train_sdfs[shape_idx]
        pred = predicted_sdf(model, get_z(latent_codes, shape_idx), set_grid)
    
        X, Y = true_sdf[:, :, 0], true_sdf[:, :, 1]
        Z_true, Z_pred = true_sdf[:, :, 2], pred[:, :, 2]
    
        # ground truth: filled interior (sdf <= 0) plus its outline
        ax.contourf(X, Y, Z_true, levels=[Z_true.min(), 0], colors=['lightgray'])
        ax.contour(X, Y, Z_true, levels=[0], colors='black', linewidths=1.5)
    
        # prediction: decoded zero-contour
        ax.contour(X, Y, Z_pred, levels=[0], colors='red', linewidths=1.5)
    
        rmse = float(np.sqrt(np.mean((Z_pred - Z_true) ** 2)))
        ax.set_title(f"wing {shape_idx} — {train_files[shape_idx]}\nSDF RMSE {rmse:.4f}",
                     fontsize=9)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
    
    fig.suptitle('Reconstruction: ground truth (black) vs decoded (red)')
    plt.tight_layout()
    sanity_path = diagnostics_dir / "training_reconstruction_sanity.png"
    fig.savefig(sanity_path, dpi=180)
    plt.close(fig)
    tqdm.write(f"Saved training reconstruction diagnostic: {sanity_path}")
    
    # =============================================================================
    # TRAINING CURVES
    # =============================================================================
    # Train and validation are plotted as separate figures rather than shared axes,
    # because validation losses are much larger in absolute terms and would flatten
    # the training curves into a straight line if drawn together.
    #
    # Four panels per figure:
    #   sdf         — reconstruction error. This term alone trains the decoder,
    #                 since the other two do not depend on its weights.
    #   stress      — Kruskal stress-1: how faithfully latent distances reproduce
    #                 true shape distances. Acts on the latent codes only.
    #   weighted    — the three terms as they actually enter the objective, i.e.
    #                 after their weights, on a log axis. This is the honest view
    #                 of what the optimizer sees: the terms differ by orders of
    #                 magnitude, so the stress term carries almost all of the total.
    #                 They are not in competition for the same parameters, though —
    #                 only the latent codes receive gradients from all three.
    #   total       — the weighted sum being minimized. Included for completeness;
    #                 because it is dominated by the stress term it carries little
    #                 information the stress panel does not already show.
    #
    # The stress curves store Kruskal stress-1 (rooted), while the objective uses
    # the un-rooted normalized stress, so the weighted panel squares it back.
    #
    # Validation runs every VAL_EVERY epochs, so its points are placed on the
    # correct epoch numbers rather than against the validation index.
    # =============================================================================
    
    train_epochs = np.arange(1, len(train_loss_curve) + 1)
    val_epochs = np.asarray([
        epoch
        for epoch in range(1, epoch_counter + 1)
        if epoch % VAL_EVERY == 0 or epoch == epoch_counter
    ])
    
    # Weighted contributions, as the terms enter the objective
    w_sdf_train    = np.array(train_sdf_loss_curve)
    w_latent_train = (1 / SIGMA**2) * np.array(train_latent_curve)
    w_stress_train = LAMBDA_DIST * np.array(train_stress_curve) ** 2
    
    w_sdf_val    = np.array(val_sdf_loss_curve)
    w_latent_val = (1 / SIGMA**2) * np.array(val_latent_curve)
    w_stress_val = LAMBDA_DIST * np.array(val_stress_curve) ** 2
    
    figures = [
        ("Training", 'tab:blue', train_epochs,
         train_sdf_loss_curve, train_stress_curve, train_loss_curve,
         w_sdf_train, w_latent_train, w_stress_train),
        ("Validation", 'tab:orange', val_epochs,
         val_sdf_loss_curve, val_stress_curve, val_loss_curve,
         w_sdf_val, w_latent_val, w_stress_val),
    ]
    
    for (split, color, epochs_axis, sdf_curve, stress_curve, total_curve,
         w_sdf, w_latent, w_stress) in figures:
    
        if len(epochs_axis) == 0:            # nothing logged for this split yet
            continue
    
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    
        # --- individual terms ---
        for ax, title, curve in [(axes[0], 'SDF reconstruction', sdf_curve),
                                 (axes[1], 'Stress (Kruskal-1)', stress_curve)]:
            ax.plot(epochs_axis, curve, color=color, marker='o', markersize=3)
            ax.set_title(title)
            ax.set_ylabel('loss')
    
        # --- weighted contributions to the objective ---
        ax = axes[2]
        ax.plot(epochs_axis, w_sdf,    label=r'$\mathcal{L}_{sdf}$')
        ax.plot(epochs_axis, w_latent, label=r'$\sigma^{-2}\mathcal{L}_{lat}$')
        ax.plot(epochs_axis, w_stress, label=r'$\lambda_{dist}\mathcal{L}_{stress}$')
        ax.set_yscale('log')                 # terms span several orders of magnitude
        ax.set_title('Weighted contributions')
        ax.set_ylabel('contribution to total')
        ax.legend(fontsize=8)
    
        # --- total ---
        ax = axes[3]
        ax.plot(epochs_axis, total_curve, color=color, marker='o', markersize=3)
        ax.set_title('Total loss (minimized)')
        ax.set_ylabel('loss')
    
        for ax in axes:
            ax.set_xlabel('epoch')
            ax.grid(alpha=0.3)
    
        fig.suptitle(f'{split} curves — {epoch_counter} epochs, '
                     f'{num_train_wings} training wings')
        plt.tight_layout()
        curve_path = diagnostics_dir / f"{split.lower()}_training_curves.png"
        fig.savefig(curve_path, dpi=180)
        plt.close(fig)
        tqdm.write(f"Saved {split.lower()} curves: {curve_path}")
    
    # --- final values and the balance between terms ---
    for split, sdf_c, stress_c, total_c, ws, wl, wst in [
            ('train', train_sdf_loss_curve, train_stress_curve, train_loss_curve,
             w_sdf_train, w_latent_train, w_stress_train),
            ('val',   val_sdf_loss_curve,   val_stress_curve,   val_loss_curve,
             w_sdf_val, w_latent_val, w_stress_val)]:
    
        if len(total_c) == 0:
            continue
    
        total = ws[-1] + wl[-1] + wst[-1]
        print(f"final {split:<5} | sdf {sdf_c[-1]:.3e} | stress {stress_c[-1]:.6f} "
              f"| total {total_c[-1]:.3e}")
        print(f"{'':>12}share of objective — sdf {100*ws[-1]/total:5.1f}% | "
              f"latent {100*wl[-1]/total:5.1f}% | stress {100*wst[-1]/total:5.1f}%")
    
"""**PHASE 3 - SHAPE RECONSTRUCTION **"""

# =============================================================================
# LOAD CHECKPOINT
# =============================================================================
# Entry point for any session that does not train from scratch. Training may
# have happened much earlier, so this cell rebuilds the decoder and the
# training latent codes from disk; everything below it can then run without
# retraining.
#
# The architecture is rebuilt from the hyperparameters stored *inside* the
# checkpoint, not from the notebook's config cell, so a checkpoint always
# reconstructs the network it was actually trained with. The two are compared
# afterwards and any mismatch is reported: loading a model trained at one grid
# resolution into a session configured for another fails silently and produces
# plausible-looking nonsense.
#
# Requires: the imports cell, and Cell A (SineLayer / SirenDeepSDF2D).
# =============================================================================

checkpoint_filenames = {
    "best": "sirendeepsdf_checkpoint_best.pt",
    "best-before-40": "sirendeepsdf_checkpoint_best_before_40.pt",
    "final": "sirendeepsdf_checkpoint.pt",
}
checkpoint_filename = checkpoint_filenames[CLI_ARGS.checkpoint_kind]
checkpoint_path = os.path.join(OUTPUT_ROOT, checkpoint_filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# weights_only=False is needed because the checkpoint also stores config and
# curves, not just tensors. Only load checkpoints you trust.
ckpt = None
if (
    CLI_ARGS.checkpoint_kind == "best"
    and not os.path.isfile(checkpoint_path)
):
    # Checkpoints produced before separate final/best files were introduced
    # stored the restored best-validation state at the historical filename.
    legacy_path = os.path.join(OUTPUT_ROOT, "sirendeepsdf_checkpoint.pt")
    if os.path.isfile(legacy_path):
        legacy_ckpt = torch.load(
            legacy_path,
            map_location=device,
            weights_only=False,
        )
        if "checkpoint_kind" not in legacy_ckpt:
            checkpoint_path = legacy_path
            ckpt = legacy_ckpt
            print(
                "Using legacy checkpoint containing the restored "
                "best-validation state."
            )

if ckpt is None and not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
        f"Requested {CLI_ARGS.checkpoint_kind!r} checkpoint does not exist at "
        f"{checkpoint_path}. Train seed {SEED} first or select an available "
        "--checkpoint-kind."
    )
announce_phase(f"Reloading checkpoint from {checkpoint_path}")
if ckpt is None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
cfg = ckpt["config"]

# --- Rebuild the decoder with the architecture it was trained with ---
model = SirenDeepSDF2D(latent_dim=cfg["LATENT_DIM"],
                       hidden_dim=cfg["HIDDEN_DIM"],
                       omega0_first=cfg["OMEGA0_FIRST"],
                       omega0_hidden=cfg["OMEGA0_HIDDEN"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- Restore the training latent codes ---
# Wing count is taken from the saved codes themselves rather than hardcoded.
num_train_wings = ckpt["latent_codes_state_dict"]["weight"].shape[0]
latent_codes = nn.Embedding(num_train_wings, cfg["LATENT_DIM"]).to(device)
latent_codes.load_state_dict(ckpt["latent_codes_state_dict"])

# --- Restore training history, so the curve cells work in a load-only run ---
epoch_counter        = ckpt["epochs_trained"]
best_epoch           = ckpt.get("best_epoch")
checkpoint_kind      = ckpt.get("checkpoint_kind", "legacy_best")
selected_epoch       = ckpt.get("selected_epoch", best_epoch)
train_files          = ckpt["train_files"]
curves               = ckpt["curves"]
train_loss_curve     = curves["train_loss"]
train_sdf_loss_curve = curves["train_sdf"]
train_stress_curve   = curves["train_stress"]
train_latent_curve   = curves["train_latent"]
val_loss_curve       = curves["val_loss"]
val_sdf_loss_curve   = curves["val_sdf"]
val_stress_curve     = curves["val_stress"]
val_latent_curve     = curves["val_latent"]

# --- Report, and warn if the session config disagrees with the checkpoint ---
print(f"Loaded {checkpoint_path}")
print(f"  {num_train_wings} training wings | {epoch_counter} epochs trained")
print(f"  checkpoint kind: {checkpoint_kind} | selected epoch: {selected_epoch}")
if best_epoch is not None and checkpoint_kind == "final":
    print(f"  best validation epoch during training: {best_epoch}")
print(f"  device: {next(model.parameters()).device}")

for name, session_value in [("S", S), ("LATENT_DIM", LATENT_DIM),
                            ("HIDDEN_DIM", HIDDEN_DIM),
                            ("TARGET_SIZE", TARGET_SIZE),
                            ("ROOT_CROP_X", ROOT_CROP_X)]:
    if cfg.get(name) != session_value:
        print(f"  WARNING: {name} is {session_value} in this session "
              f"but {cfg.get(name)} in the checkpoint")

# =============================================================================
# RECONSTRUCTION COMPONENTS
# =============================================================================
# Functions used by the reconstruction phase, where the decoder is frozen and
# an observed (damaged) wing is explained by finding both a latent code and a
# similarity transform placing that shape onto the observation.
#
# The transform is parameterized as (log_lam, angle, t): log-scale rather than
# scale so it stays positive under unconstrained optimization, and a single
# angle rather than a matrix so the rotation is exactly a rotation.
# =============================================================================

# Smooth (sigmoid) approximation of the Heaviside step. Turns an SDF into a
# soft occupancy field: ~1 inside the shape, ~0 outside, with a transition of
# width ~eps across the boundary. Differentiable, unlike a hard threshold,
# which is what allows the shape to be compared by gradient descent.
def torch_smooth_heaviside(sdf, eps=0.01):
    return torch.sigmoid(-sdf / eps)


# Log-density of z under a Parzen (kernel density) estimate built from the
# training latents. Used as the shape prior: maximizing it keeps the optimized
# code inside the region of latent space the decoder was actually trained on,
# instead of drifting into territory where its output is meaningless.
def parzen_log_density(z, Z_all, sigma=1.0):
    dists_sq = torch.sum((Z_all - z) ** 2, dim=1)
    log_kernels = -dists_sq / (2 * sigma ** 2)
    return torch.logsumexp(log_kernels, dim=0)


# Binary weight mask that discards query points left of the root cut, so the
# region removed from the training shapes is not scored during fitting.
def root_region_weight_from_Xquery(Xquery, root_x=-0.7, root_margin=0.08):
    x = Xquery[:, 0]
    weights = torch.ones_like(x)
    weights[x < root_x + root_margin] = 0.0
    return weights


# Decodes z under a similarity transform, differentiably. Output grid points
# are mapped back into model space by the inverse transform and queried there,
# so gradients flow to log_lam, angle and t as well as to z.
# Returns the SDF values and the query coordinates (needed for spatial weights).
def predicted_sdf_transformed_torch(model, z, coords, log_lam, angle, t):
    lam = torch.exp(log_lam)
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    Xquery = ((coords - t) @ R) / lam
    z_rep = z.repeat(coords.shape[0], 1)
    sdf_vals = model(z_rep, Xquery).squeeze()
    return sdf_vals, Xquery


# Non-differentiable counterpart of the above, for numpy grids. Returns an
# (S, S, 3) array in the usual (x, y, value) layout.
def predicted_sdf_transformed(model, z, set_grid, lam, R, t):
    model.eval()
    Ycoords = set_grid.reshape(-1, 2)
    Xquery = ((Ycoords - t) @ R) / lam
    coords = torch.tensor(Xquery, dtype=torch.float32).to(device)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    z = z.to(device)
    z_rep = z.repeat(coords.shape[0], 1)
    with torch.no_grad():
        pred_vals = model(z_rep, coords)
    pred_vals = pred_vals.squeeze().cpu().numpy()
    out_flat = np.concatenate([Ycoords, pred_vals[:, None]], axis=1)
    H, W = set_grid.shape[:2]
    return out_flat.reshape(H, W, 3)


# Picks a diverse set of starting latents by clustering the training codes and
# returning, for each cluster centre, the index of the nearest real code. The
# fit is non-convex in z, so the initialization matters; sweeping over spread
# representatives is cheaper than a random restart and covers the space better.
def get_cluster_representatives(Z_all, n_clusters=10, random_state=SEED):
    Z_np = Z_all.detach().cpu().numpy()
    # Use an explicit integer for compatibility with older scikit-learn
    # versions (including the cluster's version), which accept ``n_init`` as
    # an integer but fail during fit when given the newer ``"auto"`` value.
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    kmeans.fit(Z_np)

    representative_indices = []
    for centroid in kmeans.cluster_centers_:
        dists = np.linalg.norm(Z_np - centroid, axis=1)
        representative_indices.append(int(np.argmin(dists)))
    return representative_indices


# Writes a binary mask as a PNG: shape black on white, matching the convention
# of the input scans. Expects a 2D array in image orientation.
def save_binary_png(binary, save_path):
    img = (1 - binary.astype(np.uint8)) * 255
    Image.fromarray(img).save(save_path)


# Decodes a latent code under a similarity transform and thresholds it into a
# binary mask, returned in image orientation so it overlays the input PNG
# pixel-for-pixel.
def binary_from_transform(model, z, set_grid, lam, angle, t, device):
    coords = torch.tensor(set_grid.reshape(-1, 2), dtype=torch.float32, device=device)
    z = z.to(device)
    if z.dim() == 1:
        z = z.unsqueeze(0)

    log_lam = torch.tensor(np.log(lam), dtype=torch.float32, device=device)
    angle_t = torch.tensor(float(angle), dtype=torch.float32, device=device)
    t_t     = torch.tensor(np.asarray(t), dtype=torch.float32, device=device)

    with torch.no_grad():
        sdf, _ = predicted_sdf_transformed_torch(model, z, coords, log_lam, angle_t, t_t)

    H, W = set_grid.shape[:2]
    binary = (sdf.reshape(H, W).cpu().numpy() <= 0).astype(np.uint8)
    return binary[::-1, :]        # grid -> image orientation


# Keeps only the largest connected component of a binary mask, removing
# speckle and detached fragments.
def keep_largest_component(binary, connectivity=2):
    binary = binary > 0
    structure = (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
                 if connectivity == 1 else np.ones((3, 3)))
    labeled, num = label(binary, structure=structure)
    if num == 0:
        return binary.astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)


# --- Diagnostics -------------------------------------------------------------

# Overlays the target occupancy field (blue) and the current prediction (red)
# during optimization, to watch the fit converge.
def plot_current_characteristic(model, z, set_grid, chi_y, log_lam, angle, t,
                                device, step, eps_chi=0.005):
    H, W = set_grid.shape[:2]
    coords = torch.tensor(set_grid.reshape(-1, 2), dtype=torch.float32, device=device)
    with torch.no_grad():
        sdf_current, _ = predicted_sdf_transformed_torch(
            model=model, z=z, coords=coords, log_lam=log_lam, angle=angle, t=t)

    C2 = torch_smooth_heaviside(sdf_current.reshape(H, W),
                                eps=eps_chi).detach().cpu().numpy()
    X, Y, C1 = chi_y[:, :, 0], chi_y[:, :, 1], chi_y[:, :, 2]

    figure = plt.figure(figsize=(7, 7))
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    plt.imshow(C1, extent=extent, origin="lower", alpha=0.5, vmin=0, vmax=1)
    plt.imshow(C2, extent=extent, origin="lower", alpha=0.5, vmin=0, vmax=1)
    plt.contour(X, Y, C1, levels=[0.5], colors="blue", linewidths=2)
    plt.contour(X, Y, C2, levels=[0.5], colors="red", linewidths=2)
    plt.scatter([0], [0], color="black", s=25)          # origin
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect("equal")
    plt.title(f"target y (blue) vs current prediction (red) | step {step}")
    inference_diagnostics = diagnostics_dir / "inference"
    inference_diagnostics.mkdir(parents=True, exist_ok=True)
    figure_path = (
        inference_diagnostics
        / f"characteristic_overlay_step_{step:04d}.png"
    )
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    tqdm.write(f"Saved inference characteristic diagnostic: {figure_path}")


# Projects the current code onto the 2D PCA of the training latents, showing
# where the optimization has travelled relative to the training distribution.
def plot_current_latent_pca(latent_codes, z, init_shape_idx, step):
    with torch.no_grad():
        Z = latent_codes.weight.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
    if z_np.ndim == 1:
        z_np = z_np[None, :]

    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)
    z_2d = pca.transform(z_np)

    figure = plt.figure(figsize=(8, 8))
    plt.scatter(Z_2d[:, 0], Z_2d[:, 1], color="red", alpha=0.8, label="training latents")
    for idx, (x, y) in enumerate(Z_2d):
        plt.text(x, y, str(idx), fontsize=8)
    plt.scatter(Z_2d[init_shape_idx, 0], Z_2d[init_shape_idx, 1],
                color="green", s=180, edgecolors="black", label="initial latent")
    plt.scatter(z_2d[:, 0], z_2d[:, 1],
                color="blue", s=220, edgecolors="black", label="current estimate")
    plt.title(f"PCA latent space | step {step}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    inference_diagnostics = diagnostics_dir / "inference"
    inference_diagnostics.mkdir(parents=True, exist_ok=True)
    figure_path = inference_diagnostics / f"latent_pca_step_{step:04d}.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    tqdm.write(f"Saved inference latent diagnostic: {figure_path}")

# =============================================================================
# RECONSTRUCTION HYPERPARAMETERS
# =============================================================================
# Settings for fitting a latent code and a pose to an observed wing. The
# decoder is frozen throughout; only z and the transform are optimized.
# These are the values used for the results reported in the paper.
# =============================================================================

# --- Initialization sweep ---
N_CLUSTERS   = 10      # candidate starting latents (k-means representatives)
PRE_STEPS    = 30      # pose-only steps per candidate during the sweep

# --- Main optimization ---
OPT_STEPS    = CLI_ARGS.inference_steps
OPT_LR_G     = 1e-2    # learning rate for the pose (log_lam, angle, t), Adam
OPT_LR_Z     = 1e-4    # learning rate for the latent code, plain gradient step

# --- Data term ---
CHI_EPS      = 0.005   # smooth-Heaviside width; smaller = sharper boundary
LOSS_P       = 2       # exponent of the pointwise occupancy mismatch
OUTSIDE_W    = 0.001   # weight of points outside the observation vs inside

# --- Spatial weighting ---
OPT_ROOT_X      = -0.8    # root cut in query space (matches ROOT_CROP_X)
OPT_ROOT_MARGIN = 0.00    # extra margin discarded beyond the cut
REGION_X_MIN    = -0.75   # emphasized region: lower x bound
REGION_X_MAX    = 0.0     # emphasized region: upper x bound
REGION_WEIGHT   = 5.0     # weight multiplier inside that region

# --- Latent prior ---
PARZEN_MU    = 1e-3    # weight of the Parzen log-density prior
PARZEN_SIGMA = 0.01    # kernel bandwidth; must match the scale of the latent
                       # codes, whose norms are ~1e-2 — a bandwidth of order 1
                       # would make the prior nearly flat and hence inert

# --- Bounds on the pose (keeps the transform physically plausible) ---
LAM_MIN, LAM_MAX = 0.3, 3.0
T_ABS_MAX = 1.0

# --- Reporting ---
PRINT_EVERY = 50       # steps between progress lines
PLOT_EVERY = CLI_ARGS.inference_plot_every or None

# =============================================================================
# RECONSTRUCTION — fit a latent code and a pose to an observed wing
# =============================================================================
# Given an image of a damaged wing, finds the shape in the learned manifold
# that best explains it, together with the similarity transform placing that
# shape onto the observation. The decoder is frozen: the observation is
# explained by choosing a point in shape space, never by changing the space.
#
# Phase 1 — initialization sweep. The fit is non-convex in z, so a bad start
#   converges to a bad shape. Several diverse training codes are tried, each
#   given a short pose-only optimization, and the best is kept.
#
# Phase 2 — joint optimization of the pose and the latent code, minimizing
#     data_loss + reg_loss
#   where data_loss is the weighted occupancy mismatch between the decoded
#   shape and the observation, and reg_loss is minus the Parzen log-density of
#   z under the training latents, keeping the code in a region the decoder can
#   be trusted on.
#
# Weighting of the data term reflects what a damaged wing actually tells us:
#   inside_weight  — missing material means the observation is a subset of the
#                    true shape, so predicting material where none is observed
#                    is penalized only lightly (OUTSIDE_W), while failing to
#                    cover observed material is penalized fully.
#   root_mask      — the root region was removed during training and carries
#                    no information, so it is excluded.
#   region_weights — an x-band given extra weight (REGION_WEIGHT).
#
# Outputs are written to the caller-provided results directory:
#   <stem>_1input.png   the observation, binarized
#   <stem>_2model.png   the fitted shape, decoded and transformed
#   <stem>_4opt_result.pt   the full result dict, reloadable with torch.load
# (slot 3 is reserved for ground truth, available only when it exists.)
# =============================================================================

def make_inference_sampling_strata(binary, sdf_values, boundary_band):
    """Partition the inference grid into four observation-aware strata."""
    observed = binary[::-1, :].reshape(-1).astype(bool)
    exterior_boundary = (
        (~observed) & (np.abs(sdf_values) <= boundary_band)
    )
    near_exterior = (
        (~observed)
        & (~exterior_boundary)
        & (sdf_values > 0)
        & (sdf_values <= 0.4)
    )
    far_background = ~(
        observed | exterior_boundary | near_exterior
    )
    return (
        np.flatnonzero(observed),
        np.flatnonzero(exterior_boundary),
        np.flatnonzero(near_exterior),
        np.flatnonzero(far_background),
    )


def stratified_inference_sample(strata, sample_count, rng):
    """Sample grid indices and return inverse-probability corrections."""
    fractions = (0.40, 0.25, 0.25, 0.10)
    selected_parts = []
    correction_parts = []
    for stratum, fraction in zip(strata, fractions):
        if len(stratum) == 0:
            continue
        requested = max(1, int(round(sample_count * fraction)))
        count = min(requested, len(stratum))
        chosen = rng.choice(stratum, size=count, replace=False)
        selected_parts.append(chosen)
        correction_parts.append(
            np.full(count, len(stratum) / count, dtype=np.float32)
        )

    if not selected_parts:
        raise ValueError("All inference sampling strata are empty")

    indices = np.concatenate(selected_parts)
    corrections = np.concatenate(correction_parts)
    permutation = rng.permutation(len(indices))
    return indices[permutation], corrections[permutation]


def optimize_theta_and_g(
    y_path,
    model,
    latent_codes,
    rep_indices,
    set_grid,
    device,
    save_outputs=True,
    results_dir=None,
    eps=CHI_EPS,
    p=LOSS_P,
    outside_penalty=OUTSIDE_W,
    mu=PARZEN_MU,
    sigma=PARZEN_SIGMA,
    steps=OPT_STEPS,
    lr_g=OPT_LR_G,
    lr_z=OPT_LR_Z,
    root_x=OPT_ROOT_X,
    root_margin=OPT_ROOT_MARGIN,
    region_x_min=REGION_X_MIN,
    region_x_max=REGION_X_MAX,
    region_weight=REGION_WEIGHT,
    pre_steps=PRE_STEPS,
    print_every=PRINT_EVERY,
    plot_every=PLOT_EVERY,
    use_sampling=False,
    sample_count=65536,
    sampling_resample_every=10,
    sampling_boundary_band=0.03,
    sampling_final_dense_steps=100,
):
    if use_sampling and not 0 < sampling_final_dense_steps < steps:
        raise ValueError(
            "sampling_final_dense_steps must be between 1 and steps - 1"
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # --- Observation: image -> padded binary -> grid -> SDF ---
    # No root crop and no alignment: recovering the pose is the point.
    binary_y = pad_wing(y_path)
    grid_y   = move_image_to_grid(set_grid, binary_y)
    sdf_y    = compute_sdf(grid_y)

    coords = torch.tensor(set_grid.reshape(-1, 2), dtype=torch.float32, device=device)
    sdf_y_torch = torch.tensor(sdf_y[:, :, 2].reshape(-1),
                               dtype=torch.float32, device=device)
    chi_y = torch_smooth_heaviside(sdf_y_torch, eps=eps)

    # Observed material counts fully; unobserved regions only lightly, since
    # damage removes material and cannot add it.
    inside_weight = chi_y + outside_penalty * (1.0 - chi_y)

    Z_all = latent_codes.weight.detach().to(device)

    full_batch = {
        "coords": coords,
        "target_chi": chi_y,
        "inside_weight": inside_weight,
        "correction": None,
    }
    sampling_rng = np.random.default_rng(SEED)
    sampling_strata = None
    if use_sampling:
        sampling_strata = make_inference_sampling_strata(
            binary_y,
            sdf_y[:, :, 2].reshape(-1),
            sampling_boundary_band,
        )

    def make_sampled_batch():
        indices, corrections = stratified_inference_sample(
            sampling_strata,
            sample_count,
            sampling_rng,
        )
        index_tensor = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=device,
        )
        return {
            "coords": coords.index_select(0, index_tensor),
            "target_chi": chi_y.index_select(0, index_tensor),
            "inside_weight": inside_weight.index_select(0, index_tensor),
            "correction": torch.as_tensor(
                corrections,
                dtype=torch.float32,
                device=device,
            ),
        }

    sweep_batch = make_sampled_batch() if use_sampling else full_batch

    # --- Shared data term, used by both phases ---
    def compute_data_loss(sdf_pred, Xquery, batch):
        chi_pred = torch_smooth_heaviside(sdf_pred, eps=eps)
        root_mask = root_region_weight_from_Xquery(Xquery=Xquery, root_x=root_x,
                                                   root_margin=root_margin)
        region_weights = torch.ones_like(sdf_pred)
        x_query = Xquery[:, 0]
        region_weights[(x_query >= region_x_min) & (x_query <= region_x_max)] = region_weight

        weights = root_mask * region_weights * batch["inside_weight"]
        if batch["correction"] is not None:
            weights = weights * batch["correction"]
        point_loss = torch.abs(batch["target_chi"] - chi_pred) ** p
        return (1.0 / p) * torch.sum(weights * point_loss) / (weights.sum() + 1e-8)

    # ---------------- PHASE 1: initialization sweep ----------------
    # Each candidate code is held fixed while only the pose is optimized, so
    # candidates are ranked on how well their shape can be positioned.
    def run_short_g_optim(z_candidate):
        z_fixed  = z_candidate.clone().detach()
        _log_lam = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        _angle   = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        _t       = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)
        _optimizer = torch.optim.Adam([_log_lam, _angle, _t], lr=lr_g)

        data_loss = torch.tensor(float('inf'), device=device)
        for _ in range(pre_steps):
            _optimizer.zero_grad()
            sdf_pred, Xquery = predicted_sdf_transformed_torch(
                model=model, z=z_fixed, coords=sweep_batch["coords"],
                log_lam=_log_lam, angle=_angle, t=_t)
            data_loss = compute_data_loss(
                sdf_pred,
                Xquery,
                sweep_batch,
            )
            if torch.isnan(data_loss):
                break
            data_loss.backward()
            torch.nn.utils.clip_grad_norm_([_log_lam, _angle, _t], max_norm=1.0)
            _optimizer.step()
            with torch.no_grad():
                _log_lam.clamp_(np.log(LAM_MIN), np.log(LAM_MAX))
                _t.clamp_(-T_ABS_MAX, T_ABS_MAX)

        return data_loss.item(), _log_lam.detach(), _angle.detach(), _t.detach()

    tqdm.write(
        f"--- Coarse sweep over {len(rep_indices)} candidates "
        f"({pre_steps} steps each, "
        f"{'stratified sample' if use_sampling else 'full grid'}) ---"
    )

    best_sweep_loss = float('inf')
    best_rep_idx = None
    best_z_init = best_log_lam = best_angle = best_t = None

    sweep_progress = tqdm(
        rep_indices,
        desc="Inference initialization sweep",
        unit="candidate",
        dynamic_ncols=True,
    )
    for idx in sweep_progress:
        z_candidate = Z_all[idx].unsqueeze(0).clone()
        sweep_loss, _log_lam, _angle, _t = run_short_g_optim(z_candidate)
        sweep_progress.set_postfix(
            candidate=idx,
            loss=f"{sweep_loss:.4g}",
            refresh=True,
        )
        if sweep_loss < best_sweep_loss:
            best_sweep_loss = sweep_loss
            best_rep_idx    = idx
            best_z_init     = z_candidate
            best_log_lam    = _log_lam.clone()
            best_angle      = _angle.clone()
            best_t          = _t.clone()

    if best_rep_idx is None:
        raise RuntimeError("Sweep failed: every candidate produced NaN.")
    tqdm.write(
        f"--- Best candidate idx={best_rep_idx} "
        f"loss={best_sweep_loss:.6f} ---"
    )

    # ---------------- PHASE 2: joint optimization ----------------
    # The pose is optimized with Adam; the latent code takes a plain gradient
    # step at a much smaller rate, so the shape drifts slowly while the pose
    # adapts quickly. Both are updated from the same backward pass.
    z0      = best_z_init.clone().detach()
    z       = z0.clone().detach().requires_grad_(True)
    log_lam = best_log_lam.clone().requires_grad_(True)
    angle   = best_angle.clone().requires_grad_(True)
    t       = best_t.clone().requires_grad_(True)

    optimizer_g = torch.optim.Adam([{"params": [log_lam, angle, t], "lr": lr_g}])

    losses, data_losses, reg_losses = [], [], []
    best_loss = float('inf')
    best_snapshot = None
    best_step = None
    sampled_steps = (
        steps - sampling_final_dense_steps
        if use_sampling
        else 0
    )
    current_batch = None if use_sampling else full_batch

    inference_progress = tqdm(
        range(steps),
        desc="Inference joint optimization",
        unit="step",
        dynamic_ncols=True,
    )
    for step in inference_progress:
        sampled_stage = use_sampling and step < sampled_steps
        if sampled_stage:
            if (
                current_batch is None
                or step % sampling_resample_every == 0
            ):
                current_batch = make_sampled_batch()
            stage_name = "sampled"
        else:
            if use_sampling and step == sampled_steps:
                # Sampled and dense objectives are close approximations but
                # not numerically identical. Choose the final result only
                # among iterates scored during exact full-grid refinement.
                best_loss = float('inf')
                best_snapshot = None
                best_step = None
            current_batch = full_batch
            stage_name = "dense"

        optimizer_g.zero_grad()
        if z.grad is not None:
            z.grad.zero_()

        sdf_pred, Xquery = predicted_sdf_transformed_torch(
            model=model,
            z=z,
            coords=current_batch["coords"],
            log_lam=log_lam,
            angle=angle,
            t=t,
        )

        data_loss = compute_data_loss(
            sdf_pred,
            Xquery,
            current_batch,
        )
        reg_loss  = -mu * parzen_log_density(z, Z_all, sigma=sigma)
        loss = data_loss + reg_loss

        if torch.isnan(loss):
            raise RuntimeError(
                f"NaN during {stage_name} inference at step {step}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_lam, angle, t], max_norm=1.0)
        optimizer_g.step()

        with torch.no_grad():
            if z.grad is not None:
                z -= lr_z * z.grad                    # manual step on the code
            log_lam.clamp_(np.log(LAM_MIN), np.log(LAM_MAX))
            t.clamp_(-T_ABS_MAX, T_ABS_MAX)

        # Keep the best iterate, not the last: the trajectory is not monotone
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_step = step
            best_snapshot = {
                "loss":  loss.item(),
                "lam":   torch.exp(log_lam).detach().cpu().item(),
                "angle": angle.detach().cpu().item(),
                "t":     t.detach().cpu().numpy().copy(),
                "z":     z.detach().cpu().clone(),
            }

        losses.append(loss.item())
        data_losses.append(data_loss.item())
        reg_losses.append(reg_loss.item())

        if print_every and step % print_every == 0:
            z_change    = torch.norm(z - z0).item()
            z_grad_norm = z.grad.norm().item() if z.grad is not None else 0.0
            inference_progress.set_postfix(
                stage=stage_name,
                loss=f"{loss.item():.4g}",
                best=f"{best_loss:.4g}",
                lam=f"{torch.exp(log_lam).item():.3f}",
                refresh=True,
            )
            tqdm.write(
                f"step {step:04d} [{stage_name}] | "
                f"loss={loss.item():.6f} | "
                f"data={data_loss.item():.6f} | reg={reg_loss.item():.6f} | "
                f"z_change={z_change:.4f} | z_grad={z_grad_norm:.6f} | "
                f"lam={torch.exp(log_lam).item():.4f} | "
                f"angle={angle.item():.4f} | "
                f"t=[{t[0].item():.4f}, {t[1].item():.4f}]"
            )

        if plot_every and step % plot_every == 0:
            chi_y_grid = np.concatenate([
                set_grid[:, :, :2],
                chi_y.reshape(set_grid.shape[:2]).detach().cpu().numpy()[:, :, None]
            ], axis=-1)
            plot_current_characteristic(model=model, z=z, set_grid=set_grid,
                                        chi_y=chi_y_grid, log_lam=log_lam,
                                        angle=angle, t=t, device=device, step=step)
            plot_current_latent_pca(latent_codes=latent_codes, z=z,
                                    init_shape_idx=best_rep_idx, step=step)

    inference_progress.close()
    if best_snapshot is None:
        raise RuntimeError("Joint inference produced no valid dense iterate")

    results = {
        "loss":            best_snapshot["loss"],
        "lam":             best_snapshot["lam"],
        "angle":           best_snapshot["angle"],
        "t":               best_snapshot["t"],
        "z_optimized":     best_snapshot["z"],
        "z_initial":       z0.detach().cpu(),
        "best_step":       best_step,
        "best_sweep_loss": best_sweep_loss,
        "best_rep_idx":    best_rep_idx,
        "rep_indices":     rep_indices,
        "losses":          losses,
        "data_losses":     data_losses,
        "reg_losses":      reg_losses,
        "input_path":      y_path,
        "inference_sampling": use_sampling,
        "sampling_config": (
            {
                "sample_count": sample_count,
                "resample_every": sampling_resample_every,
                "boundary_band": sampling_boundary_band,
                "sampled_steps": sampled_steps,
                "final_dense_steps": sampling_final_dense_steps,
            }
            if use_sampling
            else None
        ),
    }

    # ---------------- OUTPUTS ----------------
    if save_outputs:
        if results_dir is None:
            results_dir = os.path.join(OUTPUT_ROOT, "reconstruction")
        os.makedirs(results_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(y_path))[0]

        save_binary_png(binary_y, os.path.join(results_dir, f"{stem}_1input.png"))

        model_binary = binary_from_transform(model, results["z_optimized"], set_grid,
                                             results["lam"], results["angle"],
                                             results["t"], device)
        save_binary_png(model_binary, os.path.join(results_dir, f"{stem}_2model.png"))

        torch.save(results, os.path.join(results_dir, f"{stem}_4opt_result.pt"))
        print(f"Saved results to {results_dir}/{stem}_*")

    return results

# =============================================================================
# BATCH RECONSTRUCTION — evaluation datasets
# =============================================================================
# Batch inference is the default. Outputs mirror the consolidated dataset
# hierarchy below OUTPUT_ROOT/reconstruction so identical specimen names in
# different damage regimes cannot overwrite one another. Existing complete
# outputs are skipped, making interrupted cluster jobs safe to resume.
# =============================================================================

def selected_damage_types(root, requested):
    available = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and any(path.glob("*.png"))
    )
    if not available:
        raise FileNotFoundError(f"No artificial-damage folders found in {root}")
    selected = available if requested is None else list(dict.fromkeys(requested))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(
            f"Damage types absent from {root}: {', '.join(missing)}"
        )
    return selected


def limited(items, limit):
    return items if limit is None else items[:limit]


def build_inference_jobs(args):
    jobs = []
    groups = set(args.inference_groups)

    if "slides-test-damaged" in groups:
        paths = limited(
            sorted(SLIDES_TEST_DAMAGED.glob("*.png")),
            args.inference_limit,
        )
        if not paths:
            raise FileNotFoundError(
                f"No damaged test wings found in {SLIDES_TEST_DAMAGED}"
            )
        jobs.extend(
            {
                "group": "slides-test-damaged",
                "damage_type": "",
                "input_path": path,
                "ground_truth_path": None,
                "relative_output_dir": Path("00_slides") / "02_test_damaged",
            }
            for path in paths
        )

    if "slides-artificial" in groups:
        filenames = limited(
            split_filenames(TEST_SPLIT),
            args.inference_limit,
        )
        for damage_type in selected_damage_types(
            SLIDES_ARTIFICIAL_DAMAGE,
            args.damage_types,
        ):
            for filename in filenames:
                jobs.append(
                    {
                        "group": "slides-artificial",
                        "damage_type": damage_type,
                        "input_path": (
                            SLIDES_ARTIFICIAL_DAMAGE / damage_type / filename
                        ),
                        "ground_truth_path": SLIDES_UNDAMAGED / filename,
                        "relative_output_dir": (
                            Path("00_slides")
                            / "01_artificial_damage"
                            / damage_type
                        ),
                    }
                )

    if "live-bees-artificial" in groups:
        filenames = limited(
            sorted(path.name for path in LIVE_BEES_UNDAMAGED.glob("*.png")),
            args.inference_limit,
        )
        if not filenames:
            raise FileNotFoundError(
                f"No undamaged live-bee wings found in {LIVE_BEES_UNDAMAGED}"
            )
        for damage_type in selected_damage_types(
            LIVE_BEES_ARTIFICIAL_DAMAGE,
            args.damage_types,
        ):
            for filename in filenames:
                jobs.append(
                    {
                        "group": "live-bees-artificial",
                        "damage_type": damage_type,
                        "input_path": (
                            LIVE_BEES_ARTIFICIAL_DAMAGE / damage_type / filename
                        ),
                        "ground_truth_path": LIVE_BEES_UNDAMAGED / filename,
                        "relative_output_dir": (
                            Path("01_live_bees")
                            / "01_artificial_damage"
                            / damage_type
                        ),
                    }
                )

    if not jobs:
        raise ValueError("No inference jobs were selected")

    missing = [
        path
        for job in jobs
        for path in (job["input_path"], job["ground_truth_path"])
        if path is not None and not path.is_file()
    ]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:20])
        remainder = len(missing) - 20
        suffix = f"\n  ... and {remainder} more" if remainder > 0 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} inference input/ground-truth files:\n"
            f"{preview}{suffix}"
        )
    return jobs


def reconstruction_output_paths(job, reconstruction_root):
    output_dir = reconstruction_root / job["relative_output_dir"]
    stem = job["input_path"].stem
    paths = {
        "input": output_dir / f"{stem}_1input.png",
        "model": output_dir / f"{stem}_2model.png",
        "result": output_dir / f"{stem}_4opt_result.pt",
    }
    if job["ground_truth_path"] is not None:
        paths["ground_truth"] = output_dir / f"{stem}_3gt.png"
    return output_dir, paths


def write_inference_summary(rows, path):
    fieldnames = [
        "group",
        "damage_type",
        "input_path",
        "decoder_compiled",
        "inference_sampling",
        "status",
        "loss",
        "best_step",
        "output_dir",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


announce_phase("Discovering batch inference inputs")
inference_jobs = build_inference_jobs(CLI_ARGS)
counts = {}
for job in inference_jobs:
    key = job["group"]
    counts[key] = counts.get(key, 0) + 1
print(
    f"Selected {len(inference_jobs)} inference inputs: "
    + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
)

announce_phase("Selecting diverse latent-code initialization candidates")
set_grid = make_square_grid()
Z_all = latent_codes.weight.detach()
rep_indices = get_cluster_representatives(Z_all, n_clusters=N_CLUSTERS)
print("Candidate latents:", rep_indices)

for parameter in model.parameters():
    parameter.requires_grad_(False)
    parameter.grad = None
model.eval()
inference_model = model
if CLI_ARGS.inference_compile:
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "Default compiled inference requires PyTorch 2.0 or newer. "
            "Pass --no-inference-compile to use eager inference."
        )
    announce_phase(
        "Creating torch.compile inference decoder "
        f"(mode={CLI_ARGS.inference_compile_mode})"
    )
    inference_model = torch.compile(
        model,
        backend="inductor",
        mode=CLI_ARGS.inference_compile_mode,
        # Dense inference has one fixed decoder shape and uses the exact
        # configuration from the earlier compile benchmark. Sampling can
        # produce different batch sizes across images, so compile it
        # dynamically to avoid repeated shape-specialized graphs.
        dynamic=CLI_ARGS.inference_sampling,
    )
else:
    print("Inference decoder: eager (--no-inference-compile)")

if CLI_ARGS.inference_sampling:
    print(
        "Inference optimization: stratified sampling "
        f"({CLI_ARGS.inference_sample_count} coordinates, "
        f"resample every {CLI_ARGS.inference_sampling_resample_every} steps, "
        f"{CLI_ARGS.inference_sampling_final_dense_steps} final dense steps)"
    )
else:
    print("Inference optimization: original full grid")

reconstruction_root = Path(OUTPUT_ROOT) / "reconstruction"
summary_rows = []
completed = skipped = failed = 0

batch_progress = tqdm(
    inference_jobs,
    desc="SliceLearn batch inference",
    unit="wing",
    dynamic_ncols=True,
)
for job in batch_progress:
    input_path = job["input_path"]
    output_dir, output_paths = reconstruction_output_paths(
        job,
        reconstruction_root,
    )
    row = {
        "group": job["group"],
        "damage_type": job["damage_type"],
        "input_path": str(input_path),
        "decoder_compiled": CLI_ARGS.inference_compile,
        "inference_sampling": CLI_ARGS.inference_sampling,
        "status": "",
        "loss": "",
        "best_step": "",
        "output_dir": str(output_dir),
        "error": "",
    }

    if (
        not CLI_ARGS.overwrite_inference
        and all(path.is_file() for path in output_paths.values())
    ):
        skipped += 1
        row["status"] = "skipped_existing"
        summary_rows.append(row)
        batch_progress.set_postfix(
            done=completed,
            skipped=skipped,
            failed=failed,
            refresh=True,
        )
        continue

    announce_phase(
        f"Inference {completed + skipped + failed + 1}/"
        f"{len(inference_jobs)}: {input_path}"
    )
    try:
        results = optimize_theta_and_g(
            y_path=str(input_path),
            model=inference_model,
            latent_codes=latent_codes,
            rep_indices=rep_indices,
            set_grid=set_grid,
            device=device,
            results_dir=str(output_dir),
            plot_every=PLOT_EVERY,
            use_sampling=CLI_ARGS.inference_sampling,
            sample_count=CLI_ARGS.inference_sample_count,
            sampling_resample_every=(
                CLI_ARGS.inference_sampling_resample_every
            ),
            sampling_boundary_band=(
                CLI_ARGS.inference_sampling_boundary_band
            ),
            sampling_final_dense_steps=(
                CLI_ARGS.inference_sampling_final_dense_steps
            ),
        )

        if job["ground_truth_path"] is not None:
            ground_truth = pad_wing(job["ground_truth_path"])
            output_dir.mkdir(parents=True, exist_ok=True)
            save_binary_png(ground_truth, output_paths["ground_truth"])

        completed += 1
        row["status"] = "completed"
        row["loss"] = results["loss"]
        row["best_step"] = results["best_step"]
        tqdm.write(
            f"Completed {input_path.name}: loss={results['loss']:.6f}, "
            f"best step={results['best_step']}, "
            f"initial latent={results['best_rep_idx']}"
        )
    except Exception as error:
        failed += 1
        row["status"] = "failed"
        row["error"] = repr(error)
        tqdm.write(f"FAILED {input_path}: {error!r}")

    summary_rows.append(row)
    batch_progress.set_postfix(
        done=completed,
        skipped=skipped,
        failed=failed,
        refresh=True,
    )

batch_progress.close()
summary_path = Path(OUTPUT_ROOT) / "inference_summary.csv"
write_inference_summary(summary_rows, summary_path)
print(
    f"\nBatch inference finished: {completed} completed, {skipped} skipped, "
    f"{failed} failed. Summary: {summary_path}"
)
if failed:
    raise RuntimeError(
        f"{failed} inference jobs failed; see {summary_path} for details."
    )
