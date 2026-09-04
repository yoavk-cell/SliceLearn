# SliceLearn

**Learning Similarity-Invariant Shape Manifolds for Wing Damage Estimation**

Yoav Kamir¹, Roberta Hunt¹˒², Charlie Nicholson³, François Lauze¹
¹ University of Copenhagen · ² Lund University · ³ Roger Williams University

Computer Vision for Natural History (CVNH) Workshop, ECCV 2026, Malmö, Sweden.

[Trained models & data (Zenodo)](https://doi.org/10.5281/zenodo.21819432) · [TransCNN-HAE baseline fork](https://github.com/robertahunt/TransCNN-HAE)

---

## Method

SliceLearn estimates wing-area loss from a single image of a damaged wing. The intact wing is never observed, so the missing area is inferred by reconstructing a plausible complete shape, while separating biologically meaningful variation from translation, rotation, and scale.

Two stages:

1. **Canonical normalization.** Each intact wing is mapped to a unique similarity-normalized representative using its centroid, geometric covariance, and gyration radius. Motivated by slice theory: locally, the space of planar domains factors into an intrinsic shape component and the action of the similarity group Sim(2). Individual left and right wings are asymmetric, so a unique orientation exists.

2. **Implicit shape prior.** The family of normalized intact wings is learned with a modified DeepSDF auto-decoder using SIREN activations. Each wing is the zero level set of a continuous field conditioned on a 16-dimensional latent code.

At inference the decoder is frozen and used as a prior: the latent code `z` and the pose `g = (λ, R, t)` are optimized jointly against the observed contour. The reconstructed intact shape gives a direct estimate of the missing area. The representation is continuous and differentiable, so it also supports geometric measurements beyond pixel-level reconstruction.

The method is not wing-specific. It applies to any collection of asymmetric planar shapes where partial observations must be reconstructed in a pose-invariant way.

## Contents

| Path | Description |
|---|---|
| `scripts/slicelearn_code.py` | SliceLearn. Training and batch inference over the evaluation datasets; produced the reported results. |
| `scripts/average_reference_wing.py` | Builds the average reference wing from normalized training masks. Required by the registration baselines. |
| `scripts/ants_average_reference_baseline.py` | ANTsPy registration baseline. Fits a similarity transform from the average reference to a damaged mask. |
| `scripts/run_registration_datasets.py` | Mask registration over the slide, live-bee, and naturally damaged datasets. |
| `scripts/run_registration_test50.py` | Entry point for the above with default settings. |
| `scripts/run_segmentation_registration_test50.py` | Textured-reference variant, registering on segmentation luminance with masked Mattes. |
| `scripts/scale_normalize_slicelearn_datasets.py` | Rescales live-bee images to the slide dataset's pixels-per-centimetre using each image's scale bar. |
| `scripts/analyze_model_pixel_metrics.py` | Scores model predictions against ground truth and writes the metrics CSV behind the reported tables. |

The TransCNN-HAE inpainting baseline is in a separate repository: [robertahunt/TransCNN-HAE](https://github.com/robertahunt/TransCNN-HAE).

## Requirements

`torch`, `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `pandas`, `matplotlib`, `pillow`, `opencv-python`, `pyyaml`, `tqdm`. The registration baselines additionally require `antspyx`.

A GPU is recommended for training and effectively required for batch inference. Inputs are PNGs of segmented wings on a white background, fitting within a 1001 × 1001 canvas.

Every script accepts `--help`. Reported results are averaged over seeds 0, 1, and 2.

## Datasets

Both datasets are *Bombus terrestris* forewings.

**Slide dataset** — training, validation, and testing. Forewings from commercial colonies (N = 20) collected May–August 2024, mounted on microscope slides and scanned on an Epson Perfection V39 at 618 pixels/cm, with masks obtained via blob detection and SAM2. 150 undamaged pairs split 200 train / 50 validation / 50 test, keeping pairs from the same individual together.

**Live bee dataset** — generalization testing only. Bees photographed alive and returned to the hive, each wing immobilized against a white background with a 5 mm scale bar, using a Canon EOS M50 Mark II with a 28 mm macro lens, then rescaled to 618 pixels/cm. 12 high-visibility wings.

Test wings are damaged artificially in nine controlled ways — three styles (vertical crop, angled crop, chunks) at three severities — so the removed area is known exactly. Nine naturally damaged wings are also reconstructed for qualitative assessment.

Annotation data and trained models: [10.5281/zenodo.21819432](https://doi.org/10.5281/zenodo.21819432)

## Citation

```bibtex
@inproceedings{kamir2026slicelearn,
  title     = {Learning Similarity-Invariant Shape Manifolds for Wing Damage Estimation},
  author    = {Kamir, Yoav and Hunt, Roberta and Nicholson, Charlie and Lauze, Fran{\c c}ois},
  booktitle = {Computer Vision for Natural History (CVNH) Workshop,
               European Conference on Computer Vision (ECCV)},
  year      = {2026},
  address   = {Malm{\"o}, Sweden}
}
```

Trained models and annotation data:

```bibtex
@dataset{kamir2026slicelearn_data,
  title     = {SliceLearn: trained models and annotation data},
  author    = {Kamir, Yoav and Hunt, Roberta and Nicholson, Charlie and Lauze, Fran{\c c}ois},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21819432},
  url       = {https://doi.org/10.5281/zenodo.21819432}
}
```

## Acknowledgments

Thanks to the organizers and participants of ACCESS (Computational Entomology Summer School) 2024 for creating a melting pot for interdisciplinary work, which provided the meeting point that led to this project.

Builds on DeepSDF, SIREN, SAM 2, [TransCNN-HAE](https://github.com/robertahunt/TransCNN-HAE), and ANTsPy.
