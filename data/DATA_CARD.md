# Data card

## Training data: PBC (peripheral blood cells)

| | |
|---|---|
| Source | Acevedo A, Merino A, Alférez S, Molina Á, Boldú L, Rodellar J. *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems.* Data in Brief 30 (2020) 105474. https://doi.org/10.1016/j.dib.2020.105474 |
| Original archive | Mendeley Data, https://doi.org/10.17632/snkd93bnjr.1 (17,092 images, 8 classes) |
| Version used here | The 13,344-image copy with YOLO cell boxes published at https://github.com/medmabcf/White-Blood-Cell-Detection-Dataset. The original archive and the MedMNIST mirrors could not be downloaded from the build environment. |
| Licence | CC BY 4.0 |
| Capture | CellaVision DM96 automated analyzer, Hospital Clinic of Barcelona, May–Grünwald–Giemsa stain, 360 x 363 JPEG |
| Donors | Individuals without infection, blood disease or cancer, and not on drug treatment at the time of sampling (per the source paper) |
| Classes used | 7: basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil. Platelets are not in this copy. |

### What I checked

- Every image has exactly one box, and every box label agrees with its filename prefix (BA, EO, ERB, PMY, MY, MMY, IG, LY, MO, SNE, BNE, NEUTROPHIL). `scripts/prepare_data.py` stops if either check fails.
- **14 pairs of identical images** (identical pixels in the center square that the model sees) were found. 8 of those pairs crossed the source copy's own train/val/test split. Each pair is now kept inside one split, so no image appears in both training and test data. A perceptual-hash search found no further near-duplicates at 6 bits or fewer out of 256.
- **One pair has conflicting labels:** `BNE_191112.jpg` (band neutrophil) and `EO_225902.jpg` (eosinophil) are the same image. Both are in the training set. The cell has two lobes and orange-pink granules, so it reads as an eosinophil, and the band neutrophil label is probably the error. It is 1 of 9,340 training images and was left in place.
- The finer labels in the filenames are kept as `subtype` (for example band vs segmented neutrophil, and myelocyte vs metamyelocyte vs promyelocyte) for error analysis. The model is trained on the 7 classes.

### Class and split counts

| Class | Train | Validation | Test | Total |
|---|---|---|---|---|
| Basophil | 752 | 107 | 214 | 1,073 |
| Eosinophil | 2,053 | 293 | 587 | 2,933 |
| Erythroblast | 645 | 92 | 185 | 922 |
| Immature granulocyte | 1,884 | 270 | 538 | 2,692 |
| Lymphocyte | 813 | 116 | 232 | 1,161 |
| Monocyte | 950 | 136 | 272 | 1,358 |
| Neutrophil | 2,243 | 321 | 641 | 3,205 |
| **Total** | **9,340** | **1,335** | **2,669** | **13,344** |

The split is stratified by class and grouped by duplicate group (`StratifiedGroupKFold`, seed 42): 20% test first, then one eighth of the rest for validation. The full assignment is in `data/manifest.csv`.

### Known gaps

- **No patient identifiers.** Cells from the same person can fall in both training and test sets, which makes test results optimistic compared with new patients.
- **One site and one analyzer.** Stain, optics and camera are the same for every image.
- **Normal donors only.** No blasts, atypical lymphocytes, parasites or abnormal red cells.
- **Class balance is not a real differential.** Eosinophils and immature granulocytes are heavily over-represented compared with normal blood.

## Version 1 external check: BCCD fields

| | |
|---|---|
| Source | BCCD dataset, https://github.com/Shenggan/BCCD_Dataset (MIT licence) |
| Content | 364 smear photos, 640 x 480, with boxes for white cells, red cells and platelets. |
| Used for | `data/external/bccd_wbc`: 358 white cell crops (one per photo, box plus 20% margin). `data/external/bccd_rbc_only`: 150 patches of red cells away from any white cell. Version 1 used these only to test whether the app notices an image from elsewhere. |

## Version 2: labeled cells from other labs

Version 1 was trained on one lab and failed on all others, so version 2 adds labeled white cells from three other sources.

| Source | Cells | What it is | Licence |
|---|---|---|---|
| BCCD, cleaned crops (https://github.com/apple2373/BCCD) | 351 | White cells cut from the BCCD photos above, with a subtype label each. | The repository states no licence of its own; the underlying BCCD photos are MIT. |
| Zheng et al. Dataset 1 (https://github.com/zxaoyou/segmentation_WBC) | 300 | 120 x 120 images from Jiangxi Tecom Science Corporation, China: Motic camera, and a rapid white cell stain that is not a standard Romanowsky stain. Yellow background. | GPL-3.0 |
| Zheng et al. Dataset 2 (same repository) | 100 | 300 x 300 images collected from the CellaVision blog. | GPL-3.0 (images originally from the CellaVision blog) |

Class counts (train half / test half):

| | Basophil | Eosinophil | Lymphocyte | Monocyte | Neutrophil |
|---|---|---|---|---|---|
| BCCD | 1 / 2 | 44 / 44 | 16 / 17 | 10 / 10 | 103 / 104 |
| Jiangxi Tecom | 0 / 1 | 11 / 11 | 26 / 27 | 24 / 24 | 88 / 88 |
| CellaVision blog | 1 / 2 | 6 / 6 | 18 / 19 | 7 / 8 | 16 / 17 |

- **Five classes only.** None of these sources has erythroblasts or immature granulocytes, so on other labs those two classes are never correct answers.
- **Labels not checked by me.** I used the labels as published. Some BCCD labels are known to be noisy, and I did not re-read the cells.
- **Split.** Each source is split 50/50 into train and test halves, stratified by class (`data/manifest_external.csv`). The final model trains on the train halves and is scored on the test halves. The leave-one-lab-out models never see any cell from the lab they are scored on.
- **Replay set.** `models/train/replay_64c.npz` holds 1,330 PBC images (1,050 training, 280 validation, 190 per class) in the standard framing at 64 px, so `scripts/finetune_local.py` can run without the raw download. They are redistributed under PBC's CC BY 4.0 licence with the citation above.
- **Not redistributed.** The GPL-licensed Zheng images are not copied into this repository; `scripts/prepare_external.py` reads them from a local clone. Only BCCD cells appear in `data/samples`.

## Version 2: images for the blood cell check

| Set | Source | Count |
|---|---|---|
| Smear patches with no white cell | Cut from BCCD photos, draaslan detection images (https://github.com/draaslan/blood-cell-detection-dataset, MIT) and PBC image corners, away from every white cell box | 1,866 train, 888 test |
| Not a blood smear (training) | CIFAR-10 photos (Krizhevsky 2009, via https://github.com/YoongiKim/CIFAR-10-images) and 3,000 generated images: paper sheets in many colors, printed documents, screenshots, gradients, noise and textures | 5,538 |
| Not a blood smear (held-out tests) | Other CIFAR-10 photos (1,000), the same generator with a new seed (600), a pattern type never used in training (200), and 19 bundled scikit-image photos | 1,819 |

Whole fields with white cell boxes (BCCD and draaslan, 464 images, split 50/50 in `data/fields.json`) test the cell finder. The app's field samples come from the test half, and none of their cells were used to train or tune the classifier.

## Files in this folder

| Path | What it is | In git |
|---|---|---|
| `manifest.csv` | Every PBC image with class, subtype, split, duplicate group and pixel hash | yes |
| `samples/` | 21 PBC test-set cells (3 per class); `samples/other/`: 4 BCCD test-half cells, 3 whole fields from the test half, and 5 images that are not white cells | yes |
| `demo_batch/` | 120 test-set cells for the differential demo, with `demo_batch_labels.csv` | yes |
| `external/` | BCCD crops used by the version 1 check | yes |
| `manifest_external.csv` | Every labeled cell from the other labs, with lab, class and split | yes |
| `fields.json` | Whole-field images with white cell boxes, for the cell finder | yes |
| `raw/` | The downloads (see README for commands) | no |
| `processed/` | Arrays rebuilt by `prepare_data.py`, `prepare_external.py` and `prepare_canonical.py` | no |
