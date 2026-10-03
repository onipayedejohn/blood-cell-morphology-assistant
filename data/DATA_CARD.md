# Data card

## Training data: PBC (peripheral blood cells)

| | |
|---|---|
| Source | Acevedo A, Merino A, Alférez S, Molina Á, Boldú L, Rodellar J. *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems.* Data in Brief 30 (2020) 105474. https://doi.org/10.1016/j.dib.2020.105474 |
| Original archive | Mendeley Data, https://doi.org/10.17632/snkd93bnjr.1 (17,092 images, 8 classes) |
| Version used here | The 13,344-image copy with YOLO cell boxes published at https://github.com/medmabcf/White-Blood-Cell-Detection-Dataset. The original archive and the MedMNIST mirrors could not be downloaded from the build environment. |
| Licence | CC BY 4.0 |
| Capture | CellaVision DM96 automated analyser, Hospital Clinic of Barcelona, May–Grünwald–Giemsa stain, 360 x 363 JPEG |
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
- **One site and one analyser.** Stain, optics and camera are the same for every image.
- **Normal donors only.** No blasts, atypical lymphocytes, parasites or abnormal red cells.
- **Class balance is not a real differential.** Eosinophils and immature granulocytes are heavily over-represented compared with normal blood.

## External images: BCCD

| | |
|---|---|
| Source | BCCD dataset, https://github.com/Shenggan/BCCD_Dataset (MIT licence) |
| Content | 364 smear photos, 640 x 480, with boxes for white cells, red cells and platelets. No white cell subtype labels. |
| Used for | `data/external/bccd_wbc`: 358 white cell crops (one per photo, box plus 20% margin). `data/external/bccd_rbc_only`: 150 patches of red cells away from any white cell. |
| Purpose | Only to test whether the app notices that an image comes from somewhere else. Accuracy cannot be measured without subtype labels. |

## Files in this folder

| Path | What it is | In git |
|---|---|---|
| `manifest.csv` | Every PBC image with class, subtype, split, duplicate group and pixel hash | yes |
| `samples/` | 21 test-set cells (3 per class) and 4 questionable images for the app | yes |
| `demo_batch/` | 120 test-set cells for the differential demo, with `demo_batch_labels.csv` | yes |
| `external/` | BCCD crops described above | yes |
| `raw/` | The downloads (see README for commands) | no |
| `processed/` | 64 px and 112 px arrays rebuilt by `scripts/prepare_data.py` | no |
