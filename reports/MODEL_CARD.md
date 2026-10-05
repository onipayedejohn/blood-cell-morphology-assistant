# Model card: blood cell morphology assistant, version 2

Two models run in sequence: a **blood cell check** that decides whether an image is a stained white cell at all, and a **classifier** that names the cell type. Version 1's card is kept in the git history and its outputs in `models/v1` and `reports/metrics/v1`.

## Summary

| | Blood cell check | Classifier |
|---|---|---|
| Task | Sort an image into white cell, smear with no white cell, or not a smear | Classify one white cell into 7 types |
| Classes | white_cell, smear_no_wbc, not_smear | basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil |
| Model | CNN, 4 blocks of two 3x3 convolutions (16 to 128 channels), 295,347 parameters | CNN, 4 blocks of two 3x3 convolutions (32 to 256 channels), 1,176,935 parameters |
| Input | Center square of the image as uploaded, 64 x 64 | The cell in standard framing (`src/bloodsmear/canonical.py`): recentered on the nucleus, cropped to 2.4 times its size, background color divided out, 64 x 64 |
| Format | ONNX, 1.2 MB | ONNX, 4.7 MB, also returns 8 x 8 x 256 feature maps for the heatmap and the unfamiliar-image check |
| Speed (1 image, CPU) | 1 to 2 ms | about 3 ms |
| Intended use | Teaching and research on how image models behave on blood smears | |
| Not for | Patient care, screening, or any decision about a real patient | |

## Training data

See `data/DATA_CARD.md` for sources, licences and counts.

- **Classifier:** PBC training split (9,340 cells, one hospital in Barcelona, CellaVision DM96), plus 75% of the training half of each of three other sources: BCCD (131 cells), Jiangxi Tecom (112) and CellaVision blog images (about 36). Each external cell is repeated 10 times per epoch. Validation: PBC validation split (1,335) and the remaining quarter of each external training half.
- **Blood cell check:** 3,000 PBC training cells and the external training halves (repeated 6 times) as white cells; 1,866 smear patches without a white cell; 5,538 not-smear images (CIFAR-10 photos and generated paper, documents, screens, gradients, noise and textures). 15% to 20% of each group is kept for validation.

## Training

- **Classifier:** started from the version 1 weights (PBC only). AdamW, learning rate 1e-3 with cosine decay, weight decay 1e-4, batch 64, 10 epochs, class weights inversely proportional to frequency. Precise batch norm before each validation check. The checkpoint with the best score (half PBC validation macro-F1, half external validation accuracy) was kept: epoch 10. 34.6 minutes on 2 CPU cores, about 9.5 Wh (CodeCarbon estimate).
- **Augmentation (classifier):** for 75% of images, stain jitter in color-deconvolution space (Ruifrok and Johnston 2001), white balance, brightness, contrast, saturation, hue and gamma, free rotation, zoom 0.7 to 1.35, uneven lighting, blur, lower resolution, sensor noise, JPEG, and flat gray borders like the padding tight crops get. The other 25% are only flipped and rotated.
- **Blood cell check:** from scratch, 10 epochs, the same augmentation without the borders. About 12 minutes.

## Choices made on validation data

| Choice | Rule | Value |
|---|---|---|
| Temperature | Minimize negative log-likelihood on validation logits | T = 0.81 |
| Review level | Send the least confident 2% of validation cells for review | 51% confidence |
| Unfamiliar-image level | 99th percentile of validation Mahalanobis distances | 28.3 |
| Blood cell check level | Lowest level at which at most 0.5% of validation images without a white cell pass, clamped to 0.3 to 0.9 | 0.3 (the floor) |

The blood cell check's rule was changed once. The first rule (let 99% of validation cells through) hit its 0.9 ceiling and refused 26% of cells from a lab the check had never seen (`reports/metrics/gate_first_rule_lolo_bccd.json`).

## Results

### Labs the classifier never saw

Separate models, each trained with one lab left out completely.

| Lab | Cells | Accuracy (95% CI) | Balanced accuracy | Answers at 90%+ confidence: share, accuracy |
|---|---|---|---|---|
| BCCD | 351 | 50.7% (45.3% to 55.8%) | 49.1% | 24%, 68% |
| Jiangxi Tecom | 300 | 34.0% (29.0% to 39.3%) | 35.7% | 39%, 51% |

Version 1 scored 0.0% and 3.3% on the same cells. The BCCD figure is slightly optimistic, because three set-ups were compared on it (`reports/metrics/v2_experiments.json`). Neutrophils read as eosinophils is the commonest error on both labs (122 BCCD and 94 Jiangxi Tecom cells). Answers the status line marked Confident were right only 55% (BCCD) and 25% (Jiangxi Tecom) of the time.

### Adapting to a lab

`scripts/finetune_local.py`, starting from the model that never saw the lab, trained on part of that lab's cells and scored on the rest:

| Lab | Trained on | Scored on | Before | After | Balanced accuracy after |
|---|---|---|---|---|---|
| Jiangxi Tecom | 84 | 44 | 29.5% | 86.4% | 81.6% |
| BCCD | 99 | 52 | 42.3% | 80.8% | 66.0% |

Most of the gain is in neutrophils. Rare types stayed weak, and BCCD eosinophil recall fell from 77% to 54%.

### Final classifier on held-out test cells

| Test cells | n | Accuracy (95% CI) | Balanced accuracy | Macro-F1 |
|---|---|---|---|---|
| PBC (training lab) | 2,669 | 93.7% (92.8% to 94.6%) | 94.3% | 92.9% |
| BCCD test half | 177 | 84.7% (79.7% to 89.8%) | 68.5% | |
| Jiangxi Tecom test half | 151 | 88.7% (83.4% to 94.0%) | 63.7% | |
| CellaVision blog test half | 52 | 61.5% (48.1% to 75.0%) | 50.6% | |

Other cells from the three external sources were in training, so their rows describe adapted labs, not new ones. Version 1 scored 98.4% on the PBC test set; version 2 gives up 4.6 points there.

| PBC class | Precision | Recall | Test cells |
|---|---|---|---|
| Basophil | 87.6% | 99.1% | 214 |
| Eosinophil | 100% | 98.5% | 587 |
| Erythroblast | 87.6% | 95.1% | 185 |
| Immature granulocyte | 90.9% | 87.2% | 538 |
| Lymphocyte | 90.5% | 94.4% | 232 |
| Monocyte | 86.1% | 93.0% | 272 |
| Neutrophil | 99.7% | 92.7% | 641 |

Calibration error on the PBC test set is 1.5%. The status line held back 67 PBC test cells and caught 31 of the 168 errors; cells marked Confident were right 94.7% of the time.

### Blood cell check

| Images | n | Passed |
|---|---|---|
| White cells, PBC test | 2,669 | 100% |
| White cells, external test halves | 380 | 100% |
| White cells, BCCD, from a check trained without BCCD | 351 | 93.4% |
| Smear patches, no white cell | 888 | 0.6% |
| Everyday photos (unseen CIFAR-10) | 1,000 | 0.1% |
| Paper, documents, screens, noise (new seed) | 600 | 0.2% |
| A pattern type never used in training | 200 | 2.0% |
| scikit-image photos and pages | 19 | 0% |

## Export check

On 64 test images the classifier's ONNX export gave the same top class as Keras every time, with a largest logit difference of 4e-6. For the blood cell check the largest logit difference was 2e-6.

## Limitations

- On a new lab, accuracy is low and confidence does not fix it. Adapt the model first.
- Erythroblasts and immature granulocytes were learned from one lab only.
- Blasts, atypical lymphocytes, parasites and abnormal red cells are not classes.
- Phone photos through an eyepiece were only simulated.
- No patient identifiers in PBC, so test cells may share donors with training cells.
- External labels were used as published and not re-read.
- The heatmap is 8 x 8, so it shows roughly where the evidence is.
