# Model card: blood cell classifier

## Summary

| | |
|---|---|
| Task | Classify one cropped cell from a Romanowsky-stained peripheral blood smear into 7 types |
| Classes | basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil |
| Model | Convolutional network trained from scratch: 4 blocks of two 3x3 convolutions with batch norm, max pooling, global average pooling, one dense layer. 1,176,935 parameters. |
| Input | Center square of the image, resized to 64 x 64 RGB with Lanczos filtering, scaled to [0, 1] |
| Outputs | 7 logits and the last feature maps (8 x 8 x 256). The app turns these into calibrated probabilities, a class activation map and an unfamiliar-image score. |
| Format | ONNX (opset 17), 4.7 MB, run with ONNX Runtime on CPU |
| Intended use | Teaching and research on how image models behave on blood smears |
| Not for | Patient care, screening, or any decision about a real patient |

## Training

- **Data:** PBC dataset (Acevedo et al., 2020, CC BY 4.0), 9,340 training and 1,335 validation images. See `data/DATA_CARD.md`.
- **Augmentation:** random flips, 90° rotations, and small brightness, contrast, saturation and hue changes.
- **Optimizer:** AdamW (learning rate 2e-3, cosine decay, weight decay 1e-4), batch 64, 25 epochs, class weights inversely proportional to class frequency.
- **Batch-norm statistics** are recomputed on 2,048 clean training images before each validation check (precise BN). In the first run, without this step, inference-mode validation macro-F1 was 0.10 at epoch 4, and the same weights scored 0.96 after recomputing. Those two figures come from my debugging session; that run's log and checkpoint were overwritten.
- **Checkpoint:** the epoch with the best validation macro-F1 (0.9915, epoch 24).
- **Cost:** 40 minutes on 2 CPU cores, about 10.8 Wh and 5.2 g CO2e at Ghana's grid intensity (CodeCarbon estimate).

## Model selection (validation set)

| Model | Validation macro-F1 | Parameters | Training time | Training energy |
|---|---|---|---|---|
| Logistic regression on 24 x 24 pixels | 0.779 | 12,103 | 0.3 min | 0.07 Wh |
| **CNN, 64 x 64 input (chosen)** | **0.992** | 1,176,935 | 40 min | 10.8 Wh |
| CNN, 112 x 112 input | 0.991 | 663,247 | 76 min | 20.8 Wh |

The rule, set before training: keep 112 px only if it beats 64 px by more than 0.005 macro-F1. It did not. The 64 px model needs about 40% less computation per image (211 vs 366 million multiply-adds).

## Test results (2,669 images)

The test set was used only for final scoring. It was scored twice, before and after the confidence-rule change described below; the model and the metrics in this section are the same in both runs.

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Accuracy | 98.4% | 97.8% to 98.8% |
| Balanced accuracy | 98.5% | 98.0% to 99.0% |
| Macro-F1 | 98.4% | 97.9% to 98.9% |
| Expected calibration error | 0.42% before, 0.47% after temperature scaling (T = 0.92) | |

| Class | Precision | Recall | Test cells |
|---|---|---|---|
| Basophil | 98.6% | 99.1% | 214 |
| Eosinophil | 100% | 100% | 587 |
| Erythroblast | 97.9% | 98.9% | 185 |
| Immature granulocyte | 95.6% | 97.6% | 538 |
| Lymphocyte | 98.7% | 99.1% | 232 |
| Monocyte | 97.8% | 98.2% | 272 |
| Neutrophil | 99.4% | 96.9% | 641 |

**Where the errors are.** 18 of the 44 test errors are neutrophils read as immature granulocytes, 16 of them band forms (band neutrophil recall 94.7%). That is the boundary between band neutrophils and metamyelocytes, which sit next to each other in maturation and on which human readers also disagree. Promyelocytes are sometimes read as monocytes (96.7% correct).

**Calibration.** The model was already well calibrated. Temperature scaling (T = 0.92) slightly sharpened the scores and nudged ECE from 0.42% to 0.47%. Most test cells receive confidence above 95%. The few mid-range bins hold around 20 cells each and are overconfident, so a mid-range confidence should be read loosely.

## The status line

| Check | How it was set | Test cells held back |
|---|---|---|
| Confidence at least 81% | Sends the least confident 2% of validation cells for review | 70 |
| Unfamiliar image | Mahalanobis distance of pooled features from the nearest class center, warning above the 99th percentile of validation distances | 30 |

Together they held back 98 of 2,669 test cells (3.7%; 2 cells failed both checks) and caught 25 of the 44 errors. Accepted cells were 99.3% correct. **19 errors were still marked Confident, 10 of them above 90% confidence.**

The first confidence rule was "the lowest level, searching up from 40%, at which accepted validation cells are at least 99% correct". It settled at 40%, held back no test cells, and left 42 of 44 errors marked Confident. I replaced it with the review budget after that first test run, so the status-line figures are not a fully clean held-out estimate. The model, temperature and unfamiliar-image level did not change.

## Images from elsewhere

| Images | Flagged as unfamiliar |
|---|---|
| PBC test cells (training lab) | 1.1% of 2,669 |
| BCCD white cells (another lab, stain and camera) | 99.7% of 358 |
| BCCD red cells only, no white cell | 100% of 150 |
| Random noise | 100% of 100 |
| Blank fields | 100% of 50 |

Distances separate PBC test cells from BCCD white cells with an AUROC of 0.999. Without the check, the model would have called 302 of the 358 BCCD white cells immature granulocytes and 50 of them erythroblasts, with a median confidence of 99.9%. On a real smear that would read as a left shift. BCCD has no subtype labels, so accuracy on it cannot be measured, but it is clearly poor. **This model does not transfer to other labs without retraining.**

## Limitations

- One site and one automated microscope (CellaVision DM96). Other stains, optics, cameras and magnifications are out of scope.
- Donors without infection or blood disease. Blasts, atypical lymphocytes, parasites and abnormal red cells are not classes.
- No patient identifiers, so test cells may share donors with training cells.
- The class activation map is only 8 x 8, so it shows roughly where the evidence is, not fine structure.
- Platelets are not included.
- One training pair has identical pixels but conflicting labels (band neutrophil and eosinophil). See the data card.

## Export check

On 64 test images the ONNX export gave the same top class as Keras every time, with a largest logit difference of 5.7e-6. ONNX Runtime takes 1.9 ms per cell against 24.7 ms for TensorFlow on the same CPU, and 55 ms for a batch of 32 (timings vary a little between runs).
