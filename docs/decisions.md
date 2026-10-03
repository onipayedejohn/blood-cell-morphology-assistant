# Decisions

Short notes on the choices that shaped the project, and why.

## 1. Blood cells rather than another chest X-ray model

The bootcamp notebook already trains a pneumonia classifier on PneumoniaMNIST. I wanted an imaging project that uses what I actually do at the bench: reading peripheral blood films and doing differential counts. Blood cell morphology also makes the model's mistakes easy to judge, because I know which cells are genuinely hard to tell apart.

## 2. Full PBC images instead of BloodMNIST, tested at 64 and 112 px

BloodMNIST is built from the same PBC images but downsized. Granules are what separate basophils, eosinophils and neutrophils, so resolution might matter. I compared a 64 px model with a 112 px model on the validation set and fixed the rule before looking: keep 112 px only if it beats 64 px by more than 0.005 macro-F1, otherwise keep the smaller, faster model.

The original Mendeley archive, Zenodo (MedMNIST) and Hugging Face were all blocked from the build environment. The 13,344-image GitHub copy is a CC BY 4.0 derivative of the same data, with the labels stored in box files. Platelets are not in that copy, so the model has seven classes, not eight.

## 3. My own split, grouped by duplicates

The source split did not account for duplicates: I found 14 pairs of identical images, and 8 of those pairs had one copy in training and the other in validation or test. I made a new stratified split in which every pair stays in one split. The test set (20%) was split off first and used only for final scoring (twice; see section 7).

## 4. Training from scratch, on CPU

Pre-trained ImageNet weights could not be downloaded either, and the machine has two CPU cores. A small four-block CNN (0.7 to 1.2 million parameters) trains in 40 to 76 minutes and is small enough to deploy for free. Augmentation matches the domain: cells have no fixed orientation (flips and 90° rotations), and stain intensity varies between slides (small brightness, contrast, saturation and hue shifts).

## 5. Precise batch norm

The first run looked broken: training loss fell, but validation macro-F1 was 0.02 for three epochs and 0.10 at epoch 4. The weights were fine. With the network in training mode the same checkpoint scored 91% accuracy on 512 validation images. The batch-norm running averages had drifted away from the statistics of clean images, because they were collected on augmented batches while the weights moved quickly. Recomputing them on 2,048 clean training images before each validation check ("precise BN", Wu and Johnson, *Rethinking "Batch" in BatchNorm*, 2021) moved validation macro-F1 at epoch 4 from 0.10 to 0.96. Checkpoints are saved after that step, so the exported model carries the corrected statistics. The figures in this section come from my debugging session; the first run's log and checkpoint were overwritten when I retrained.

## 6. A heatmap that needs no gradients

The network ends in global average pooling and one dense layer. That makes a class activation map (Zhou et al., CVPR 2016) exact and cheap: the last feature maps weighted by the dense layer's weights for the chosen class. The ONNX model returns the feature maps as a second output, so the app draws the map with numpy and does not need TensorFlow or a gradient library.

## 7. Calibrated confidence and a status line

A softmax score is not automatically a probability, so I fitted temperature scaling on validation logits. Here the model was already well calibrated: T came out at 0.92, and test ECE moved from 0.42% to 0.47%. The status line combines two checks:

- **Confidence.** Below 81%, the level that sends the least confident 2% of validation cells for review, the app says to check the cell yourself. My first rule was "the lowest level, from 40% up, at which accepted validation cells are at least 99% correct". It settled at 40%, held back nothing on the test set, and left 42 of 44 test errors marked Confident. I replaced it after that first test run, which means the status-line figures are not a fully clean held-out estimate.
- **Unfamiliar image.** The Mahalanobis distance of the image's pooled features from the nearest class centre (shared covariance with Ledoit–Wolf shrinkage, fitted on training features). Above the 99th percentile of validation distances, the app warns that the image does not look like its training data. This catches images a confident softmax would happily mislabel, such as a field of red cells or random noise.

Both levels were set on validation data. Together they caught 25 of the 44 test errors, and 19 errors still passed as Confident.

## 8. The differential is reported the way a lab reports it

Percentages are over white cells only. Erythroblasts are nucleated red cells, so they are reported per 100 white cells. Cells that fail either check are held back for review instead of being counted. The app warns when fewer than 100 white cells were counted, because a manual differential counts at least 100.

## 9. ONNX Runtime in the app

Training uses TensorFlow and Keras, as in the bootcamp. The app runs the ONNX export instead: the runtime is small, starts quickly on Streamlit Community Cloud and is faster per image. `scripts/evaluate.py` checks that the export gives the same top class as Keras on test images before saving it.

## 10. Honest external check

BCCD has no white cell subtype labels, so I cannot report accuracy on another lab's images. I report only whether the unfamiliar-image check fires on them, and say in the app that performance on other labs is unknown and probably much lower.
