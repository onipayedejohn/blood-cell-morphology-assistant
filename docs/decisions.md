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
- **Unfamiliar image.** The Mahalanobis distance of the image's pooled features from the nearest class center (shared covariance with Ledoit–Wolf shrinkage, fitted on training features). Above the 99th percentile of validation distances, the app warns that the image does not look like its training data. This catches images a confident softmax would happily mislabel, such as a field of red cells or random noise.

Both levels were set on validation data. Together they caught 25 of the 44 test errors, and 19 errors still passed as Confident.

## 8. The differential is reported the way a lab reports it

Percentages are over white cells only. Erythroblasts are nucleated red cells, so they are reported per 100 white cells. Cells that fail either check are held back for review instead of being counted. The app warns when fewer than 100 white cells were counted, because a manual differential counts at least 100.

## 9. ONNX Runtime in the app

Training uses TensorFlow and Keras, as in the bootcamp. The app runs the ONNX export instead: the runtime is small, starts quickly on Streamlit Community Cloud and is faster per image. `scripts/evaluate.py` checks that the export gives the same top class as Keras on test images before saving it.

## 10. Honest external check

BCCD has no white cell subtype labels, so I cannot report accuracy on another lab's images. I report only whether the unfamiliar-image check fires on them, and say in the app that performance on other labs is unknown and probably much lower.

# Version 2

## 11. Why version 1 called a black sheet an erythroblast

Two separate faults. The app showed the class even when the unfamiliar-image check fired, so a warning sat next to a confident-looking answer. And a classifier only knows its seven classes, so every input becomes one of them. Version 1 also failed on real cells from other labs: tested on labeled cells from BCCD, Jiangxi Tecom and the CellaVision blog, it was right 0%, 3% and 30% of the time.

## 12. A separate blood cell check, and no class when it refuses

The fix for the first fault is a second model whose only job is to say whether the image is a white cell, a smear without one, or not a smear. When it refuses, the app shows no class at all. A "classify anyway" button is kept behind an expander for people who know better, with a warning that the answer means nothing for images that are not blood cells.

The check is trained on many kinds of "not a smear" (CIFAR-10 photos and generated sheets, documents, screenshots, gradients, noise and textures) and tested on kinds it never saw: other CIFAR photos, a new generator seed, one pattern type left out of training, and 19 scikit-image photos.

Its acceptance level was first set from the cell side (let 99% of validation cells through). That rule hit its 0.9 ceiling and refused 26% of cells from a lab the check had never seen. I replaced it with a rule set from the other side: the lowest level at which at most 0.5% of validation images without a white cell pass. I made this change after seeing the unseen-lab result.

## 13. Training on four sources, judged on the ones left out

The only honest estimate for a new hospital is a lab the model has never seen. I trained two extra classifiers with one lab left out completely (BCCD, then Jiangxi Tecom) and scored each on every cell of its missing lab. The final model trains on three quarters of every source's training half, validates on the last quarter, and is scored on the test halves.

## 14. Standard framing

Looking at the first leave-one-lab-out run's mistakes showed that labs differ in framing as much as in stain: the nucleus spans about 37% of the image side in PBC images against 55% to 66% in the other sources' crops, and backgrounds run from pink to yellow to gray (measurements in `reports/metrics/v2_experiments.json`). `canonical.py` finds the nucleus, crops 2.4 times its size around it (padding with the background color where needed) and divides out the background color. The same function builds the training arrays and runs in the app.

The honest result: on the unseen BCCD cells, the run with standard framing scored 50.7%, against 50.4% for the first attempt, which is within run-to-run noise. The second run was better on the training lab (PBC validation accuracy 95.0% against 91.5%), but it also differed in starting weights (version 1 instead of scratch), epochs, learning rate and the padding augmentation, so that gain cannot be put down to framing alone. I kept framing because it makes single-cell uploads and whole-field cut-outs reach the model in the same framing, whatever the crop.

Padding happens far more often for external cells (93%) than for PBC cells (28%), which could become a clue to the source. The augmentation therefore adds random flat gray borders to training images.

## 15. Stain-strength normalization, tried and dropped

BCCD nuclei are about three times paler than PBC nuclei, measured as optical density (medians 1.2 and 3.5 for the 98th percentile). Scaling every cell so that percentile matches the PBC median (the intensity step of Macenko-style normalization) seemed the obvious next step. It lowered accuracy on the unseen BCCD cells from 50.7% to 40.5%, so I removed it. The comparison is in `reports/metrics/v2_experiments.json`. Choosing between these set-ups on BCCD makes the BCCD figure slightly optimistic; Jiangxi Tecom was not used for any choice.

## 16. Batched augmentation

Augmenting one image at a time took about 8 minutes per epoch on two CPU cores (observed, not logged). `augment.py` applies every transform to a whole batch with per-image random settings, which cut that to about 3.5 minutes per epoch including training. A quarter of each batch gets only flips and rotations, so the classifier keeps seeing clean cells like the ones it is scored on.

## 17. A pretrained backbone, left for a GPU

ImageNet-pretrained features usually transfer better between labs. The official Keras MobileNet weights could be downloaded, and after one epoch a MobileNet model (cut after block 11, 96 px) already had a higher macro-F1 on the unseen BCCD cells than the kept model (0.49 against 0.44). But one epoch took about 20 minutes on two CPU cores, against 3.5 for the small CNN, and a fair comparison needs three full runs. I stopped it, and it is the first thing to try with a GPU.

## 18. Adapting to a lab instead of hoping

Nothing tried here made the model reliable on a lab it had never seen. Training on 84 or 99 of that lab's own labeled cells helped most: accuracy on held-back cells went from 30% to 86% (Jiangxi Tecom) and 42% to 81% (BCCD) in about three minutes on two CPU cores. Most of the gain was in neutrophils; rare types stayed weak, and BCCD eosinophil recall fell from 77% to 54%. `scripts/finetune_local.py` packages that step for a hospital: one folder per cell type, a held-back share scored before and after, a replay set of original cells so the model does not forget them, and recalibrated thresholds. It writes the adapted model to its own folder, so the shipped model is untouched until someone chooses to replace it.

## 19. Finding cells in whole fields

Users upload whole fields, not tidy crops. A learned detector would need training data I do not have for most labs, so the finder is classical: a nucleus score (darkness plus blue over red and green), Otsu's threshold, morphology, and joining pieces that belong to one nucleus. Red cells are dark pink, so a blue-over-red filter keeps them out. Each cut-out goes through the blood cell check, which is meant to reject what the finder gets wrong (the finder was scored on its own, so this was not measured separately). In auto mode, if the middle of a smear holds no white cell, the app searches the whole image instead.
