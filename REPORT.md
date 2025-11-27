# Food-10 Image Classification – Technical Report

> Fill in the metric placeholders once you complete training/evaluation. Use this document (≈1–2 pages) as part of the deliverables package.

---

## 1. Problem Overview

- **Task:** Fine-grained food image classification (10 classes) using transfer learning.
- **Dataset:** Kaggle “Food items classification dataset (10 classes)” – 7,500 training + 2,500 test images, balanced (250 samples/class/test split).
- **Objective metric:** Macro F1-score on the held-out test split.

Use cases include restaurant menu automation, recommendation/search for food-delivery apps, and diet-tracking assistants.

---

## 2. Experimental Setup

| Component            | Configuration (edit as needed)                                  |
|----------------------|------------------------------------------------------------------|
| Backbone             | `resnet50` (pretrained ImageNet) / `resnetv2_50` (timm)          |
| Input resolution     | 224 × 224                                                        |
| Optimizer            | AdamW (`lr=1e-4`, `weight_decay=1e-4`)                           |
| Scheduler            | Cosine annealing (`eta_min=1e-6`, `T_max=NUM_EPOCHS`)            |
| Epochs               | 30 (early stopping patience = 7 on macro F1)                     |
| Batch size           | 32                                                               |
| Augmentations        | Resize → random crop, horizontal flip, ±15° rotation, color jitter|
| Hardware             | (e.g., NVIDIA RTX 3060 12GB, CUDA 12.1)                          |

> Mention any deviations (e.g., EfficientNet-B0 run, different hyperparameters, mixup, label smoothing).

---

## 3. Results

Populate the table after running `python train_food10.py`.

| Model             | Macro F1 | Accuracy | Notes                                |
|-------------------|----------|----------|--------------------------------------|
| ResNet-50 (base)  |          |          |                                      |
| ResNetV2-50 (timm)|          |          | Set `MODEL_NAME="resnetv2_50"`       |
| EfficientNet-B0   |          |          | (optional second experiment)         |

- **Best checkpoint:** `checkpoints/<model>_best.pth`
- **Training curves:** `results/<model>_training_history.png`
- **Confusion matrix:** `results/<model>_confusion_matrix.png`
- **Per-class F1 bar plot:** `results/<model>_per_class_f1.png`

Insert the final confusion matrix PNG and a sample Grad-CAM visualization into your submission deck/report.

---

## 4. Qualitative Analysis

1. **Correct predictions:** Describe representative successes (e.g., correctly identifying “fried_rice” despite cluttered background).
2. **Common confusions:** Note the most confused class pairs referencing the confusion matrix (e.g., “noodles vs. pasta”).
3. **Grad-CAM insights:** Summarize the regions highlighted for 2–3 samples (heatmaps saved via `gradcam.py` or Streamlit app).

---

## 5. Error Analysis & Future Work

- Increase input resolution (e.g., 299×299) or apply Test-Time Augmentation to boost F1.
- Experiment with EfficientNet-B3 / ConvNeXt-T for stronger baselines.
- Incorporate class-balanced sampling or focal loss if misclassifications are imbalanced.
- Deploy via TorchScript or build a lightweight API for inference.

---

## 6. Reproducibility Checklist

- [ ] Committed exact `config.py` used for best run.
- [ ] Saved `requirements.txt` (with `kaggle` package).
- [ ] Preserved checkpoint + metrics JSON in `results/`.
- [ ] Captured confusion matrix + Grad-CAM screenshots.
- [ ] Updated this report with final numbers before delivery.

---

**Prepared by:** _<Your Name / Team>_<br>
**Date:** _<YYYY-MM-DD>_


