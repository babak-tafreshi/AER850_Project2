# AER850 — Fast Defect Classifier (Crack / Missing-Head / Paint-Off)

Transfer-learning pipeline for detecting three surface/fastener defects in aircraft imagery.  
Final system uses **PyTorch EfficientNet-B0** with a head-only warm-up and short **fine-tuning**.  
Trained/validated on 1942/431 images and evaluated on a 539-image test set.

---

## Project Summary

- **Task:** 3-class image classification — `crack`, `missing-head`, `paint-off`.
- **Backbone:** EfficientNet-B0 (ImageNet weights), 256×256 inputs.
- **Training:** Head-only → fine-tune last ~80% of features, label smoothing (0.05), class-weighted CE.
- **Augmentations:** resize, flips, mild affine (±10°, ±6% translate), autocontrast, sharpness.
- **Hardware:** Colab T4 GPU; CPU will work but is slower.

**Final performance**
- **Validation accuracy:** ~95.4% (best checkpoint)
- **Test accuracy:** **94.81%** (macro F1 ≈ 0.944)

| Split | Images | Accuracy |
|---|---:|---:|
| Validation | 431 | 95.36% (best) |
| Test | 539 | **94.81%** |

Per-class (Test set):
- **crack:** P=0.930, R=0.943, F1=0.937 (n=211)  
- **missing-head:** P=0.980, R=0.985, F1=0.983 (n=200)  
- **paint-off:** P=0.927, R=0.898, F1=0.913 (n=128)

---

##  Dataset & Splits

train/ (1942 images)
valid/ (431 images)
test/ (539 images)


Class mapping (Keras/PyTorch agree):

```python
{'crack': 0, 'missing-head': 1, 'paint-off': 2}
