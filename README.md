# Project A: CNN from Scratch — CIFAR-10 Image Classifier

## Overview
A TensorFlow/Keras notebook that explores CIFAR-10, implements convolution and pooling demonstrations, trains LeNet-5 and a custom CNN, compares two augmentation methods, uses MobileNetV2 transfer learning, builds a results table, and demonstrates a residual block.

## Dataset
- CIFAR-10: 60,000 colour images
- Image dimensions: 32 × 32 × 3
- Training set: 50,000 images
- Test set: 10,000 images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Keras downloads CIFAR-10 automatically the first time it is loaded.

## Requirements
Python version compatible with your TensorFlow release, TensorFlow/Keras, NumPy, pandas, Matplotlib, scikit-learn, and Jupyter.

Install packages:
```bash
python -m pip install -r requirements.txt
```

## Run
1. Open `DL_PR4A.ipynb` in Jupyter Notebook or VS Code.
2. Select the Python environment where requirements are installed.
3. Run cells from top to bottom.
4. Training all models may take a long time on CPU. For a quick code check, temporarily lower epoch counts; restore the task settings for the final run.
5. The test set is reserved for final evaluation. Validation samples are taken from the training set.
6. Read the final comparison table and make sure your written summary matches the measured results.

## Tasks included

### A1 — Load and explore
- Inspect shapes and class distribution.
- Plot 5 samples per class in a 5 × 10 grid.
- Normalize image pixels from 0–255 to 0–1.
- One-hot encode labels.
- Explain CNN versus ANN and note visually similar classes.

### A2 — Convolution
- Demonstrate a convolution filter on a small image.
- Compare `VALID` and `SAME` padding and strides.
- Display 16 feature maps.

### A3 — Pooling and architectures
- Compare max pooling and average pooling.
- Demonstrate `Flatten`.
- Build and train a LeNet-5-style CNN.
- Build a custom CNN with fewer than 2 million parameters.

### A4 — Train and evaluate
- Adam optimizer and categorical cross-entropy.
- Up to 100 epochs, batch size 64, 10% validation split.
- `EarlyStopping` and `ReduceLROnPlateau`.
- Accuracy/loss curves, classification report, confusion matrix, and misclassified examples.

### A5 — Augmentation
- `ImageDataGenerator` with rotation, shifts, horizontal flips, and zoom.
- Train/evaluate an augmented scratch CNN.
- Use modern Keras augmentation layers in a Functional API model.
- Compare validation curves and test accuracy.

### A6 — Transfer learning and residual connection
- Resize images to 96 × 96 for MobileNetV2.
- Feature extraction with pretrained ImageNet weights.
- Fine-tune the final 30 base-model layers with a small learning rate.
- Compare all six required models using architecture, augmentation, transfer learning, test accuracy, and parameter count.
- Highlight maximum test accuracy.
- Build a Functional API residual block and explain skip connections.

## Notes
- The first MobileNetV2 run may download ImageNet weights and therefore needs an internet connection.
- CIFAR-10 images are normalized to [0, 1]; the model maps them to [-1, 1] before MobileNetV2.
- Results vary with hardware, package versions, random variation, and training duration. Do not invent accuracy values.
- Fine-tuning requires more compute than feature extraction.
- If memory is limited, reduce batch size or run models individually and document changes.

## Submission checklist
- [ ] Run all cells without errors.
- [ ] Confirm all six models have evaluated test accuracies.
- [ ] Check plots and the final comparison table.
- [ ] Save the notebook as `DL_PR4A.ipynb`.
- [ ] Export to HTML if required.
- [ ] Submit this README and `requirements.txt`.
