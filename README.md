# Handwritten Character Recognition with EMNIST using Transfer Learning

This project presents a high-accuracy solution for handwritten letter and character recognition using the EMNIST dataset. It leverages a pre-trained Convolutional Neural Network (CNN), fine-tunes it with advanced augmentation and regularization techniques, and employs a systematic approach to overcome the challenges inherent in the dataset.

## Key Features

-   **High Accuracy:** Achieves **88.42%** on the imbalanced *EMNIST ByClass* split and **90.66%** on the *EMNIST Balanced* split, performing close to or at state-of-the-art benchmarks.
-   **Transfer Learning:** Utilizes pre-trained `EfficientNet-B2` and `EfficientNet-B3` models, modified for grayscale input and the specific EMNIST class structure.
-   **Advanced Augmentation:** Employs `MixUp` and `CutMix` to effectively regularize the model, significantly reducing overfitting and improving generalization.
-   **Class Imbalance Handling:** Implements a custom-weighted `KLDivLoss` function to address the severe class imbalance in the *ByClass* dataset, ensuring fair training across all characters.
-   **Systematic Optimization:** Uses `Weights & Biases` for experiment tracking and systematically determines the best optimizer (`Lion`), learning rate scheduler (`CosineAnnealingWarmRestarts`), and hyperparameters.

---

## Dataset: EMNIST

This project uses the **EMNIST (Extended MNIST)** dataset, which is a large collection of handwritten characters and digits.

-   **Structure:** The images are reformatted into a $28 \times 28$ grayscale format, similar to the original MNIST dataset.
-   **Dataset Splits Used:**
    -   **ByClass:** 62 classes, highly imbalanced. Contains 697,932 training images and 116,323 test images.
    -   **Balanced:** 47 balanced classes.
-   **Challenges:**
    -   **Class Imbalance:** In the *ByClass* split, the most frequent class appears over 17 times more often than the least frequent one (33,374 vs 1,896 samples).
    -   **Data Quality:** The dataset contains mislabeled and pre-augmented images (e.g., rotated by 90 degrees), which complicates classification.

![EMNIST Class Distribution](https://github.com/sandeepkumaraau/EMNIST-byclass-/blob/main/Project%20.pdf)
> *Class distribution for the EMNIST ByClass split, highlighting the imbalance.*

---

## Approach & Methodology

### 1. Model Architecture
The core of this project is a modified **EfficientNet** model. While several architectures were tested, `EfficientNet-B2` and `EfficientNet-B3` provided the best balance of performance and computational efficiency.

The pre-trained model was adapted for this task with two key modifications:
1.  The first convolutional layer was changed to accept 1-channel grayscale images instead of the standard 3-channel RGB input.
2.  The final classification layer was replaced with a new one tailored for the 62 classes of the EMNIST *ByClass* dataset.

### 2. Data Preprocessing & Augmentation
-   **Image Resizing:** Original $28 \times 28$ images were resized to $112 \times 112$. This resolution was experimentally determined to offer the best trade-off between feature extraction quality and computational load.
-   **Histogram Equalization:** A custom `HISTEq` transform was applied to standardize image contrast.
-   **Normalization:** The dataset's mean and standard deviation were recalculated after resizing and applied to all images.
-   **Augmentation for Regularization:** Overfitting was a significant challenge. While initial attempts included dropout and standard augmentations, the most effective strategy was a combination of **`MixUp`** and **`CutMix`**. This approach proved so effective that other augmentations and dropout were no longer necessary.

### 3. Training Strategy
-   **Framework:** The model was built and trained using **PyTorch**.
-   **Optimizer:** After experimenting with Adam, AdamW, and SGD, the **`Lion`** optimizer was found to deliver the best results.
-   **Loss Function:** To counter class imbalance in the *ByClass* split, a **`KLDivLoss`** function was used with pre-computed class weights. This ensures that the model does not become biased towards more frequent classes.
-   **Learning Rate Scheduler:** A warmup schedule was implemented using `SequentialLR`, which transitions to a `CosineAnnealingWarmRestarts` scheduler after 5 epochs. This stabilized initial training and helped convergence.

---

## Results

The model achieved performance very close to the established benchmarks for both dataset splits.

| Dataset Split     | Validation Accuracy | Benchmark | F1 Score |
| ----------------- | ------------------- | --------- | -------- |
| **EMNIST ByClass** | **88.42%** | 88.43%   | 87.26%   |
| **EMNIST Balanced** | **90.66%** | 91.06%   | 90.59%   |

#### Training Performance (ByClass Split - 88.42% Accuracy)
![Training vs Validation - ByClass]([https://i.imgur.com/gKkCj6u.png](https://github.com/sandeepkumaraau/EMNIST-byclass-/blob/main/Project%20.pdf))

#### Training Performance (Balanced Split - 90.66% Accuracy)
![Training vs Validation - Balanced]([https://i.imgur.com/uCgL9W7.png](https://github.com/sandeepkumaraau/EMNIST-byclass-/blob/main/Project%20.pdf))

---

## How to Run

### Prerequisites
Make sure you have Python and PyTorch installed. You can install the required libraries using `pip`:
```bash
pip install torch torchvision torch_optimizer torchmetrics timm lion-pytorch torcheval



