# CNN Image Denoiser — Residual Noise Prediction with PyTorch

A convolutional neural network (CNN) that removes Gaussian noise from grayscale images using residual noise prediction, built for ELEC5304 at the University of Sydney.

---

## What It Does

The model learns to denoise images by predicting the noise component itself, rather than predicting the clean image directly. Given a noisy input image, the network outputs an estimate of the noise, which is then subtracted to recover the clean image. This residual learning approach tends to converge faster and more stably than direct image regression.

The pipeline is evaluated using **PSNR (Peak Signal-to-Noise Ratio)** — the higher the PSNR, the closer the denoised output is to the ground truth.

An additional experiment tests the generalization of the Gaussian-trained model on **speckle (multiplicative) noise**, analyzing whether the learned representations transfer to a different noise distribution.

---

## Architecture

```
Input: Noisy Grayscale Image (1 × 180 × 180)
         │
         ▼
  ┌─────────────────────────────────────────────┐
  │ Layer 1:  Conv2d(1 → 64)  + ReLU            │
  │ Layers 2–12: Conv2d(64 → 64) + BN + ReLU   │  ← 11 repeated blocks
  │ Layer 13: Conv2d(64 → 1)                    │
  └─────────────────────────────────────────────┘
         │
         ▼
  Predicted Noise (1 × 180 × 180)
         │
         ▼
  Denoised Image = Noisy Input − Predicted Noise
```

| Component | Detail |
|---|---|
| Depth | 13 convolutional layers |
| Channels | 64 (intermediate) |
| Activation | ReLU (all hidden layers) |
| Normalization | BatchNorm2d (layers 2–12) |
| Weight init | Orthogonal (Conv2d), zero bias |
| Loss function | MSE on predicted vs. true noise |
| Optimizer | Adam (lr = 0.001) |
| LR schedule | StepLR — halved every 30 epochs |
| Batch size | 32 |
| Epochs | 100 |

### Noise Model

Gaussian noise is injected during dataset construction:

```python
noise = torch.randn(img.size()).mul_(sigma / 255.0)   # σ = 10 by default
noisy = (img + noise).clamp(0, 1)
```

The network is trained to predict `noisy - original` (i.e., the noise), so the loss is:

```
L = MSE( f(noisy), noisy - original )
```

### Evaluation Metric

```
PSNR = 20 × log₁₀(1 / √MSE)   [dB]
```

Higher PSNR = lower noise = better reconstruction.

---

## Dataset

400 grayscale 180×180 PNG images, split into:

- **Training:** 350 images
- **Testing:** 50 images

The dataset is hosted on Dropbox and downloaded inside the notebook (see [Running the Project](#running-the-project) below).

---

## Repository Structure

```
.
├── project_1_jorge.ipynb            # Full implementation with results
├── project_1_template_2026-1.ipynb  # Original assignment template
└── README.md
```

---

## Running the Project

### Requirements

- Python 3.8+
- PyTorch + torchvision
- Pillow
- Matplotlib
- NumPy
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
pip install torch torchvision pillow matplotlib numpy jupyter
```

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Launch the notebook**

```bash
jupyter notebook project_1_jorge.ipynb
```

**3. Download the dataset**

Run the first cell in the notebook, which downloads and unzips the image set:

```python
!wget "https://www.dropbox.com/..." -O ImageSet.zip
!unzip -q ImageSet.zip
```

**4. Run all cells**

Execute the notebook top-to-bottom. The training loop will run for 100 epochs, printing PSNR on the test set at each epoch.

**5. View results**

The final cells display:
- Side-by-side comparisons: noisy / original / denoised
- Training loss curve
- PSNR improvement over epochs
- Speckle noise generalization results

---

## Results

| Stage | Mean PSNR |
|---|---|
| Noisy input (baseline) | ~28 dB |
| After denoising (100 epochs) | Improves with training |

The additional experiment evaluates three speckle noise sigma values visually matched to the Gaussian training noise, comparing before/after PSNR to measure cross-noise generalization.

---

## Acknowledgements

This project was completed as part of **ELEC5304 — Digital Image Processing** at the **University of Sydney**, Semester 1, 2026.

Group members:

- **Matthew Kimitano** 
- **Naratorn Pisedtasalasai**
- **Jorge Lara Mino**
