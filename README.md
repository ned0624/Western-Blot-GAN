# Western Blot Image Synthesis with Conditional GAN

A Conditional Generative Adversarial Network (cGAN) built with PyTorch to synthesize Western Blot images from two template inputs — a band mask and a base pattern.

## 📌 Project Overview

Western Blot is a widely used technique in molecular biology to detect specific proteins. This project trains a conditional GAN to generate realistic Western Blot images conditioned on two template inputs, which can be used for data augmentation or simulation purposes.

| Input | Description |
|-------|-------------|
| Template 1 (BandMask) | Binary mask indicating band positions |
| Template 2 (BasePattern) | Background base pattern |
| Output | Synthesized Western Blot image |

## 🏗️ Model Architecture

### Generator (`generator.py`)
An encoder-decoder architecture conditioned on two template inputs:

```
Input: Template1 + Template2 (concatenated, 2 channels)
  → Encoder: Conv2d(2→64) → Conv2d(64→128)
  → Bottleneck: Conv2d(128→256)
  → Decoder: ConvTranspose2d(256→128) → ConvTranspose2d(128→64) → ConvTranspose2d(64→1)
  → Bilinear interpolation to match input size
Output: Synthesized grayscale image (1 channel, Tanh activation)
```

### Discriminator (`discriminator.py`)
A PatchGAN-style discriminator conditioned on both templates:

```
Input: Template1 + Template2 + Fake/Real image (concatenated, 3 channels)
  → Conv2d(3→64) → Conv2d(64→128) → Conv2d(128→256) → Conv2d(256→1)
Output: Patch-level real/fake prediction (Sigmoid)
```

### Loss Function
Combined adversarial and reconstruction loss:

```
Loss_G = BCE Loss (GAN) + λ × L1 Loss (reconstruction)
Loss_D = (BCE Loss on real + BCE Loss on fake) / 2
```

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate (G) | 0.0002 |
| Learning Rate (D) | 0.0001 |
| Adam Betas | (0.5, 0.999) |
| Lambda L1 | 100 |
| Epochs | 100 |
| Batch Size | 1 |

## 📁 Project Structure

```
Western-Blot-GAN/
├── dataset.py          # Custom Dataset class
├── generator.py        # Conditional Generator
├── discriminator.py    # Conditional Discriminator
├── train.py            # Training script
├── test.py             # Inference & evaluation script
├── wb_dataset/         # Real Western Blot images (PNG)
├── wb_template/        # BandMask templates (PNG)
├── wb_template2/       # BasePattern templates (PNG)
├── generator.pth       # Saved generator weights (generated)
├── discriminator.pth   # Saved discriminator weights (generated)
├── training_loss_plot.png  # Loss curve (generated)
├── test_image.png          # Sample results (generated)
├── loss_results.csv        # Per-image L1 loss (generated)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ned0624/Western-Blot-GAN.git
cd Western-Blot-GAN
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare dataset

Place your data in the following structure:
```
wb_dataset/      ← real Western Blot images (e.g. bg_0001.png)
wb_template/     ← BandMask templates (e.g. BandMask_bg_0001.png)
wb_template2/    ← BasePattern templates (e.g. BasePattern_bg_0001.png)
```

### 4. Train

```bash
python train.py
```

Saves `generator.pth`, `discriminator.pth`, and `training_loss_plot.png`.

### 5. Test

```bash
python test.py
```

Outputs `test_image.png` and `loss_results.csv` with per-image L1 loss.

## 🛠️ Tech Stack

- **PyTorch** — model definition, training loop, GPU support
- **Torchvision** — image transforms
- **Pillow / Matplotlib** — image I/O and visualization
- **scikit-learn** — train/test split
- **Pandas** — saving loss results to CSV

## 💡 Key Concepts Demonstrated

- Conditional GAN (cGAN) architecture
- Encoder-decoder generator with bilinear upsampling
- PatchGAN-style discriminator
- Combined GAN + L1 reconstruction loss
- Multi-input conditioning (two template images)
- Custom Dataset class with proportional resizing

## 🔗 Related Projects

- [AOI Defect Classification with CNN](https://github.com/ned0624/Defect-Classifications-of-AOI)
- [Retinal Vessel Segmentation with U-Net](https://github.com/ned0624/Retinal-Vessel-Segmentation)
- [Financial News Sentiment Analysis with BERT](https://github.com/ned0624/Financial-News-Sentiment-BERT)
