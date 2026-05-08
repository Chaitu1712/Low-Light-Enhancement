# Low-Light Image Enhancement

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)

EL-GAN (Efficient Low-Light Generative Adversarial Network) is a lightweight deep learning model designed to restore and enhance images captured in extremely poor lighting conditions. By combining **Iterative Curve Estimation** with **Depthwise Separable Convolutions**, EL-GAN achieves high-quality enhancement with significantly lower computational overhead than standard U-Net based architectures.

## 🌟 Key Features
- **Efficient Architecture:** Utilizes Depthwise Separable Convolutions to reduce parameter count, making it suitable for edge deployment.
- **Three-Stage Training:** A robust curriculum involving unsupervised pre-training, adversarial training, and perceptual fine-tuning.
- **RAW Data Support:** Custom pipeline to handle Sony SID (.ARW) raw sensor data with exposure compensation up to 300x.
- **Multi-Loss Optimization:** Combines Spatial Consistency, Color Constancy, Exposure Control, and VGG-based Perceptual losses.

## 🏗️ Model Architecture
The EL-GAN Generator estimates pixel-wise enhancement curves rather than direct pixel mapping. The transformation follows the iterative formula:
$I_{n+1} = I_n + \alpha_n I_n (I_n - 1)$

This approach ensures the dynamic range is preserved and prevents over-saturation or "clipping" of highlights.

## 📊 Evaluation Results

The model was benchmarked on the **LOL Dataset** (Standard) and the **SID Sony Dataset** (Extreme Low-Light).

| Dataset | PSNR (dB) | SSIM |
| :--- | :---: | :---: |
| **LOL (Standard)** | **18.56** | **0.74** |
| **SID (Extreme)** | **12.55** | **0.12** |

### Visual Comparisons
#### LOL Dataset (Standard Low-Light)
<img width="1500" height="1500" alt="image" src="https://github.com/user-attachments/assets/b862ca66-f331-474f-b3d6-13cf8c39a616" />

> EL-GAN restores vibrant colors and recovers structural details in indoor scenes.

#### SID Dataset (Extreme Low-Light)
<img width="1500" height="1500" alt="image" src="https://github.com/user-attachments/assets/4b9580a7-c376-4791-8855-e18e06fe35c1" />

> Even in near-total darkness, the model successfully reconstructs the scene geometry and color profile from raw sensor data.

## 🚀 Training Strategy
EL-GAN is trained in three distinct phases:
1. **Stage 1: Warm-up (5 Epochs):** Uses non-reference physics-based losses (Spatial, Exposure, Color) to teach the generator the basics of light distribution.
2. **Stage 2: Adversarial (15 Epochs):** Introduces a **PatchGAN Discriminator** to improve local texture and high-frequency details.
3. **Stage 3: Perceptual Fine-Tuning (10 Epochs):** Increases the weight of **VGG-19 Perceptual Loss** to maximize visual realism.

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chaitu1712/Low-Light-Enhancement.git
   cd EL-GAN-Enhancer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Inference on a single image:**
   ```python
   from models.generator import Generator_EL_GAN
   # Load your trained weights and run inference using the provided utility
   # See inference.py for a full example
   ```

## 📂 Dataset Credits
- **LOL Dataset:** [Low-Light Dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset)
- **SID Dataset:** [See-In-the-Dark Dataset](https://www.kaggle.com/datasets/marcorosato/sid-dataset)
- **MIT-Adobe 5K:** [Expert-retouched images for color guidance](https://www.kaggle.com/datasets/weipengzhang/adobe-fivek)

## 📝 Future Work
- [ ] Integration of **Attention Mechanisms** (Spatial & Channel) to focus on high-noise areas.
- [ ] Direct **Bayer-pattern RAW** processing to bypass ISP artifacts.
- [ ] Model quantization for **ONNX/TensorRT** mobile deployment.

## 🎓 Author
**Your Name**
*   [Kaggle Profile](https://www.kaggle.com/chaitu1712)
*   [LinkedIn Profile](www.linkedin.com/in/chaitanyapandey1712)
```
