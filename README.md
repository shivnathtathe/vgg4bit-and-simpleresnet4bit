# True 4-Bit Quantized CNN Training on CPU

[![arXiv](https://img.shields.io/badge/arXiv-2603.13931-b31b1b.svg)](https://arxiv.org/abs/2603.13931)
[![License](https://img.shields.io/badge/License-Research%20Only-blue.svg)](#license)

**Paper:** [True 4-Bit Quantized Convolutional Neural Network Training on CPU: Achieving Full-Precision Parity](https://arxiv.org/abs/2603.13931)

**Author:** Shivnath Tathe (Independent Researcher, Pune, India)

We train CNNs from scratch at true 4-bit precision on free CPUs and match FP32 accuracy. No GPUs, no pre-trained weights, no post-training quantization.

---

## Key Results

### CIFAR-10 (VGG4bit, 3.25M params)

| Metric | Value |
|---|---|
| Test Accuracy | **92.34%** |
| FP32 Baseline (VGG) | 92.5% |
| Gap | **0.16%** |
| FP32 Model Size | 12.40 MB |
| INT4 Model Size | **1.55 MB** |
| Compression | **8x** |
| Grid Utilization | 15/15 levels (full) |
| Hardware | Google Colab free CPU (Intel Xeon) |
| Training Cost | $0 |

### CIFAR-100 (VGG4bit, 3.25M params)

| Metric | Value |
|---|---|
| Test Accuracy | **70.94%** |
| Classes | 100 |
| FP32 Model Size | 12.58 MB |
| INT4 Model Size | **1.57 MB** |
| Compression | **8x** |
| Grid Utilization | 15/15 levels (full) |
| Hardware | Google Colab (GPU-assisted) |

### Mobile Validation (OnePlus 9R, ARM)

| Metric | Value |
|---|---|
| Accuracy | **83.16%** |
| Convergence | 6 epochs |
| Hardware | Consumer phone, no special kernels |

---

## Comparison with Prior Work

| Model | Bits | CIFAR-10 Acc | Memory | Hardware |
|---|---|---|---|---|
| **VGG4bit (ours)** | **4** | **92.34%** | **1.55 MB** | CPU |
| VGG FP32 baseline | 32 | 92.5% | 12.40 MB | GPU |
| DoReFa-Net (4-bit) | 4 | 85-88% | - | GPU |
| PACT (4-bit) | 4 | ~92% | - | GPU |

Our method achieves FP32 parity at 4-bit on a free CPU. Prior 4-bit QAT methods required GPUs and generally reported lower accuracy.

---

## Method

Three components enable stable 4-bit training from scratch:

**1. Symmetric 4-bit Quantization + STE**

Every forward pass quantizes all weight matrices to 15 discrete levels [-7s, ..., 0, ..., +7s]. The Straight-Through Estimator passes gradients through rounding.

```python
w_quant, scale = quantize_symmetric(weight, bits=4)
w_ste = weight + (w_quant - weight).detach()  # Forward: quantized, Backward: full-precision
```

**2. Tanh Soft Weight Clipping (our key contribution)**

Applied after every optimizer step to prevent outlier weights from inflating the quantization scale:

```python
W = tanh(W / 3.0) * 3.0
```

This keeps the scale tight so all 15 grid points cover a useful range. Without it, a single outlier can waste most of the quantization resolution.

**3. QuantAwareAdamW**

Standard AdamW + gradient clipping + tanh soft clipping after each update. Biases and batch norm parameters stay in FP32.

---

## Architecture

### VGG4bit (3.25M params)

```
Block 1: Conv4bit(3,64) -> BN -> ReLU -> Conv4bit(64,64) -> BN -> ReLU -> MaxPool
Block 2: Conv4bit(64,128) -> BN -> ReLU -> Conv4bit(128,128) -> BN -> ReLU -> MaxPool
Block 3: Conv4bit(128,256) -> BN -> ReLU -> Conv4bit(256,256) -> BN -> ReLU -> MaxPool
Classifier: Linear4bit(4096,512) -> BN -> ReLU -> Dropout(0.5) -> Linear4bit(512, num_classes)
```

### SimpleResNet4bit (~73K params)

Lightweight ResNet-style model with residual connections. 6 Conv4bit layers + global average pooling.

---

## Training Setup

| Hyperparameter | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Optimizer | QuantAwareAdamW | QuantAwareAdamW |
| Learning Rate | 0.001 | 0.001 |
| LR Schedule | Cosine Annealing (T=100) | Cosine Annealing (T=100) |
| Epochs | 150 | 150 |
| Batch Size | 128 | 128 |
| Weight Decay | 5e-4 | 5e-4 |
| Gradient Clipping | 0.5 | 0.5 |
| Tanh Alpha | 3.0 | 3.0 |
| Augmentation | RandomCrop(32, pad=4), HFlip | RandomCrop(32, pad=4), HFlip |

---

## Installation

```bash
git clone https://github.com/shivnathtathe/vgg4bit-and-simpleresnet4bit.git
cd vgg4bit-and-simpleresnet4bit
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.10+

## Usage

```bash
python main.py
```

You will be prompted to choose:
- `1` for SimpleResNet4bit (lightweight, ~73K params)
- `2` for VGG4bit (heavier, 3.25M params, higher accuracy)

CIFAR-10/100 datasets download automatically on first run.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1F0HziLzVtOQWFG0efaepXdVisHfHBBJt?usp=sharing)

---

## Training Curves

<p align="center">
  <img src="results/training_curve.png" width="600"><br>
  <em>CIFAR-10: VGG4bit converges to 92.34% test accuracy</em>
</p>

<p align="center">
  <img src="results/cifar100_curve.png" width="600"><br>
  <em>CIFAR-100: 70.94% accuracy across 100 classes</em>
</p>

<p align="center">
  <img src="results/tanh_clipping.png" width="600"><br>
  <em>Tanh soft clipping vs hard clipping: smoother gradient flow, better convergence</em>
</p>

---

## Project Structure

```
.
├── main.py              # Full training code (model, optimizer, training loop)
├── requirements.txt     # Python dependencies
├── logs/
│   ├── logs_cipher10.txt    # CIFAR-10 training logs (150 epochs)
│   └── logs_cipher100.txt   # CIFAR-100 training logs (150 epochs)
├── results/
│   ├── training_curve.png       # CIFAR-10 accuracy curve
│   ├── cifar100_curve.png       # CIFAR-100 accuracy curve
│   ├── tanh_clipping.png        # Tanh vs hard clipping comparison
│   └── ...                      # Additional visualization plots
├── paper/
│   └── main.tex                 # LaTeX source for the paper
└── LICENSE
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{tathe2026true4bit,
  title={True 4-Bit Quantized Convolutional Neural Network Training on CPU: Achieving Full-Precision Parity},
  author={Tathe, Shivnath},
  journal={arXiv preprint arXiv:2603.13931},
  year={2026}
}
```

---

## Related Work

This is **Paper 1** in our series on training neural networks from scratch at 4-bit precision:

- **Paper 1** (this repo): 4-bit CNNs on CIFAR-10/100 -- [arXiv:2603.13931](https://arxiv.org/abs/2603.13931)
- **Paper 2** (coming soon): 4-bit Transformers on WikiText-103 and Shakespeare

---

## License

Copyright (c) 2025-2026 Shivnath Tathe. All rights reserved.

This code is distributed for academic and research purposes only. Redistribution, reproduction, or commercial use is strictly prohibited without explicit written permission.

Contact: sptathe2001@gmail.com
