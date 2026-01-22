# ECG Image Enhancement - Implementation Plan

## Project Overview

**Task**: Convert low-quality photos or scanned images of ECG paper printouts into high-quality, clean scan images.

| Aspect | Details |
|--------|---------|
| **Input** | Low-quality PNG images of ECG paper printouts (photographed/degraded scans) |
| **Output** | Clean, high-quality scan images (enhanced ECG printouts) |
| **Training samples** | 977 samples, each with 9 image strips + ground truth CSV |
| **Test samples** | 2 images (24 leads total) |
| **Task Type** | Image-to-Image Translation (Quality Enhancement) |
| **Project Type** | University research assignment with written paper |
| **Compute** | Local GPU (AMD + NVIDIA) |

---

## 1. Problem Formulation

### 1.1 Task Definition

This is an **image-to-image translation and enhancement** problem:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A: Direct Enhancement** | End-to-end model maps low-quality image → high-quality scan | Simpler pipeline, optimizes for end result | May miss fine details, harder to control |
| **B: Multi-Stage Pipeline** | Stage 1: Denoise/correct → Stage 2: Enhance contrast → Stage 3: Grid/trace refinement | More interpretable, modular, easier to debug | More complex, potential error accumulation |
| **C: GAN-based Translation** | Generator creates clean scans, discriminator ensures realism | Produces sharp, realistic outputs | Training instability, mode collapse risks |

**Chosen Approach: GAN-based with Multi-Stage Components (Hybrid)** because:
- Produces photorealistic high-quality scans
- Provides clear ablation study opportunities (with/without different stages)
- More interpretable for academic writing
- Allows analysis of intermediate representations
- Can use ground truth signals to generate perfect target scans

### 1.2 Formal Problem Statement

Given an input image $I_{low} \in \mathbb{R}^{H \times W \times 3}$ of a low-quality ECG paper strip:
- **Learn enhancement function**: $f_{enhance}: I_{low} \rightarrow I_{high}$ where $I_{high}$ is a clean, high-quality scan
- **Target generation**: Use ground truth CSV signals to synthesize perfect scan images as training targets
- **Optimization**: Minimize perceptual distance between enhanced image and clean target while preserving ECG trace accuracy

---

## 2. Data Analysis & Target Generation

### 2.1 Data Structure

Each training sample contains:
- **9 PNG image strips** (numbered -0001, -0003 through -0006, -0009 through -0012) - LOW QUALITY inputs
- **1 CSV file** with 12 leads (used to generate HIGH QUALITY target scans)

Standard 12-lead ECG layout (typical):
```
Strip Layout:
┌─────────────────────────────────────────────────┐
│ I    │ aVR   │ V1   │ V4   │   (2.5s each)     │
│ II   │ aVL   │ V2   │ V5   │                   │
│ III  │ aVF   │ V3   │ V6   │                   │
├─────────────────────────────────────────────────┤
│ II (rhythm strip) - 10 seconds continuous      │
└─────────────────────────────────────────────────┘
```

### 2.2 Training Data Strategy

**Key Challenge**: We don't have paired (low-quality, high-quality) images!

**Solution**: Hybrid approach combining real and synthetic data

```
┌──────────────────────────────────────────────────────────────────┐
│                   TRAINING DATA GENERATION                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OPTION A: Synthetic Pairs from CSV (Primary Approach)           │
│  ═══════════════════════════════════════════════════════════     │
│  1. Generate Clean Target from CSV                               │
│     ├─ Parse ground truth signal values                          │
│     ├─ Render perfect ECG scan (matplotlib/PIL)                  │
│     ├─ High resolution with clean grid and traces               │
│     └─ Save as "high-quality target"                             │
│                                                                  │
│  2. Create Degraded Input                                         │
│     ├─ Take the clean synthetic scan                             │
│     ├─ Apply degradations: blur, noise, compression, lighting   │
│     ├─ Match the style of real low-quality images               │
│     └─ Save as "low-quality input"                               │
│                                                                  │
│  3. Training Pair                                                 │
│     (Degraded Synthetic Input) → Model → (Clean Synthetic Target)│
│                                                                  │
│  OPTION B: Self-Supervised with Real Images                       │
│  ═══════════════════════════════════════════════════════════     │
│  1. Use Real Image as Target                                     │
│     ├─ Take original image as "relatively clean"                │
│     └─ This is our best available quality                        │
│                                                                  │
│  2. Create Degraded Version                                       │
│     ├─ Apply additional degradation to real image               │
│     ├─ More blur, noise, artifacts                               │
│     └─ Model learns to restore to original                       │
│                                                                  │
│  3. Training Pair                                                 │
│     (Heavily Degraded Real) → Model → (Original Real Image)      │
│                                                                  │
│  HYBRID APPROACH (Recommended):                                   │
│  ═══════════════════════════════════════════════════════════     │
│  • Train on BOTH synthetic and self-supervised pairs             │
│  • Synthetic pairs: Learn ideal scan quality                     │
│  • Self-supervised: Learn to handle real artifacts              │
│  • Combine with weighted loss or alternating batches            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 Data Augmentation & Degradation Strategy

#### Degradation Pipeline (to create low-quality inputs from clean targets):

| Degradation Type | Purpose | Implementation |
|------------------|---------|----------------|
| **Blur** | Simulate camera shake, out-of-focus | Gaussian blur (σ=1-3), motion blur |
| **Noise** | Sensor noise, paper artifacts | Gaussian noise, salt-and-pepper, speckle noise |
| **Compression** | JPEG artifacts from scanning/photo | JPEG compression (quality=50-85) |
| **Lighting** | Non-uniform illumination | Brightness gradients, shadows, glare spots |
| **Color cast** | Scanner/camera white balance issues | Color temperature shifts, fading |
| **Geometric** | Photo perspective, paper warping | Small rotation (±3°), perspective distortion |
| **Grid degradation** | Aged paper, faded printing | Fade grid lines, make irregular thickness |
| **Paper texture** | Real paper artifacts | Add texture, wrinkles, coffee stains |

#### Augmentation (applied to both input and target consistently):

| Augmentation | Purpose | Application |
|--------------|---------|-------------|
| **Geometric (paired)** | Training robustness | Random rotation, flip, crop - SAME for both |
| **Resolution** | Handle different scan qualities | Random resize + resize back |

**Critical**: 
- Degradations applied ONLY to inputs (to create challenge)
- Geometric augmentations applied IDENTICALLY to both input and target
- Target remains clean/perfect

**Libraries**: `albumentations`, `imgaug`, or custom degradation pipeline

---

## 3. Model Architecture

### 3.1 Primary Architecture: Pix2Pix GAN (Recommended)

**Why Pix2Pix?** We have perfect paired training data (low-quality input, synthetic high-quality target)!

```
┌─────────────────────────────────────────────────────────────────┐
│                         PIX2PIX GAN                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GENERATOR (U-Net Architecture):                                 │
│                                                                  │
│  ┌──────────────┐                                               │
│  │ Low Quality  │                                               │
│  │ ECG Image    │                                               │
│  │  (H×W×3)     │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────┐                   │
│  │  ENCODER                                │                   │
│  │  Conv-BN-LeakyReLU-Downsample (×8)     │                   │
│  │  Channels: 64→128→256→512→512→512→512  │                   │
│  └───────────────┬─────────────────────────┘                   │
│                  │                                              │
│                  ▼                                              │
│  ┌─────────────────────────────────────────┐                   │
│  │  DECODER                                │                   │
│  │  Deconv-BN-Dropout-ReLU-Upsample (×8)  │                   │
│  │  Skip connections from encoder          │                   │
│  │  Output: (H×W×3) clean scan            │                   │
│  └───────────────┬─────────────────────────┘                   │
│                  │                                              │
│                  ▼                                              │
│  ┌──────────────────────────────┐                              │
│  │  High Quality ECG Scan       │                              │
│  │  (Generated)                 │                              │
│  └──────────────────────────────┘                              │
│                                                                  │
│  DISCRIMINATOR (PatchGAN):                                       │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐                           │
│  │ Input Image  │   │  Generated   │                           │
│  │ or Target    │   │     OR       │                           │
│  └──────┬───────┘   │  Target      │                           │
│         └───────────┴──────┬───────┘                           │
│                            ▼                                    │
│         ┌──────────────────────────────┐                       │
│         │  Conv-LeakyReLU (×5)         │                       │
│         │  Channels: 64→128→256→512    │                       │
│         │  Output: 30×30 patch matrix  │                       │
│         │  (Real vs Fake per patch)    │                       │
│         └──────────────────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Alternative: Enhanced U-Net (Non-adversarial)

Simpler alternative if GAN training is unstable:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENHANCED U-NET                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Low Quality Image (H×W×3)                               │
│         │                                                        │
│         ▼                                                        │
│  ┌────────────────────────────────────┐                         │
│  │  ENCODER with Residual Blocks      │                         │
│  │  ├─ Initial Conv (64 channels)     │                         │
│  │  ├─ ResBlock + Downsample (×4)     │                         │
│  │  ├─ Channels: 64→128→256→512       │                         │
│  │  └─ Attention gates at bottleneck  │                         │
│  └─────────────┬──────────────────────┘                         │
│                │                                                 │
│                ▼                                                 │
│  ┌────────────────────────────────────┐                         │
│  │  BOTTLENECK                        │                         │
│  │  ├─ Multi-head self-attention      │                         │
│  │  └─ Residual blocks (512 ch)       │                         │
│  └─────────────┬──────────────────────┘                         │
│                │                                                 │
│                ▼                                                 │
│  ┌────────────────────────────────────┐                         │
│  │  DECODER with Skip Connections     │                         │
│  │  ├─ Upsample + Concat + Conv (×4)  │                         │
│  │  ├─ Progressive refinement          │                         │
│  │  └─ Final Conv → 3 channels        │                         │
│  └─────────────┬──────────────────────┘                         │
│                │                                                 │
│                ▼                                                 │
│  Output: High Quality Scan (H×W×3)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Hybrid Multi-Stage Enhancement (Advanced)

For best quality, implement progressive refinement:

```
┌─────────────────────────────────────────────────────────────────┐
│                  MULTI-STAGE REFINEMENT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Low Quality   →   [Stage 1]   →   [Stage 2]   →   [Stage 3]   │
│  Input              Denoise         Enhance         Refine      │
│                     & Align         Traces          Grid        │
│                                                                  │
│  Stage 1: Denoising & Geometric Correction                       │
│  ├─ Remove blur, JPEG artifacts                                 │
│  ├─ Correct perspective distortion                              │
│  └─ Output: Cleaned, aligned image                              │
│                                                                  │
│  Stage 2: Trace Enhancement                                      │
│  ├─ Sharpen ECG signal traces                                   │
│  ├─ Improve contrast                                             │
│  └─ Output: Enhanced traces on background                       │
│                                                                  │
│  Stage 3: Grid & Paper Refinement                                │
│  ├─ Reconstruct perfect ECG grid                                │
│  ├─ Clean paper texture                                          │
│  └─ Output: Professional scan quality                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Training Strategy

### 4.1 Training Configuration

#### For Pix2Pix GAN:

| Parameter | Generator | Discriminator | Rationale |
|-----------|-----------|---------------|-----------|
| **Optimizer** | Adam | Adam | Standard for GAN training |
| **Learning Rate** | 2e-4 | 2e-4 | Balanced GAN training |
| **Beta1, Beta2** | 0.5, 0.999 | 0.5, 0.999 | Pix2Pix standard |
| **Batch Size** | 1-4 (large images) | 1-4 | Memory constraints |
| **Epochs** | 100-200 | - | GANs need more epochs |
| **Weight Decay** | 0 | 0 | Not typically used in GANs |

#### For U-Net (non-adversarial):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | AdamW | Better generalization |
| **Learning Rate** | 1e-4 | Standard for image translation |
| **LR Schedule** | Cosine annealing with warmup | Smooth decay |
| **Batch Size** | 4-16 (depends on GPU memory) | Larger if possible |
| **Epochs** | 50-100 | With early stopping (patience=15) |
| **Weight Decay** | 1e-4 | Regularization |

### 4.2 Loss Functions

#### Pix2Pix Loss:
```python
# Generator Loss
L_GAN = -log(D(G(x)))  # Fool discriminator
L_L1 = ||G(x) - y||_1  # Reconstruction loss

# Perceptual Loss (optional but recommended)
L_perceptual = MSE(VGG_features(G(x)), VGG_features(y))

# Total Generator Loss
L_G = lambda_GAN * L_GAN + lambda_L1 * L_L1 + lambda_perc * L_perceptual
# Typical: lambda_GAN=1, lambda_L1=100, lambda_perc=10

# Discriminator Loss
L_D_real = -log(D(x, y))  # Real pairs
L_D_fake = -log(1 - D(x, G(x)))  # Fake pairs
L_D = (L_D_real + L_D_fake) / 2
```

#### U-Net Loss:
```python
# L1 Loss (preferred over MSE for images)
L_L1 = MAE(generated_scan, target_scan)

# Perceptual/Feature Loss (VGG-based)
L_perceptual = MSE(VGG_features(generated), VGG_features(target))

# SSIM Loss (structural similarity)
L_SSIM = 1 - SSIM(generated_scan, target_scan)

# Edge/Gradient Loss (preserves signal sharpness)
L_gradient = MAE(gradient(generated), gradient(target))

# Combined Loss
L_total = w1*L_L1 + w2*L_perceptual + w3*L_SSIM + w4*L_gradient
# Typical: w1=100, w2=10, w3=1, w4=5
```

### 4.3 GPU Training

**Recommendation**: Use NVIDIA GPU with PyTorch + CUDA for this project (better ecosystem support).

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Mixed precision training for memory efficiency
# Use torch.cuda.amp for automatic mixed precision
```

---

## 5. Evaluation Metrics & Validation

### 5.1 Image Quality Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **PSNR** | Peak Signal-to-Noise Ratio | Overall image quality (higher = better) |
| **SSIM** | Structural Similarity Index | Perceptual quality (0-1, higher = better) |
| **LPIPS** | Learned Perceptual Image Patch Similarity | Deep perceptual distance (lower = better) |
| **MSE/MAE** | Pixel-wise error | Basic reconstruction accuracy |
| **FID** | Fréchet Inception Distance | Distribution similarity (for GAN) |

### 5.2 ECG-Specific Metrics

Since we're enhancing ECG images, we should also validate that the ECG traces are accurate:

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Trace MSE** | Error in extracted signal vs ground truth | Verify medical accuracy |
| **Peak Preservation** | R-peak detection accuracy | Critical ECG features preserved |
| **Grid Accuracy** | Grid line alignment and spacing | Professional scan quality |

### 5.3 Validation Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│                    VALIDATION APPROACH                           │
├──────────────────────────────────────────────────────────────────┤
│  1. Train/Validation Split (977 samples)                         │
│     ├─ 80% training (782 samples)                               │
│     ├─ 20% validation (195 samples)                             │
│     └─ Random split (no need to stratify for image quality)     │
│                                                                  │
│  2. K-Fold Cross-Validation (for paper)                          │
│     ├─ 5-fold CV for robust performance estimates               │
│     ├─ Report mean ± std for all metrics                        │
│     └─ Use for hyperparameter selection                         │
│                                                                  │
│  3. Ablation Studies                                             │
│     ├─ Single-stage vs multi-stage enhancement                  │
│     ├─ With/without perceptual loss                             │
│     ├─ Different backbone architectures (U-Net vs Pix2Pix)      │
│     ├─ Effect of different loss components                      │
│     └─ Impact of augmentation strategies                        │
│                                                                  │
│  4. Visual Quality Assessment                                    │
│     ├─ Side-by-side comparisons (input, output, target)        │
│     ├─ Zoom-ins on ECG trace details                            │
│     ├─ Grid quality visualization                               │
│     └─ User study (optional: medical professionals rating)     │
└──────────────────────────────────────────────────────────────────┘
```

### 5.4 Visualization for Paper

- **Qualitative**: Side-by-side input vs enhanced output vs synthetic target
- **Enhancement progression**: Show multi-stage improvements if using pipeline
- **Error heatmaps**: Visualize where the model struggles
- **Failure cases**: Examples of challenging inputs
- **Ablation visuals**: Show impact of different components

---

## 6. Implementation Workflow

### 6.1 Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      COMPLETE WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TRAINING PHASE:                                                 │
│  ═══════════════                                                 │
│                                                                  │
│  ┌─────────────┐       ┌──────────────┐      ┌──────────────┐  │
│  │   CSV       │       │  Synthetic   │      │   Degrade    │  │
│  │   Ground    │  ───▶ │  Clean Scan  │  ───▶│  (blur,      │  │
│  │   Truth     │       │  Generation  │      │  noise, etc) │  │
│  └─────────────┘       └──────────────┘      └───────┬──────┘  │
│                                                       │         │
│                        ┌──────────────────────────────┘         │
│                        ▼                                        │
│              ┌───────────────────┐                              │
│              │  Training Pairs:  │                              │
│              │  Input: Degraded  │                              │
│              │  Target: Clean    │                              │
│              └─────────┬─────────┘                              │
│                        │                                        │
│                        ▼                                        │
│            ┌──────────────────────┐                             │
│            │   Train Generator    │                             │
│            │   (U-Net / Pix2Pix)  │                             │
│            └──────────────────────┘                             │
│                                                                  │
│  INFERENCE/TEST PHASE:                                           │
│  ═════════════════════                                           │
│                                                                  │
│  ┌─────────────┐       ┌──────────────┐      ┌──────────────┐  │
│  │  Real       │       │   Trained    │      │  Enhanced    │  │
│  │  Low-Qual   │  ───▶ │   Model      │  ───▶│  High-Qual   │  │
│  │  ECG Image  │       │  (Generator) │      │  Scan        │  │
│  └─────────────┘       └──────────────┘      └──────────────┘  │
│                                                                  │
│  VALIDATION (Optional):                                          │
│  ═══════════════════                                             │
│                                                                  │
│  Enhanced Image  ───▶  [Signal Extraction]  ───▶  Compare with  │
│                        (from enhanced scan)        CSV Ground   │
│                                                    Truth        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Why This Approach Works

| Component | Rationale |
|-----------|-----------|
| **CSV → Clean Scan** | Gives us perfect, noise-free targets with known signal accuracy |
| **Synthetic Degradation** | We control what artifacts to add, can match real image statistics |
| **Optional Self-Supervised** | Can also train on real images by degrading them further |
| **Hybrid Training** | Best of both worlds: ideal quality + realistic artifacts |
| **Signal Validation** | Use CSV to verify enhanced images preserve medical accuracy |

---

## 7. Synthetic ECG Scan Generation (Critical Component)

### 7.1 Rendering Pipeline

```python
# Pseudo-code for generating clean ECG scans from CSV

def generate_clean_ecg_scan(csv_signals, metadata):
    """
    Args:
        csv_signals: DataFrame with 12 leads × N samples
        metadata: Sampling rate, lead layout, etc.
    
    Returns:
        clean_scan: High-resolution PIL/numpy image
    """
    
    # 1. Setup canvas
    dpi = 300  # Professional scan quality
    paper_size = (11, 8.5)  # Letter size in inches
    canvas = create_blank_canvas(paper_size, dpi, color='white')
    
    # 2. Draw ECG grid
    draw_ecg_grid(canvas, 
                  small_box_mm=1,    # 1mm boxes (0.04s, 0.1mV)
                  large_box_mm=5,    # 5mm boxes (0.20s, 0.5mV)
                  color='red' or 'pink',
                  line_width=0.5)
    
    # 3. Map leads to positions
    layout = get_standard_layout()  # 3×4 + rhythm strip
    
    # 4. Plot each lead
    for lead_name, signal_data in csv_signals.items():
        position = layout[lead_name]
        plot_signal_trace(canvas, 
                         signal_data,
                         position,
                         sampling_rate=metadata['fs'],
                         amplitude_scale=10,  # mm/mV standard
                         color='black',
                         line_width=1.0)
    
    # 5. Add labels and annotations
    add_lead_labels(canvas, layout)
    add_calibration_pulse(canvas)  # 1mV, 0.2s pulse
    add_metadata_text(canvas, metadata)  # Patient info, settings
    
    # 6. Convert to image
    return canvas_to_image(canvas)
```

### 7.2 Realistic Rendering Considerations

| Aspect | Implementation Details |
|--------|----------------------|
| **Grid style** | Standard pink/red grid or faded gray, match real ECG paper |
| **Trace thickness** | 0.5-1.0mm line width, anti-aliased |
| **Lead layout** | Standard 12-lead format: 3 rows × 4 columns + rhythm strip |
| **Scaling** | 10mm/mV vertical, 25mm/s horizontal (standard) |
| **Resolution** | 300 DPI minimum for professional quality |
| **Paper texture** | Optional: add subtle paper grain for realism |

---

## 8. Implementation Phases

### Phase 1: Foundation & Synthetic Data (Week 1)

- [ ] **Setup & Environment**
  - [ ] Install PyTorch with CUDA
  - [ ] Add dependencies (albumentations, matplotlib, PIL, wandb, etc.)
  - [ ] Setup project structure

- [ ] **Synthetic Clean Scan Generation**
  - [ ] Implement CSV parser for ground truth signals
  - [ ] Create ECG grid rendering function
  - [ ] Implement signal plotting on grid
  - [ ] Generate clean synthetic scans for all training samples
  - [ ] Validate that synthetic scans match real image layout

- [ ] **Degradation Pipeline**
  - [ ] Implement blur, noise, JPEG compression
  - [ ] Add lighting and color artifacts
  - [ ] Create realistic degradation combinations
  - [ ] Generate degraded versions of synthetic scans
  - [ ] Create (degraded_input, clean_target) dataset

### Phase 2: Baseline Model (Week 2)

- [ ] **U-Net Implementation**
  - [ ] Implement basic U-Net architecture
  - [ ] Setup training loop with L1 + perceptual loss
  - [ ] Train on synthetic paired data
  - [ ] Visualize initial results

- [ ] **Evaluation**
  - [ ] Implement PSNR, SSIM, LPIPS metrics
  - [ ] Establish baseline performance
  - [ ] Analyze failure cases
  - [ ] Create visualization utilities

### Phase 3: GAN Training (Week 3)

- [ ] **Pix2Pix Implementation**
  - [ ] Implement PatchGAN discriminator
  - [ ] Setup adversarial training loop
  - [ ] Balance generator and discriminator training
  - [ ] Monitor for mode collapse and training instability

- [ ] **Hyperparameter Tuning**
  - [ ] Tune loss weights (L1, GAN, perceptual)
  - [ ] Experiment with learning rates
  - [ ] Try different discriminator architectures
  - [ ] Ablation: with/without perceptual loss

### Phase 4: Real Data Integration (Week 4)

- [ ] **Self-Supervised Training**
  - [ ] Implement training on real images (degrade → restore)
  - [ ] Combine synthetic and real data training
  - [ ] Analyze performance on real test images
  - [ ] Fine-tune on real data distribution

- [ ] **Signal Validation**
  - [ ] Extract signals from enhanced images
  - [ ] Compare with CSV ground truth
  - [ ] Implement ECG-specific metrics (peak detection, etc.)
  - [ ] Ensure medical accuracy is preserved

### Phase 5: Evaluation & Paper (Week 5)

- [ ] **Comprehensive Evaluation**
  - [ ] K-fold cross-validation
  - [ ] Ablation studies (U-Net vs Pix2Pix, loss components)
  - [ ] Error analysis and failure case identification
  - [ ] Generate publication-quality visualizations

- [ ] **Final Model & Submission**
  - [ ] Train final model on full training set
  - [ ] Generate enhanced scans for test set
  - [ ] Validate quality on test images
  - [ ] Document methodology for paper

---

## 9. Project Structure

```
ekg_ml/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset class for paired data
│   │   ├── synthetic.py        # ECG scan generation from CSV
│   │   ├── degradation.py      # Image degradation pipeline
│   │   └── augmentation.py     # Geometric augmentations
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py             # U-Net generator
│   │   ├── discriminator.py    # PatchGAN discriminator
│   │   ├── pix2pix.py          # Complete Pix2Pix model
│   │   └── losses.py           # Perceptual, L1, adversarial losses
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── gan_trainer.py      # GAN-specific training
│   │   ├── callbacks.py        # Early stopping, checkpointing
│   │   └── config.py           # Hyperparameters
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # PSNR, SSIM, LPIPS
│   │   ├── signal_metrics.py   # ECG trace validation
│   │   ├── visualization.py    # Plotting utilities
│   │   └── ablation.py         # Ablation study utilities
│   │
│   └── main.py                 # Entry point
│
├── configs/
│   ├── default.yaml            # Default configuration
│   └── experiments/            # Experiment-specific configs
│
├── notebooks/
│   ├── data_exploration.ipynb  # Analyze CSV and images
│   └── synthetic_gen.ipynb     # Test scan generation
│
├── scripts/
│   ├── generate_synthetic.py   # Batch synthetic scan generation
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
│
├── data/                       # (gitignored)
│   ├── train/                  # Original low-quality images
│   ├── train_csvs/             # Ground truth CSV files
│   ├── synthetic_clean/        # Generated clean scans
│   ├── synthetic_degraded/     # Generated degraded inputs
│   ├── test/                   # Test images
│   └── sample_submission.parquet
│
├── outputs/                    # (gitignored)
│   ├── checkpoints/
│   ├── logs/
│   ├── enhanced_images/        # Model outputs
│   └── predictions/
│
├── AGENT.md                    # This file
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## 10. Dependencies

```txt
# Core ML
torch>=2.0.0
torchvision>=0.15.0

# Image processing & generation
opencv-python>=4.8.0
albumentations>=1.3.0
Pillow>=9.0.0
matplotlib>=3.7.0

# Image quality metrics
scikit-image>=0.21.0  # For SSIM, PSNR
lpips>=0.1.4          # Learned perceptual similarity

# Data handling
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=12.0.0

# Visualization & logging
wandb>=0.15.0
tensorboard>=2.13.0
tqdm>=4.65.0

# Utilities
pyyaml>=6.0
scikit-learn>=1.2.0

# Signal processing (for validation)
scipy>=1.10.0
```

---

## 11. Key Research Questions for Paper

1. **Does GAN-based training (Pix2Pix) outperform plain U-Net for ECG image enhancement?**
   - Ablation: Pix2Pix vs U-Net vs U-Net+Perceptual
   - Metrics: PSNR, SSIM, LPIPS, visual quality

2. **How important is synthetic data generation from CSV ground truth?**
   - Compare: Synthetic pairs vs self-supervised on real images vs hybrid
   - Analysis: Generalization to test set quality

3. **What degradation types most impact enhancement quality?**
   - Ablation: Train with different degradation combinations
   - Identify: Which artifacts are hardest to remove

4. **Do enhanced scans preserve medical accuracy of ECG traces?**
   - Extract signals from enhanced vs original images
   - Compare with CSV ground truth
   - Validate critical ECG features (R-peaks, intervals)

5. **What is the optimal loss function combination?**
   - Ablation: L1 alone vs L1+Perceptual vs L1+Perceptual+SSIM
   - Analysis: Trade-off between pixel accuracy and perceptual quality

6. **Can the model generalize to different types of image degradation not seen in training?**
   - Test on: Extreme blur, very low resolution, heavy noise
   - Robustness analysis for real-world deployment

---

## 12. References & Resources

- **Pix2Pix**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **Perceptual Loss**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer" (2016)
- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (2018)
- **Image Quality Assessment**: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (2004)
- **Medical Image Enhancement**: Review papers on medical image quality improvement
