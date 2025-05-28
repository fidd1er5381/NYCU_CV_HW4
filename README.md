![image](https://github.com/user-attachments/assets/e719979f-2d58-4e96-8b28-f0d5409f439b)
# NYCU Computer Vision 2025 Spring - Homework 3

StudentID: 313553023  
Name: 褚敏匡

## Introduction

This repository contains an implementation of an enhanced U-Net model with PromptIR-inspired components for image restoration. The system can effectively remove both rain and snow degradations from images using a unified model architecture, achieving superior performance with CBAM attention mechanisms, prompt learning modules, and advanced training strategies.

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd hw4
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Organize your dataset in the following structure:
```
hw4_realse_dataset/
├── train/
│   ├── degraded/
│   │   ├── rain-001.png
│   │   ├── rain-002.png
│   │   ├── snow-001.png
│   │   └── snow-002.png
│   └── clean/
│       ├── rain_clean-001.png
│       ├── rain_clean-002.png
│       ├── snow_clean-001.png
│       └── snow_clean-002.png
└── test/
    └── degraded/
        ├── test-001.png
        ├── test-002.png
        └── ...
```

## Train and Predict

To train the model:
```bash
python run.py
```

To run prediction on test data:
```bash
python run.py  # This will automatically run both training and prediction
```

Or to run prediction only with a pre-trained model:
```python
from run import generate_predictions_with_tta
generate_predictions_with_tta(
    model_path='best_model.pth',
    test_dir='hw4_realse_dataset/test/degraded',
    output_file='pred.npz'
)
```

This will generate:
- `pred.npz`: Image restoration results in the required format for Kaggle submission


### Model Architecture
- **Backbone**: Enhanced U-Net with skip connections
- **Enhanced Components**:
  - CBAM Attention Mechanism (Channel + Spatial attention)
  - PromptIR-inspired Prompt Learning Modules
  - Multi-scale Feature Fusion with Gated Mechanisms
- **Input Resolution**: 256×256 (with random crop augmentation)
- **Parameters**: 33.1M

### Training Configuration
- **Optimizer**: Adam with learning rate 1e-4
- **Loss Function**: Combined L1 (0.7) + SSIM (0.2) + Perceptual (0.1)
- **Scheduler**: ReduceLROnPlateau with patience=5
- **Batch Size**: 8
- **Epochs**: 100
- **Data Augmentation**: Random crop, flips, rotation, color jittering

### Performance Metrics
- **PSNR**: 28.34 dB
- **SSIM**: 0.851
- **Rain PSNR**: 28.41 dB
- **Snow PSNR**: 28.27 dB

## Performance Snapshot
![image](https://github.com/user-attachments/assets/6b45843d-27ed-4856-a5ff-905d647c3e37)

## Outputs

- Best Model: `best_model.pth`
- Prediction Results: `pred.npz`
- Training Visualization: `training_curves.png`

## Enhanced Architecture Pipeline

The system utilizes a comprehensive architecture with multiple advanced components:

1. **CBAM Attention Mechanism**:
   - Channel attention for feature importance weighting
   - Spatial attention for location-aware processing
   - Dual attention design for comprehensive feature enhancement

2. **PromptIR-Inspired Prompt Learning**:
   - Adaptive prompt generation based on input features
   - Degradation-aware prompt selection mechanism
   - Gated fusion for prompt integration

3. **Multi-Scale Feature Processing**:
   - Encoder-decoder architecture with skip connections
   - Feature fusion at multiple scales
   - Residual connections for detail preservation

4. **Advanced Loss Function**:
   - L1 loss for pixel-level accuracy
   - SSIM loss for structural similarity
   - Perceptual loss for edge and texture preservation

5. **Test-Time Augmentation (TTA)**:
   - Multiple transformations (horizontal/vertical flips)
   - Prediction averaging for improved robustness
   - Post-processing with edge enhancement and color correction

These techniques together significantly improve restoration quality, especially for challenging cases like heavy degradations and mixed weather conditions.

## Model Architecture Details

### Core Components

1. **Feature Extraction Layer**
   - Initial 3×3 convolution: 3 → 64 channels
   - Patch embedding for feature initialization

2. **Encoder Path**
   - 4 residual blocks: [64, 128, 256, 512] channels
   - CBAM attention at each level
   - Prompt learning modules for degradation adaptation
   - Max pooling for spatial downsampling

3. **Bottleneck Processing**
   - Deep feature processing: 1024 channels
   - Two consecutive residual blocks
   - Enhanced feature representation

4. **Decoder Path**
   - Symmetric upsampling with transposed convolutions
   - Skip connections from encoder
   - Feature fusion through concatenation and 1×1 convolutions

5. **Output Refinement**
   - Final feature refinement layers
   - Residual connection with input image
   - Output clamping to [0, 1] range

### Key Innovations

- **Prompt Learning**: Degradation-specific adaptive prompts
- **CBAM Integration**: Dual attention for enhanced feature representation
- **Multi-Scale Fusion**: Effective feature integration across scales
- **Combined Loss**: Comprehensive optimization for multiple quality aspects

## Ablation Study Results

Based on the implemented architecture components:

| Component | Description | Key Features |
|-----------|-------------|--------------|
| Base U-Net | Encoder-decoder with skip connections | Standard U-Net architecture |
| + CBAM Attention | Channel + Spatial attention | Dual attention in residual blocks |
| + Prompt Learning | PromptIR-inspired modules | Adaptive prompt generation and fusion |
| + Combined Loss | L1 + SSIM + Perceptual | Multi-objective optimization |
| **Final Model** | **Complete implementation** | **28.34 dB PSNR on Kaggle** |


## Technical Implementation

### Memory Optimization
To ensure efficient training and inference:
- Gradient accumulation for effective larger batch sizes
- Memory-efficient attention computation
- Optimized data loading with proper num_workers

### Training Stability
- Gradient clipping to prevent exploding gradients
- Early stopping to prevent overfitting
- Learning rate scheduling for optimal convergence

### Inference Optimization
- Test-time augmentation for improved results
- Post-processing pipeline for enhanced visual quality
- Efficient batch processing for faster inference


## References

1. Valanarasu, J. M. J., & Patel, V. M. (2023). PromptIR: Prompting for All-in-One Blind Image Restoration. arXiv preprint arXiv:2306.13090.

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. MICCAI.

3. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. ECCV.

4. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE TIP.

