"""
Enhanced U-Net with PromptIR-Inspired Components for Image Restoration.

This module implements an enhanced U-Net architecture that incorporates
core concepts from PromptIR, effectively handling both rain and snow
degraded images using CBAM attention mechanisms, prompt learning modules,
and advanced training strategies.

Author: 313553023
"""

import os
import math
from typing import Tuple, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Calculate PSNR between two images.
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


class ImageRestorationDataset(Dataset):
    """
    Dataset class for image restoration task.
    
    Handles both training and testing datasets with proper filename mapping
    for rain and snow degraded images.
    """
    
    def __init__(self, 
                 degraded_dir: str, 
                 clean_dir: Optional[str] = None, 
                 transform: Optional[transforms.Compose] = None, 
                 is_test: bool = False):
        """
        Initialize the dataset.
        
        Args:
            degraded_dir: Directory containing degraded images
            clean_dir: Directory containing clean images (None for test)
            transform: Image transformations to apply
            is_test: Whether this is a test dataset
        """
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.is_test = is_test
        
        # Get all image files
        if is_test:
            self.degraded_files = sorted([
                f for f in os.listdir(degraded_dir) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            # Get both rain and snow degraded images
            all_files = [
                f for f in os.listdir(degraded_dir) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]
            self.degraded_files = sorted([
                f for f in all_files 
                if f.startswith(('rain-', 'snow-'))
            ])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.degraded_files)
    
    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor], 
        Tuple[torch.Tensor, str]
    ]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            For training: (degraded_image, clean_image)
            For testing: (degraded_image, filename)
        """
        degraded_filename = self.degraded_files[idx]
        degraded_path = os.path.join(self.degraded_dir, degraded_filename)
        degraded_img = Image.open(degraded_path).convert('RGB')
        
        if not self.is_test:
            # Convert degraded filename to clean filename
            # rain-xxx.png -> rain_clean-xxx.png
            # snow-xxx.png -> snow_clean-xxx.png
            if degraded_filename.startswith('rain-'):
                clean_filename = degraded_filename.replace(
                    'rain-', 'rain_clean-'
                )
            elif degraded_filename.startswith('snow-'):
                clean_filename = degraded_filename.replace(
                    'snow-', 'snow_clean-'
                )
            else:
                raise ValueError(
                    f"Unknown degraded file format: {degraded_filename}"
                )
            
            clean_path = os.path.join(self.clean_dir, clean_filename)
            clean_img = Image.open(clean_path).convert('RGB')
        
        if self.transform:
            degraded_img = self.transform(degraded_img)
            if not self.is_test:
                clean_img = self.transform(clean_img)
        
        if self.is_test:
            return degraded_img, self.degraded_files[idx]
        else:
            return degraded_img, clean_img


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    
    Implements channel attention mechanism using global average pooling
    and max pooling to capture inter-channel dependencies.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initialize Channel Attention module.
        
        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Channel Attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Implements spatial attention mechanism to focus on important
    spatial locations in the feature map.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize Spatial Attention module.
        
        Args:
            kernel_size: Kernel size for the convolution
        """
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size//2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Spatial Attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(x_cat)
        return self.sigmoid(attention)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Combines channel and spatial attention mechanisms for enhanced
    feature representation.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initialize CBAM module.
        
        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio for channel attention
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CBAM.
        
        Args:
            x: Input tensor
            
        Returns:
            Enhanced feature tensor
        """
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block with CBAM attention.
    
    Implements a residual block with batch normalization, CBAM attention,
    and skip connections.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize Residual Block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Residual Block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        return out


class PromptGenBlock(nn.Module):
    """
    Prompt Generation Block.
    
    Generates adaptive prompts based on input features, inspired by PromptIR.
    """
    
    def __init__(self, 
                 prompt_dim: int = 64, 
                 prompt_len: int = 5, 
                 prompt_size: int = 32, 
                 lin_dim: int = 64):
        """
        Initialize Prompt Generation Block.
        
        Args:
            prompt_dim: Dimension of prompt features
            prompt_len: Number of prompt parameters
            prompt_size: Spatial size of prompt
            lin_dim: Dimension for linear layer
        """
        super(PromptGenBlock, self).__init__()
        # Learnable prompt parameters for different degradation types
        self.prompt_param = nn.Parameter(torch.randn(
            1, prompt_len, prompt_dim, prompt_size, prompt_size
        ))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, 
            padding=1, bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Prompt Generation Block.
        
        Args:
            x: Input tensor
            
        Returns:
            Generated prompt tensor
        """
        batch_size, channels, height, width = x.shape
        # Global average pooling to get feature representation
        emb = x.mean(dim=(-2, -1))  # B x C
        
        # Generate prompt weights based on input features
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        
        # Weight and combine prompts
        prompt_weights_expanded = prompt_weights.unsqueeze(-1).unsqueeze(
            -1
        ).unsqueeze(-1)
        prompt_param_repeated = self.prompt_param.repeat(
            batch_size, 1, 1, 1, 1
        )
        prompt = prompt_weights_expanded * prompt_param_repeated
        prompt = torch.sum(prompt, dim=1)
        
        # Resize prompt to match input spatial dimensions
        prompt = F.interpolate(
            prompt, (height, width), mode="bilinear", align_corners=False
        )
        prompt = self.conv3x3(prompt)
        
        return prompt


class EnhancedPromptModule(nn.Module):
    """
    Enhanced Prompt Module.
    
    Combines prompt generation with gated fusion mechanism for
    degradation-aware feature processing.
    """
    
    def __init__(self, dim: int = 64, prompt_len: int = 5):
        """
        Initialize Enhanced Prompt Module.
        
        Args:
            dim: Feature dimension
            prompt_len: Length of prompt parameters
        """
        super(EnhancedPromptModule, self).__init__()
        self.prompt_gen = PromptGenBlock(
            prompt_dim=dim, 
            prompt_len=prompt_len, 
            prompt_size=32, 
            lin_dim=dim
        )
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.gate = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Enhanced Prompt Module.
        
        Args:
            x: Input tensor
            
        Returns:
            Enhanced feature tensor with prompt guidance
        """
        # Generate degradation-specific prompt
        prompt = self.prompt_gen(x)
        
        # Concatenate and fuse features
        fused = torch.cat([x, prompt], dim=1)
        fused = self.fusion_conv(fused)
        
        # Apply gating mechanism
        gate_weights = self.gate(fused)
        
        return x + gate_weights * fused


class EnhancedUNet(nn.Module):
    """
    Enhanced U-Net with PromptIR-inspired components.
    
    Implements an enhanced U-Net architecture with CBAM attention,
    prompt learning modules, and multi-scale feature fusion.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        """
        Initialize Enhanced U-Net.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(EnhancedUNet, self).__init__()
        
        # Initial feature extraction
        self.patch_embed = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Encoder
        self.enc1 = ResidualBlock(64, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)
        
        # Prompt modules for different scales (PromptIR-inspired)
        self.prompt1 = EnhancedPromptModule(64, prompt_len=5)
        self.prompt2 = EnhancedPromptModule(128, prompt_len=5)
        self.prompt3 = EnhancedPromptModule(256, prompt_len=5)
        self.prompt4 = EnhancedPromptModule(512, prompt_len=5)
        
        # Bottleneck with enhanced processing
        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 1024),
            ResidualBlock(1024, 1024)
        )
        
        # Decoder with feature fusion
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.fusion4 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.dec4 = ResidualBlock(512, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.fusion3 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.dec3 = ResidualBlock(256, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.fusion2 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.dec2 = ResidualBlock(128, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.fusion1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.dec1 = ResidualBlock(64, 64)
        
        # Refinement layers (inspired by PromptIR)
        self.refinement = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        
        # Output layer with residual connection
        self.final_conv = nn.Conv2d(
            64, out_channels, kernel_size=3, padding=1, bias=False
        )
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Enhanced U-Net.
        
        Args:
            x: Input tensor
            
        Returns:
            Restored image tensor
        """
        # Store input for residual connection
        inp_img = x
        
        # Initial feature extraction
        x = self.patch_embed(x)
        
        # Encoder path with prompts
        e1 = self.enc1(x)
        e1_prompted = self.prompt1(e1)
        
        e2 = self.enc2(self.pool(e1_prompted))
        e2_prompted = self.prompt2(e2)
        
        e3 = self.enc3(self.pool(e2_prompted))
        e3_prompted = self.prompt3(e3)
        
        e4 = self.enc4(self.pool(e3_prompted))
        e4_prompted = self.prompt4(e4)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4_prompted))
        
        # Decoder path with improved feature fusion
        d4 = self.upconv4(b)
        d4 = self.fusion4(torch.cat([d4, e4_prompted], dim=1))
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = self.fusion3(torch.cat([d3, e3_prompted], dim=1))
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = self.fusion2(torch.cat([d2, e2_prompted], dim=1))
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = self.fusion1(torch.cat([d1, e1_prompted], dim=1))
        d1 = self.dec1(d1)
        
        # Refinement
        d1 = self.refinement(d1)
        
        # Output with residual connection (like PromptIR)
        output = self.final_conv(d1) + inp_img
        
        return torch.clamp(output, 0, 1)


class CombinedLoss(nn.Module):
    """
    Combined Loss Function for comprehensive image quality optimization.
    
    Combines L1 loss, SSIM loss, and perceptual loss for better
    image restoration quality.
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.2, 
                 gamma: float = 0.1):
        """
        Initialize Combined Loss.
        
        Args:
            alpha: Weight for L1 loss
            beta: Weight for SSIM loss
            gamma: Weight for perceptual loss
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # L1 loss weight
        self.beta = beta    # SSIM loss weight
        self.gamma = gamma  # Perceptual loss weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def ssim_loss(self, pred: torch.Tensor, 
                  target: torch.Tensor) -> torch.Tensor:
        """
        SSIM loss for better structural similarity.
        
        Args:
            pred: Predicted image tensor
            target: Target image tensor
            
        Returns:
            SSIM loss value
        """
        def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
            """Generate Gaussian kernel for SSIM calculation."""
            coords = torch.arange(size, dtype=torch.float32)
            coords -= size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g.unsqueeze(0).unsqueeze(0)
        
        def ssim(img1: torch.Tensor, img2: torch.Tensor, 
                 window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
            """Calculate SSIM between two images."""
            channel = img1.size(1)
            window = gaussian_kernel(window_size, sigma)
            window = window.expand(
                channel, 1, window_size, window_size
            ).to(img1.device)
            
            mu1 = F.conv2d(
                img1, window, padding=window_size//2, groups=channel
            )
            mu2 = F.conv2d(
                img2, window, padding=window_size//2, groups=channel
            )
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(
                img1 * img1, window, padding=window_size//2, groups=channel
            ) - mu1_sq
            sigma2_sq = F.conv2d(
                img2 * img2, window, padding=window_size//2, groups=channel
            ) - mu2_sq
            sigma12 = F.conv2d(
                img1 * img2, window, padding=window_size//2, groups=channel
            ) - mu1_mu2
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
                (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            )
            return ssim_map.mean()
        
        return 1 - ssim(pred, target)
    
    def perceptual_loss(self, pred: torch.Tensor, 
                       target: torch.Tensor) -> torch.Tensor:
        """
        Simple perceptual loss using gradient differences.
        
        Args:
            pred: Predicted image tensor
            target: Target image tensor
            
        Returns:
            Perceptual loss value
        """
        # Sobel filters for edge detection
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=torch.float32
        ).view(1, 1, 3, 3).to(pred.device)
        
        # Convert to grayscale for edge detection
        pred_gray = (0.299 * pred[:, 0:1] + 
                    0.587 * pred[:, 1:2] + 
                    0.114 * pred[:, 2:3])
        target_gray = (0.299 * target[:, 0:1] + 
                      0.587 * target[:, 1:2] + 
                      0.114 * target[:, 2:3])
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)
        
        # L1 loss on gradients
        grad_loss = (self.l1_loss(pred_grad_x, target_grad_x) + 
                    self.l1_loss(pred_grad_y, target_grad_y))
        return grad_loss
    
    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Combined Loss.
        
        Args:
            pred: Predicted image tensor
            target: Target image tensor
            
        Returns:
            Combined loss value
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = self.alpha * l1 + self.beta * ssim + self.gamma * perceptual
        return total_loss


def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                num_epochs: int, 
                device: torch.device) -> Tuple[List[float], List[float]]:
    """
    Train the image restoration model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        Tuple of (train_losses, val_psnrs)
    """
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    best_psnr = 0
    train_losses = []
    val_psnrs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for degraded, clean in tqdm(
            train_loader, 
            desc=f'Epoch {epoch+1}/{num_epochs} - Training'
        ):
            degraded, clean = degraded.to(device), clean.to(device)
            
            optimizer.zero_grad()
            outputs = model(degraded)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_psnr = 0
        
        with torch.no_grad():
            for degraded, clean in tqdm(
                val_loader, 
                desc=f'Epoch {epoch+1}/{num_epochs} - Validation'
            ):
                degraded, clean = degraded.to(device), clean.to(device)
                outputs = model(degraded)
                
                # Calculate PSNR for each image in the batch
                for i in range(outputs.size(0)):
                    psnr = calculate_psnr(outputs[i], clean[i])
                    val_psnr += psnr.item()
        
        avg_val_psnr = val_psnr / len(val_loader.dataset)
        val_psnrs.append(avg_val_psnr)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val PSNR: {avg_val_psnr:.4f}')
        
        # Save best model
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  New best model saved! PSNR: {best_psnr:.4f}')
        
        scheduler.step(avg_val_psnr)
        print()
    
    return train_losses, val_psnrs


def main() -> None:
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Enhanced data transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(256),  # Random crop for spatial diversity
        transforms.RandomHorizontalFlip(0.5),  # Horizontal flip
        transforms.RandomVerticalFlip(0.3),    # Vertical flip
        transforms.RandomRotation(10),         # Small rotation
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        ),  # Color augmentation
        transforms.ToTensor(),
    ])
    
    # Simple transform for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create datasets with proper paths
    train_dataset_full = ImageRestorationDataset(
        degraded_dir='hw4_realse_dataset/train/degraded',
        clean_dir='hw4_realse_dataset/train/clean',
        transform=train_transform
    )
    
    # Create train/validation split (80/20)
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(train_dataset_full)), [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create separate datasets for train and validation with different transforms
    train_dataset = torch.utils.data.Subset(
        train_dataset_full, train_indices.indices
    )
    
    val_dataset = ImageRestorationDataset(
        degraded_dir='hw4_realse_dataset/train/degraded',
        clean_dir='hw4_realse_dataset/train/clean',
        transform=val_transform
    )
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Total images per type: {len(train_dataset) + len(val_dataset)}')
    
    # Initialize model
    model = EnhancedUNet().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Train model
    print("Starting training...")
    train_losses, val_psnrs = train_model(
        model, train_loader, val_loader, num_epochs=100, device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_psnrs)
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print("Training completed!")
    print(f"Best validation PSNR: {max(val_psnrs):.4f} dB")


def edge_enhancement(image: Union[torch.Tensor, np.ndarray], 
                    strength: float = 0.3) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply edge enhancement using unsharp masking.
    
    Args:
        image: Input image
        strength: Enhancement strength
        
    Returns:
        Enhanced image
    """
    # Convert to numpy if tensor
    if torch.is_tensor(image):
        image_np = image.cpu().numpy()
        was_tensor = True
    else:
        image_np = image
        was_tensor = False
    
    # Apply Gaussian blur
    blurred = gaussian_filter(image_np, sigma=1.0)
    
    # Unsharp masking
    enhanced = image_np + strength * (image_np - blurred)
    enhanced = np.clip(enhanced, 0, 1)
    
    if was_tensor:
        return torch.from_numpy(enhanced).to(image.device)
    return enhanced


def color_correction(image: Union[torch.Tensor, np.ndarray], 
                    gamma: float = 1.1) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply gamma correction for better color balance.
    
    Args:
        image: Input image
        gamma: Gamma value for correction
        
    Returns:
        Color corrected image
    """
    if torch.is_tensor(image):
        corrected = torch.pow(image, 1.0/gamma)
    else:
        corrected = np.power(image, 1.0/gamma)
    return corrected


def noise_reduction(image: Union[torch.Tensor, np.ndarray], 
                   strength: float = 0.1) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply light noise reduction.
    
    Args:
        image: Input image
        strength: Noise reduction strength
        
    Returns:
        Denoised image
    """
    if torch.is_tensor(image):
        image_np = image.cpu().numpy()
        was_tensor = True
    else:
        image_np = image
        was_tensor = False
    
    smoothed = gaussian_filter(image_np, sigma=0.5)
    result = (1 - strength) * image_np + strength * smoothed
    result = np.clip(result, 0, 1)
    
    if was_tensor:
        return torch.from_numpy(result).to(image.device)
    return result


def post_process_image(image: Union[torch.Tensor, np.ndarray]
                      ) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply complete post-processing pipeline.
    
    Args:
        image: Input image
        
    Returns:
        Post-processed image
    """
    # Edge enhancement
    enhanced = edge_enhancement(image, strength=0.3)
    
    # Color correction
    corrected = color_correction(enhanced, gamma=1.1)
    
    # Light noise reduction
    final = noise_reduction(corrected, strength=0.1)
    
    return final


def test_time_augmentation(model: nn.Module, 
                          image: torch.Tensor, 
                          device: torch.device) -> torch.Tensor:
    """
    Apply test time augmentation for better predictions.
    
    Args:
        model: Trained model
        image: Input image tensor
        device: Device to run on
        
    Returns:
        Averaged prediction tensor
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original image
        pred1 = model(image)
        predictions.append(pred1)
        
        # Horizontal flip
        image_hflip = torch.flip(image, [3])
        pred2 = model(image_hflip)
        pred2 = torch.flip(pred2, [3])  # Flip back
        predictions.append(pred2)
        
        # Vertical flip
        image_vflip = torch.flip(image, [2])
        pred3 = model(image_vflip)
        pred3 = torch.flip(pred3, [2])  # Flip back
        predictions.append(pred3)
        
        # Both flips
        image_hvflip = torch.flip(torch.flip(image, [2]), [3])
        pred4 = model(image_hvflip)
        pred4 = torch.flip(torch.flip(pred4, [2]), [3])  # Flip back
        predictions.append(pred4)
    
    # Average all predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction


def generate_predictions_with_tta(model_path: str, 
                                 test_dir: str, 
                                 output_file: str, 
                                 use_tta: bool = True, 
                                 use_post_processing: bool = True) -> None:
    """
    Generate predictions with TTA and post-processing.
    
    Args:
        model_path: Path to trained model
        test_dir: Directory containing test images
        output_file: Output file path for predictions
        use_tta: Whether to use test time augmentation
        use_post_processing: Whether to apply post-processing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = EnhancedUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Data transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Test dataset
    test_dataset = ImageRestorationDataset(
        degraded_dir=test_dir,
        clean_dir=None,
        transform=transform,
        is_test=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f'Test dataset size: {len(test_dataset)}')
    
    predictions = {}
    
    for degraded, filename in tqdm(
        test_loader, 
        desc='Generating predictions with enhancements'
    ):
        degraded = degraded.to(device)
        
        if use_tta:
            # Use test time augmentation
            output = test_time_augmentation(model, degraded, device)
        else:
            # Standard prediction
            with torch.no_grad():
                output = model(degraded)
                
        if use_post_processing:
            # Apply post-processing
            output = post_process_image(output)
        
        # Convert to numpy and scale to 0-255
        output_np = output.cpu().numpy()[0]  # Remove batch dimension
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        
        predictions[filename[0]] = output_np
    
    # Save predictions
    np.savez(output_file, **predictions)
    print(f'Predictions saved to {output_file}')
    print(f'Number of predictions: {len(predictions)}')
    
    # Verify the saved file
    loaded = np.load(output_file)
    print(f'Verification - Keys in saved file: {len(loaded.files)}')
    print(f'Sample prediction shape: {loaded[loaded.files[0]].shape}')


def run_complete_workflow() -> None:
    """Run the complete training and inference workflow."""
    print("="*50)
    print("HW4 Image Restoration - Complete Workflow")
    print("="*50)
    
    # Step 1: Train the model
    print("\nStep 1: Training the model...")
    main()
    
    # Step 2: Generate predictions
    print("\nStep 2: Generating predictions...")
    generate_predictions_with_tta(
        model_path='best_model.pth',
        test_dir='hw4_realse_dataset/test/degraded',
        output_file='pred.npz'
    )
    
    print("\nWorkflow completed successfully!")
    print("Files generated:")
    print("- best_model.pth: Trained model weights")
    print("- pred.npz: Test predictions for submission")
    print("- training_curves.png: Training visualization")


if __name__ == '__main__':
    # You can run the complete workflow or individual steps
    
    # Option 1: Run complete workflow (training + inference)
    run_complete_workflow()
    
    # Option 2: Only training
    # main()
    
    # Option 3: Only generate predictions (if model is already trained)
    # generate_predictions_with_tta(
    #     model_path='best_model.pth',
    #     test_dir='hw4_realse_dataset/test/degraded',
    #     output_file='pred.npz'
    # )