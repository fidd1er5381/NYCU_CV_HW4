import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# PSNR calculation function
def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# Dataset class
class ImageRestorationDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, transform=None, is_test=False):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.is_test = is_test
        
        # Get all image files
        if is_test:
            self.degraded_files = sorted([f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            # Get both rain and snow degraded images
            all_files = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.degraded_files = sorted([f for f in all_files if f.startswith(('rain-', 'snow-'))])
    
    def __len__(self):
        return len(self.degraded_files)
    
    def __getitem__(self, idx):
        degraded_filename = self.degraded_files[idx]
        degraded_path = os.path.join(self.degraded_dir, degraded_filename)
        degraded_img = Image.open(degraded_path).convert('RGB')
        
        if not self.is_test:
            # Convert degraded filename to clean filename
            # rain-xxx.png -> rain_clean-xxx.png
            # snow-xxx.png -> snow_clean-xxx.png
            if degraded_filename.startswith('rain-'):
                clean_filename = degraded_filename.replace('rain-', 'rain_clean-')
            elif degraded_filename.startswith('snow-'):
                clean_filename = degraded_filename.replace('snow-', 'snow_clean-')
            else:
                raise ValueError(f"Unknown degraded file format: {degraded_filename}")
            
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

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# CBAM (Convolutional Block Attention Module)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# Residual Block with CBAM
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        return out

# Improved Prompt Learning Module (closer to PromptIR)
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=64, prompt_len=5, prompt_size=32, lin_dim=64):
        super(PromptGenBlock, self).__init__()
        # Learnable prompt parameters for different degradation types
        self.prompt_param = nn.Parameter(torch.randn(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Global average pooling to get feature representation
        emb = x.mean(dim=(-2, -1))  # B x C
        
        # Generate prompt weights based on input features
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)  # B x prompt_len
        
        # Weight and combine prompts
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.repeat(B, 1, 1, 1, 1)
        prompt = torch.sum(prompt, dim=1)  # B x prompt_dim x prompt_size x prompt_size
        
        # Resize prompt to match input spatial dimensions
        prompt = F.interpolate(prompt, (H, W), mode="bilinear", align_corners=False)
        prompt = self.conv3x3(prompt)
        
        return prompt

# Enhanced Prompt Module (replacing the simple PromptModule)
class EnhancedPromptModule(nn.Module):
    def __init__(self, dim=64, prompt_len=5):
        super(EnhancedPromptModule, self).__init__()
        self.prompt_gen = PromptGenBlock(
            prompt_dim=dim, 
            prompt_len=prompt_len, 
            prompt_size=32, 
            lin_dim=dim
        )
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.gate = nn.Sigmoid()
        
    def forward(self, x):
        # Generate degradation-specific prompt
        prompt = self.prompt_gen(x)
        
        # Concatenate and fuse features
        fused = torch.cat([x, prompt], dim=1)
        fused = self.fusion_conv(fused)
        
        # Apply gating mechanism
        gate_weights = self.gate(fused)
        
        return x + gate_weights * fused

# Enhanced U-Net with PromptIR-inspired components
class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(EnhancedUNet, self).__init__()
        
        # Initial feature extraction
        self.patch_embed = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
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
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
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

# Combined Loss Function for better image quality
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # L1 loss weight
        self.beta = beta    # SSIM loss weight
        self.gamma = gamma  # Perceptual loss weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def ssim_loss(self, pred, target):
        """SSIM loss for better structural similarity"""
        def gaussian_kernel(size, sigma):
            coords = torch.arange(size, dtype=torch.float32)
            coords -= size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g.unsqueeze(0).unsqueeze(0)
        
        def ssim(img1, img2, window_size=11, sigma=1.5):
            channel = img1.size(1)
            window = gaussian_kernel(window_size, sigma)
            window = window.expand(channel, 1, window_size, window_size).to(img1.device)
            
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()
        
        return 1 - ssim(pred, target)
    
    def perceptual_loss(self, pred, target):
        """Simple perceptual loss using gradient differences"""
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        
        # Convert to grayscale for edge detection
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)
        
        # L1 loss on gradients
        grad_loss = self.l1_loss(pred_grad_x, target_grad_x) + self.l1_loss(pred_grad_y, target_grad_y)
        return grad_loss
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = self.alpha * l1 + self.beta * ssim + self.gamma * perceptual
        return total_loss

# Training function
def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_psnr = 0
    train_losses = []
    val_psnrs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for degraded, clean in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
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
            for degraded, clean in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
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

# Main training script
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Enhanced data transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(256),  # Random crop for spatial diversity
        transforms.RandomHorizontalFlip(0.5),  # Horizontal flip
        transforms.RandomVerticalFlip(0.3),    # Vertical flip
        transforms.RandomRotation(10),         # Small rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Color augmentation
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
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices.indices)
    
    val_dataset = ImageRestorationDataset(
        degraded_dir='hw4_realse_dataset/train/degraded',
        clean_dir='hw4_realse_dataset/train/clean',
        transform=val_transform
    )
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Total images per type: {len(train_dataset) + len(val_dataset)}')
    
    # Initialize model
    model = EnhancedUNet().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Train model
    print("Starting training...")
    train_losses, val_psnrs = train_model(model, train_loader, val_loader, num_epochs=100, device=device)
    
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

# Post-processing functions for better image quality
def edge_enhancement(image, strength=0.3):
    """Apply edge enhancement using unsharp masking"""
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

def color_correction(image, gamma=1.1):
    """Apply gamma correction for better color balance"""
    if torch.is_tensor(image):
        corrected = torch.pow(image, 1.0/gamma)
    else:
        corrected = np.power(image, 1.0/gamma)
    return corrected

def noise_reduction(image, strength=0.1):
    """Apply light noise reduction"""
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

def post_process_image(image):
    """Apply complete post-processing pipeline"""
    # Edge enhancement
    enhanced = edge_enhancement(image, strength=0.3)
    
    # Color correction
    corrected = color_correction(enhanced, gamma=1.1)
    
    # Light noise reduction
    final = noise_reduction(corrected, strength=0.1)
    
    return final

def test_time_augmentation(model, image, device):
    """Apply test time augmentation for better predictions"""
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

# Enhanced prediction function with TTA and post-processing
def generate_predictions_with_tta(model_path, test_dir, output_file, use_tta=True, use_post_processing=True):
    """Generate predictions with TTA and post-processing"""
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
    
    for degraded, filename in tqdm(test_loader, desc='Generating predictions with enhancements'):
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

# Complete workflow function
def run_complete_workflow():
    """Run the complete training and inference workflow"""
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
    #generate_predictions_with_tta(
    #    model_path='best_model.pth',
    #    test_dir='hw4_realse_dataset/test/degraded',
    #    output_file='pred.npz'
    #)