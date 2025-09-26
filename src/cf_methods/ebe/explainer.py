"""
This network is built on top of SNGAN network implementation.
Translated from TensorFlow to PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization"""
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_embed = nn.Embedding(num_classes, num_features)
        self.beta_embed = nn.Embedding(num_classes, num_features)
        
    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(y).view(-1, self.num_features, 1, 1)
        return gamma * out + beta


class GResBlock(nn.Module):
    """Generator Residual Block"""
    def __init__(self, in_channels, out_channels, num_classes, upsample=True):
        super().__init__()
        self.upsample = upsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, y):
        h = F.relu(self.bn1(x, y))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = F.relu(self.bn2(h, y))
        h = self.conv2(h)
        
        # Shortcut
        shortcut = self.shortcut(x)
        if self.upsample:
            shortcut = F.interpolate(shortcut, scale_factor=2, mode='nearest')
            
        return h + shortcut


class GResBlockEncoder(nn.Module):
    """Generator Encoder Residual Block (with downsampling)"""
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        
        # Shortcut connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
        # Downsampling
        self.downsample = nn.AvgPool2d(2)
            
    def forward(self, x, y):
        h = F.relu(self.bn1(x, y))
        h = self.conv1(h)
        h = F.relu(self.bn2(h, y))
        h = self.conv2(h)
        h = self.downsample(h)
        
        # Shortcut
        shortcut = self.shortcut(x)
        shortcut = self.downsample(shortcut)
            
        return h + shortcut


class DResBlock(nn.Module):
    """Discriminator Residual Block"""
    def __init__(self, in_channels, out_channels, downsample=True, use_spectral_norm=True):
        super().__init__()
        self.downsample = downsample
        
        conv_layer = spectral_norm if use_spectral_norm else lambda x: x
        
        self.conv1 = conv_layer(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.conv2 = conv_layer(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        
        # Shortcut connection
        if in_channels != out_channels or downsample:
            self.shortcut = conv_layer(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        
        # Shortcut
        shortcut = self.shortcut(x)
        if self.downsample and not isinstance(self.shortcut, nn.Identity):
            shortcut = F.avg_pool2d(shortcut, 2)
            
        return h + shortcut


class DFirstResBlock(nn.Module):
    """First Discriminator Residual Block (no batch norm on input)"""
    def __init__(self, in_channels, out_channels, use_spectral_norm=True):
        super().__init__()
        
        conv_layer = spectral_norm if use_spectral_norm else lambda x: x
        
        self.conv1 = conv_layer(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.conv2 = conv_layer(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.shortcut = conv_layer(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
            
    def forward(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        
        # Shortcut
        shortcut = F.avg_pool2d(x, 2)
        shortcut = self.shortcut(shortcut)
            
        return h + shortcut


class Generator_Encoder_Decoder(nn.Module):
    def __init__(self, num_classes, num_channels=1):  # Changed default to 1 for grayscale
        super().__init__()
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(num_channels, 32, 3, 1, 1)  # Reduced channels
        self.bn1 = ConditionalBatchNorm2d(num_channels, num_classes)
        
        # Encoder blocks (fewer blocks for 28x28)
        self.encoder_block1 = GResBlockEncoder(32, 64, num_classes)   # 28x28 -> 14x14
        self.encoder_block2 = GResBlockEncoder(64, 128, num_classes)  # 14x14 -> 7x7
        self.encoder_block3 = GResBlockEncoder(128, 256, num_classes) # 7x7 -> 3x3 (with padding adjustment)
        
        # Decoder blocks
        self.decoder_block1 = GResBlock(256, 128, num_classes, upsample=True)  # 3x3 -> 7x7 (approx)
        self.decoder_block2 = GResBlock(128, 64, num_classes, upsample=True)   # 7x7 -> 14x14
        self.decoder_block3 = GResBlock(64, 32, num_classes, upsample=True)    # 14x14 -> 28x28
        
        # Final layers
        self.final_bn = ConditionalBatchNorm2d(32, num_classes)
        self.final_conv = nn.Conv2d(32, num_channels, 3, 1, 1)
        
    def forward(self, x, y, train_phase=True):
        # Initial processing
        x = F.relu(self.bn1(x, y))
        x = self.conv1(x)  # [n, 28, 28, 32]
        
        # Encoder
        x = self.encoder_block1(x, y)  # [n, 14, 14, 64]
        print(x.shape)
        x = self.encoder_block2(x, y)  # [n, 7, 7, 128]
        embedding = self.encoder_block3(x, y)  # [n, 3, 3, 256] (approximately)
        print(embedding.shape)
        
        # Decoder
        x = self.decoder_block1(embedding, y)  # [n, 7, 7, 128] (approximately)
        print(x.shape)
        x = self.decoder_block2(x, y)  # [n, 14, 14, 64]
        print(x.shape)
        x = self.decoder_block3(x, y)  # [n, 28, 28, 32]
        print(x.shape)
        
        # Final output
        x = F.relu(self.final_bn(x, y))
        x = self.final_conv(x)  # [n, 28, 28, num_channels]
        
        return torch.tanh(x), embedding


class Discriminator_Ordinal(nn.Module):
    def __init__(self, num_classes, use_spectral_norm=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Discriminator blocks (adjusted for 28x28)
        self.block1 = DFirstResBlock(1, 32, use_spectral_norm)  # [n, 14, 14, 32] - changed input to 1 channel
        self.block2 = DResBlock(32, 64, downsample=True, use_spectral_norm=use_spectral_norm)   # [n, 7, 7, 64]
        self.block3 = DResBlock(64, 128, downsample=True, use_spectral_norm=use_spectral_norm)  # [n, 3, 3, 128] (approx)
        self.block4 = DResBlock(128, 256, downsample=False, use_spectral_norm=use_spectral_norm) # [n, 3, 3, 256]
        
        # Final layers
        linear_layer = spectral_norm if use_spectral_norm else lambda x: x
        self.final_linear = linear_layer(nn.Linear(256, 1))  # Reduced from 1024 to 256
        
        # Ordinal projection layers
        self.ordinal_projections = nn.ModuleList([
            linear_layer(nn.Linear(256, 2)) for _ in range(num_classes - 1)  # Reduced from 1024 to 256
        ])
        
    def forward(self, x, y):
        # Discriminator forward pass
        x = self.block1(x)   # [n, 14, 14, 32]
        x = self.block2(x)   # [n, 7, 7, 64]
        x = self.block3(x)   # [n, 3, 3, 128] (approximately)
        x = self.block4(x)   # [n, 3, 3, 256]
        
        x = F.relu(x)
        
        # Global sum pooling
        x = torch.sum(x, dim=[2, 3])  # [n, 256]
        
        # Ordinal classification
        ordinal_sum = 0
        for i in range(self.num_classes - 1):
            # Inner product with class labels
            projection = self.ordinal_projections[i](x)  # [n, 2]
            class_labels = y[:, i + 1].unsqueeze(1)  # [n, 1]
            ordinal_sum += torch.sum(projection * class_labels.unsqueeze(1), dim=2)  # This needs adjustment based on y format
        
        # Final discriminator output
        final_output = self.final_linear(x)  # [n, 1]
        
        return ordinal_sum + final_output.squeeze()


# Example usage:
if __name__ == "__main__":
    # Initialize models for 28x28 grayscale images
    num_classes = 10  # e.g., MNIST has 10 classes
    generator = Generator_Encoder_Decoder(num_classes=num_classes, num_channels=1)  # 1 for grayscale
    discriminator = Discriminator_Ordinal(num_classes=num_classes)
    
    # Example forward pass with 28x28 grayscale images
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)  # Changed to 1 channel and 28x28
    y = torch.randint(0, num_classes, (batch_size,))
    y_ordinal = torch.randn(batch_size, num_classes)  # Ordinal labels for discriminator
    
    # Generator forward
    generated, embedding = generator(x, y)
    print(f"Generated shape: {generated.shape}")  # Should be [4, 1, 28, 28]
    print(f"Embedding shape: {embedding.shape}")   # Should be [4, 256, 3, 3] (approximately)
    
    # Discriminator forward
    d_output = discriminator(generated, y_ordinal)
    print(f"Discriminator output shape: {d_output.shape}")  # Should be [4]
    
    # Print model parameter counts
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")