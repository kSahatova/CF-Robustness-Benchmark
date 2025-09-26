"""
This network is built on top of SNGAN network implementation.
Translated from TensorFlow to PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from blocks import ConditionalBatchNorm2d


class DResBlock(nn.Module):
    """Discriminator Residual Block - matches TensorFlow implementation exactly"""
    def __init__(self, in_channels, out_channels, downsample=True, use_spectral_norm=True):
        super().__init__()
        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        conv_layer = spectral_norm if use_spectral_norm else lambda x: x
        
        self.conv1 = conv_layer(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        self.conv2 = conv_layer(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True))
        
        # Identity/shortcut connection - only create if needed
        if downsample:
            self.identity_conv = conv_layer(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True))
            
    def forward(self, x):
        # Store input for residual connection
        temp = x
        
        # Main path: ReLU -> Conv -> ReLU -> Conv -> (optional downsample)
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2, 2)  # downsampling after 2nd conv in D
            # Identity mapping with 1x1 conv
            temp = self.identity_conv(temp)
            temp = F.avg_pool2d(temp, 2, 2)
        
        return h + temp


class DFirstResBlock(nn.Module):
    """First Discriminator Residual Block - matches TensorFlow implementation exactly"""
    def __init__(self, in_channels, out_channels, use_spectral_norm=True):
        super().__init__()
        
        conv_layer = spectral_norm if use_spectral_norm else lambda x: x
        
        self.conv1 = conv_layer(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        self.conv2 = conv_layer(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True))
        self.identity_conv = conv_layer(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True))
            
    def forward(self, x):
        # Store input for residual connection
        temp = x
        
        # Main path: Conv -> ReLU -> Conv -> Downsample
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2, 2)
        
        # Identity mapping: Downsample -> 1x1 Conv
        temp = F.avg_pool2d(temp, 2, 2)
        temp = self.identity_conv(temp)
            
        return h + temp


class Discriminator_Ordinal(nn.Module):
    def __init__(self, num_classes, use_spectral_norm=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Discriminator blocks - following the exact TensorFlow structure
        self.block1 = DFirstResBlock(1, 64, use_spectral_norm)  # [n, 14, 14, 64]
        self.block2 = DResBlock(64, 128, downsample=True, use_spectral_norm=use_spectral_norm)   # [n, 7, 7, 128]
        self.block3 = DResBlock(128, 256, downsample=True, use_spectral_norm=use_spectral_norm)  # [n, 3, 3, 256]
        self.block4 = DResBlock(256, 512, downsample=True, use_spectral_norm=use_spectral_norm)  # [n, 1, 1, 512] (approx)
        # Note: Adjusted for 28x28 -> fewer blocks needed
        
        # Final layers
        linear_layer = spectral_norm if use_spectral_norm else lambda x: x
        self.final_linear = linear_layer(nn.Linear(512, 1))
        
        # Inner product layers for ordinal classification
        # Each creates a projection similar to dense layer with 2 outputs
        self.inner_product_layers = nn.ModuleList([
            linear_layer(nn.Linear(512, 2)) for _ in range(num_classes - 1)
        ])
        
    def forward(self, x, y):
        # Discriminator forward pass - exact sequence from TensorFlow
        x = self.block1(x)   # DFirstResblock
        x = self.block2(x)   # D_Resblock with downsample
        x = self.block3(x)   # D_Resblock with downsample  
        x = self.block4(x)   # D_Resblock with downsample
        # Note: Removed blocks 5&6 for 28x28 input
        
        x = F.relu(x)  # ReLU before global pooling
        
        # Global sum pooling (equivalent to tf's global_sum_pooling)
        x = torch.sum(x, dim=[2, 3])  # [n, 512]
        
        # Ordinal classification part - matching TensorFlow logic exactly
        temp = None
        for i in range(self.num_classes - 1):
            # Inner_product equivalent - project features and multiply by class labels
            projection = self.inner_product_layers[i](x)  # [n, 2]
            # Extract the specific class label y[:,i+1] 
            class_weight = y[:, i + 1].unsqueeze(-1)  # [n, 1]
            inner_prod_result = torch.sum(projection * class_weight.unsqueeze(-1), dim=-1)  # [n]
            
            if i == 0:
                temp = inner_prod_result
            else:
                temp = temp + inner_prod_result
        
        # Final discriminator output (dense layer)
        final_output = self.final_linear(x).squeeze()  # [n]
        
        # Combine ordinal and discriminator outputs
        return temp + final_output


# Example usage:
if __name__ == "__main__":
    # Initialize models for 28x28 grayscale images
    num_classes = 10  # e.g., MNIST has 10 classes
    # generator = Generator_Encoder_Decoder(num_classes=num_classes, num_channels=1)  # 1 for grayscale
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
    d_output = discriminator(generated.detach(), y_ordinal)  # detach to avoid generator gradients
    print(f"Discriminator output shape: {d_output.shape}")  # Should be [4]
    
    # Print model parameter counts
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    # Test discriminator with proper ordinal labels format
    # For ordinal regression, y_ordinal should be in cumulative format
    # e.g., for class 3 out of 10: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    ordinal_labels = torch.zeros(batch_size, num_classes)
    class_labels = torch.randint(0, num_classes, (batch_size,))
    for i in range(batch_size):
        ordinal_labels[i, :class_labels[i]] = 1
    
    d_output_proper = discriminator(generated.detach(), ordinal_labels)
    print(f"Discriminator output with proper ordinal labels: {d_output_proper.shape}")