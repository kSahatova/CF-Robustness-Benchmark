import torch 
from torch import nn
import torch.nn.functional as F
from blocks import ConditionalBatchNorm2d


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
        x = self.encoder_block2(x, y)  # [n, 7, 7, 128]
        embedding = self.encoder_block3(x, y)  # [n, 3, 3, 256] (approximately)
        
        # Decoder
        x = self.decoder_block1(embedding, y)  # [n, 7, 7, 128] (approximately)
        x = self.decoder_block2(x, y)  # [n, 14, 14, 64]
        x = self.decoder_block3(x, y)  # [n, 28, 28, 32]
        
        # Final output
        x = F.relu(self.final_bn(x, y))
        x = self.final_conv(x)  # [n, 28, 28, num_channels]
        
        return torch.tanh(x), embedding