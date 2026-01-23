import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - reduces parameters significantly"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                    padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Ensure reduced channels is at least 4 to avoid degenerate FC layers
        reduced_channels = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (lightweight version)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class AttentionGate(nn.Module):
    """Attention Gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # g: gating signal (from decoder)
        # x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 if sizes don't match
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class LightweightEncoderBlock(nn.Module):
    """Lightweight encoder block with depthwise separable convolutions"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        # Use reduction that ensures at least 4 channels in attention
        reduction = max(1, out_channels // 4)
        self.cbam = CBAM(out_channels, reduction=reduction)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        
        # Resize residual if needed
        if residual.shape[2:] != x.shape[2:]:
            residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return x + residual


class LightweightDecoderBlock(nn.Module):
    """Lightweight decoder block with attention gate"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attention_gate = AttentionGate(in_channels, skip_channels, out_channels // 2)
        self.conv1 = DepthwiseSeparableConv(in_channels + skip_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Apply attention gate to skip connection
        skip = self.attention_gate(x, skip)
        
        # Ensure sizes match before concatenation
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LightweightAttentionUNet(nn.Module):
    """
    Lightweight Attention U-Net for ECG Image Digitization
    
    Features:
    - Depthwise separable convolutions (reduces params by ~8-9x per layer)
    - CBAM attention in encoder
    - Attention gates in skip connections
    - Residual connections for better gradient flow
    
    Approximate params: ~1.5-2M (vs ~30M for standard U-Net)
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super().__init__()
        
        features = [base_features, base_features*2, base_features*4, base_features*8]
        # features = [32, 64, 128, 256]
        
        # Encoder
        self.enc1 = LightweightEncoderBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = LightweightEncoderBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = LightweightEncoderBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = LightweightEncoderBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(features[3], features[3]*2),
            DepthwiseSeparableConv(features[3]*2, features[3]*2),
            CBAM(features[3]*2, reduction=16)
        )
        
        # Decoder
        self.dec4 = LightweightDecoderBlock(features[3]*2, features[3], features[3])
        self.dec3 = LightweightDecoderBlock(features[3], features[2], features[2])
        self.dec2 = LightweightDecoderBlock(features[2], features[1], features[1])
        self.dec1 = LightweightDecoderBlock(features[1], features[0], features[0])
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(features[0], out_channels, 1),
            nn.Tanh()  # For ECG: pixel values between -1 and 1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder path with attention-gated skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        return self.output(d1)


class SimpleEncoderBlock(nn.Module):
    """Simple encoder block with two convolutions"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SimpleDecoderBlock(nn.Module):
    """Simple decoder block with upsampling and convolutions"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Ensure sizes match before concatenation
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SimpleUNet(nn.Module):
    """
    Simple U-Net for ECG Image Digitization
    
    A much simpler and easier to train U-Net architecture without attention mechanisms
    or depthwise separable convolutions. This model uses standard convolutions and
    a straightforward encoder-decoder structure.
    
    Features:
    - Standard 2D convolutions
    - Batch normalization
    - ReLU activations
    - Skip connections
    - ~4-5M parameters (easier to train than lightweight version)
    
    This model is recommended for:
    - Faster training and convergence
    - Better initial results
    - Simpler debugging
    - Less GPU memory requirements
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        
        features = [base_features, base_features*2, base_features*4, base_features*8]
        # features = [64, 128, 256, 512]
        
        # Encoder
        self.enc1 = SimpleEncoderBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = SimpleEncoderBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = SimpleEncoderBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = SimpleEncoderBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = SimpleEncoderBlock(features[3], features[3]*2)
        
        # Decoder
        self.dec4 = SimpleDecoderBlock(features[3]*2, features[3], features[3])
        self.dec3 = SimpleDecoderBlock(features[3], features[2], features[2])
        self.dec2 = SimpleDecoderBlock(features[2], features[1], features[1])
        self.dec1 = SimpleDecoderBlock(features[1], features[0], features[0])
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(features[0], out_channels, 1),
            nn.Tanh()  # For ECG: pixel values between -1 and 1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder path with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        return self.output(d1)