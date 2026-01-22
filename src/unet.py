import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualConvBlock(nn.Module):
    """Residual convolution block with optional attention"""
    def __init__(self, in_channels, out_channels, use_attention=True, dropout=0.1):
        super().__init__()
        self.use_attention = use_attention
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Residual connection (1x1 conv if channels change)
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        
        # Attention module
        if use_attention:
            self.attention = CBAM(out_channels)

    def forward(self, x):
        residual = self.residual(x)
        
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out = out + residual
        out = self.activation(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        return out


class DoubleConv(nn.Module):
    """Two consecutive convolution layers with batch norm and LeakyReLU"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """
    Improved UNet architecture with:
    - Attention gates for skip connections
    - CBAM attention in encoder blocks
    - Residual connections
    - Dropout for regularization
    - Tanh output activation for bounded outputs
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=64, dropout=0.1):
        super().__init__()
        
        features = base_features
        
        # Encoder (downsampling path) with residual blocks and attention
        self.enc1 = ResidualConvBlock(in_channels, features, use_attention=True, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResidualConvBlock(features, features * 2, use_attention=True, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResidualConvBlock(features * 2, features * 4, use_attention=True, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ResidualConvBlock(features * 4, features * 8, use_attention=True, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck with higher dropout for regularization
        self.bottleneck = ResidualConvBlock(features * 8, features * 16, use_attention=True, dropout=dropout * 2)
        
        # Decoder (upsampling path) with attention gates
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(features * 8, features * 8, features * 4)
        self.dec4 = DoubleConv(features * 16, features * 8, dropout=dropout)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(features * 4, features * 4, features * 2)
        self.dec3 = DoubleConv(features * 8, features * 4, dropout=dropout)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(features * 2, features * 2, features)
        self.dec2 = DoubleConv(features * 4, features * 2, dropout=dropout)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.att1 = AttentionGate(features, features, features // 2)
        self.dec1 = DoubleConv(features * 2, features, dropout=dropout)
        
        # Final output layer with Tanh for bounded outputs matching [-1, 1] normalization
        self.out = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with attention gates on skip connections
        dec4 = self.upconv4(bottleneck)
        enc4_att = self.att4(dec4, enc4)
        dec4 = torch.cat([dec4, enc4_att], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        enc3_att = self.att3(dec3, enc3)
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        enc2_att = self.att2(dec2, enc2)
        dec2 = torch.cat([dec2, enc2_att], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        enc1_att = self.att1(dec1, enc1)
        dec1 = torch.cat([dec1, enc1_att], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


class UNetLite(nn.Module):
    """
    Lighter UNet for faster training/inference.
    Fewer parameters but still effective for ECG extraction.
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32, dropout=0.1):
        super().__init__()
        
        features = base_features
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, features, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(features, features * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(features * 2, features * 4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features * 4, features * 8, dropout=dropout * 2)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features * 8, features * 4, dropout=dropout)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features * 4, features * 2, dropout=dropout)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features * 2, features, dropout=dropout)
        
        # Output with Tanh
        self.out = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)
