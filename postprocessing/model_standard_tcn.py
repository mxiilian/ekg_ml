"""
TCN-Repair-Net Modell für EKG-NaN-Imputation.

Input: concat(x_filled, mask) → (24, T)
Output: Residual Δ → (12, T)
Finale Vorhersage: y_hat = x_filled + Δ
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import Optional


class TemporalConvBlock(nn.Module):
    """
    Einzelner TCN-Block mit Dilation.
    
    - Zwei Conv1d-Schichten mit Dilation
    - WeightNorm
    - ReLU
    - Dropout
    - Residual-Verbindung im Block
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Use "same" padding to preserve temporal length.
        padding = ((kernel_size - 1) * dilation) // 2
        
        self.conv1 = weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        ))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        ))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            out: (B, out_channels, T)
        """
        identity = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return out + identity


class TCNRepairNet(nn.Module):
    """
    Temporal Convolutional Network für EKG-Reparatur.
    
    Architektur:
    - Input: (B, 24, T) — concat(x_filled, mask)
    - TCN-Stack mit zunehmenden Dilationen
    - Output: (B, 12, T) — Residual Δ
    - Final prediction: y_hat = x_filled + Δ
    """
    
    def __init__(
        self,
        in_channels: int = 24,  # x_filled (12) + mask (12)
        out_channels: int = 12,  # 12 Leads
        hidden_channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        # TCN blocks mit exponentiellen Dilationen
        self.tcn_blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            block = TemporalConvBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            self.tcn_blocks.append(block)
        
        # Output projection → Residual (12 Leads)
        # Use small random initialization for residuals (not zeros!)
        self.output_proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        nn.init.kaiming_uniform_(self.output_proj.weight, a=0.01)  # He initialization with small scale
        if self.output_proj.bias is not None:
            nn.init.uniform_(self.output_proj.bias, -0.01, 0.01)
    
    def forward(self, x_filled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_filled: (B, 12, T) — EKG mit NaNs als 0 gefüllt
            mask: (B, 12, T) — Binary Mask (1=vorhanden, 0=fehlt)
            
        Returns:
            delta: (B, 12, T) — Geschätzter Residual
        """
        # Input kombinieren: concat(x_filled, mask)
        # x_filled: (B, 12, T)
        # mask: (B, 12, T)
        x = torch.cat([x_filled, mask], dim=1)  # (B, 24, T)
        
        # Input projection
        x = self.input_proj(x)  # (B, hidden_channels, T)
        
        # TCN blocks with residual connection
        residual_x = x
        for block in self.tcn_blocks:
            x = block(x)  # (B, hidden_channels, T)
        
        # Add residual connection from input projection
        x = x + residual_x
        
        # Output projection → residual
        delta = self.output_proj(x)  # (B, 12, T)
        
        return delta
    
    def predict(self, x_filled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Finale Vorhersage: y_hat = x_filled + Δ
        
        Args:
            x_filled: (B, 12, T)
            mask: (B, 12, T)
            
        Returns:
            y_hat: (B, 12, T)
        """
        delta = self.forward(x_filled, mask)
        y_hat = x_filled + delta
        return y_hat


def build_model(
    hidden_channels: int = 64,
    num_blocks: int = 4,
    dropout: float = 0.2,
    device: torch.device = None,
) -> TCNRepairNet:
    """
    Modell-Builder.
    
    Args:
        hidden_channels: Anzahl Hidden Channels in TCN
        num_blocks: Anzahl TCN Blöcke
        dropout: Dropout Rate
        device: torch device
        
    Returns:
        Modell auf device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TCNRepairNet(
        in_channels=24,
        out_channels=12,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        kernel_size=3,
        dropout=dropout,
    )
    
    model = model.to(device)
    
    return model


if __name__ == '__main__':
    # Test-Code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(device=device)
    print(f"Model: {model}")
    print(f"Device: {device}")
    
    # Test Forward Pass
    B, C, T = 2, 12, 1000
    x_filled = torch.randn(B, C, T, device=device)
    mask = torch.randint(0, 2, (B, C, T), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        delta = model(x_filled, mask)
        y_hat = model.predict(x_filled, mask)
    
    print(f"Input shapes: x_filled={x_filled.shape}, mask={mask.shape}")
    print(f"Output shapes: delta={delta.shape}, y_hat={y_hat.shape}")
    print("Test erfolgreich!")
