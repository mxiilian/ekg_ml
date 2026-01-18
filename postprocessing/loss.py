"""
Loss-Funktionen für EKG-NaN-Imputation.

- Masked Loss: Nur auf fehlenden Stellen (mask=0)
- Optional: Kleiner Stabilisierungs-Loss auf vorhandenen Stellen (mask=1)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MaskedMSELoss(nn.Module):
    """
    MSE Loss nur auf Stellen wo mask=0 (fehlende Werte).
    
    Loss = MSE(y_hat[mask==0], y_target[mask==0])
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        y_hat: torch.Tensor,
        y_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_hat: (B, C, T) — Modell-Output
            y_target: (B, C, T) — Ground Truth
            mask: (B, C, T) — 1=vorhanden, 0=fehlt
            
        Returns:
            loss: Scalar
        """
        # Nur auf fehlenden Stellen (mask == 0) und gültigen Targets
        valid_mask = torch.isfinite(y_target) & torch.isfinite(y_hat)
        missing_mask = (mask == 0) & valid_mask
        
        if not missing_mask.any():
            # Keine fehlenden Werte → Loss trotzdem am Graph haengen lassen
            return y_hat.sum() * 0.0
        
        diff = y_hat[missing_mask] - y_target[missing_mask]
        mse = torch.mean(diff ** 2)
        
        return mse


class MaskedHuberLoss(nn.Module):
    """
    Huber Loss (robust gegen Outlier) nur auf fehlenden Stellen.
    
    Huber(x) = { 0.5*x^2 wenn |x| <= delta, delta*(|x|-0.5*delta) sonst }
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=delta, reduction='none')
        self.reduction = reduction
    
    def forward(
        self,
        y_hat: torch.Tensor,
        y_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_hat: (B, C, T)
            y_target: (B, C, T)
            mask: (B, C, T) — 1=vorhanden, 0=fehlt
            
        Returns:
            loss: Scalar
        """
        # Nur auf fehlenden Stellen und gültigen Targets
        valid_mask = torch.isfinite(y_target) & torch.isfinite(y_hat)
        missing_mask = (mask == 0) & valid_mask
        
        if not missing_mask.any():
            # Keine fehlenden Werte → Loss trotzdem am Graph haengen lassen
            return y_hat.sum() * 0.0
        
        loss_per_element = self.huber(y_hat, y_target)
        loss = loss_per_element[missing_mask].mean()
        
        return loss


class CombinedMaskedLoss(nn.Module):
    """
    Kombinierter Loss:
    - Primär: MSE/Huber auf fehlenden Stellen (mask=0)
    - Optional: Kleiner Zusatz-Loss auf vorhandenen Stellen (mask=1)
    
    Loss = w_missing * L_missing + w_present * L_present
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',  # 'mse' oder 'huber'
        w_missing: float = 1.0,
        w_present: float = 0.01,  # Sehr klein, nur zur Stabilisierung
        delta: float = 1.0,  # Für Huber Loss
    ):
        super().__init__()
        
        self.w_missing = w_missing
        self.w_present = w_present
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.primary_loss = MaskedMSELoss()
        elif loss_type == 'huber':
            self.primary_loss = MaskedHuberLoss(delta=delta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        y_hat: torch.Tensor,
        y_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            y_hat: (B, C, T)
            y_target: (B, C, T)
            mask: (B, C, T) — 1=vorhanden, 0=fehlt
            
        Returns:
            total_loss: Scalar
            loss_dict: Dict mit Breakdown
        """
        # Primärer Loss auf fehlenden Stellen
        loss_missing = self.primary_loss(y_hat, y_target, mask)
        
        # Sekundärer Loss auf vorhandenen Stellen (optional)
        valid_mask = torch.isfinite(y_target) & torch.isfinite(y_hat)
        present_mask = (mask == 1) & valid_mask
        if present_mask.any() and self.w_present > 0:
            loss_per_element = self.mse_loss(y_hat, y_target)
            loss_present = loss_per_element[present_mask].mean()
        else:
            # Kein present loss → trotzdem am Graph haengen lassen
            loss_present = y_hat.sum() * 0.0
        
        total_loss = self.w_missing * loss_missing + self.w_present * loss_present
        
        loss_dict = {
            'loss_missing': float(loss_missing.item()),
            'loss_present': float(loss_present.item()),
            'loss_total': float(total_loss.item()),
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    # Test-Code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, C, T = 4, 12, 500
    y_hat = torch.randn(B, C, T, device=device)
    y_target = torch.randn(B, C, T, device=device)
    mask = torch.randint(0, 2, (B, C, T), dtype=torch.float32, device=device)
    
    # Test MSE Loss
    mse_loss = MaskedMSELoss()
    loss_mse = mse_loss(y_hat, y_target, mask)
    print(f"Masked MSE Loss: {loss_mse.item():.6f}")
    
    # Test Huber Loss
    huber_loss = MaskedHuberLoss()
    loss_huber = huber_loss(y_hat, y_target, mask)
    print(f"Masked Huber Loss: {loss_huber.item():.6f}")
    
    # Test Combined Loss
    combined_loss = CombinedMaskedLoss(loss_type='mse')
    total_loss, loss_dict = combined_loss(y_hat, y_target, mask)
    print(f"\nCombined Loss:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.6f}")
    
    print("\nTest erfolgreich!")
