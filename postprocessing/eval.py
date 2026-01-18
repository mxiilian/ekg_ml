"""
Evaluation-Metriken und Visualisierung für EKG-NaN-Imputation.

Metriken:
- RMSE / MAE nur auf NaN-Stellen
- Optionaler globaler RMSE
- Auswertung getrennt nach Limb Leads und Chest Leads
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from pathlib import Path


class ECGMetrics:
    """
    Metriken für EKG-Reparatur.
    """
    
    # Lead-Gruppen
    LIMB_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
    CHEST_LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    LEAD_NAMES = LIMB_LEADS + CHEST_LEADS
    
    @staticmethod
    def mae_on_missing(
        y_hat: np.ndarray,
        y_target: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        Mean Absolute Error nur auf fehlenden Stellen (mask=0).
        
        Args:
            y_hat: (C, T) — Vorhersage
            y_target: (C, T) — Ground Truth
            mask: (C, T) — 1=vorhanden, 0=fehlt
            
        Returns:
            mae: Scalar
        """
        valid_mask = np.isfinite(y_target) & np.isfinite(y_hat)
        missing_mask = (mask == 0) & valid_mask
        if not missing_mask.any():
            return 0.0
        
        mae = np.mean(np.abs(y_hat[missing_mask] - y_target[missing_mask]))
        return float(mae)
    
    @staticmethod
    def rmse_on_missing(
        y_hat: np.ndarray,
        y_target: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        Root Mean Squared Error nur auf fehlenden Stellen.
        
        Args:
            y_hat: (C, T)
            y_target: (C, T)
            mask: (C, T)
            
        Returns:
            rmse: Scalar
        """
        valid_mask = np.isfinite(y_target) & np.isfinite(y_hat)
        missing_mask = (mask == 0) & valid_mask
        if not missing_mask.any():
            return 0.0
        
        mse = np.mean((y_hat[missing_mask] - y_target[missing_mask]) ** 2)
        rmse = np.sqrt(mse)
        return float(rmse)
    
    @staticmethod
    def rmse_global(
        y_hat: np.ndarray,
        y_target: np.ndarray,
    ) -> float:
        """
        Globaler RMSE auf allen Stellen.
        
        Args:
            y_hat: (C, T)
            y_target: (C, T)
            
        Returns:
            rmse: Scalar
        """
        valid_mask = np.isfinite(y_target) & np.isfinite(y_hat)
        if not valid_mask.any():
            return 0.0
        mse = np.mean((y_hat[valid_mask] - y_target[valid_mask]) ** 2)
        rmse = np.sqrt(mse)
        return float(rmse)
    
    @staticmethod
    def per_lead_metrics(
        y_hat: np.ndarray,
        y_target: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Metriken pro Lead.
        
        Args:
            y_hat: (12, T)
            y_target: (12, T)
            mask: (12, T)
            
        Returns:
            dict: {lead_name: {'mae': ..., 'rmse': ..., 'rmse_global': ...}}
        """
        results = {}
        
        for c, lead_name in enumerate(ECGMetrics.LEAD_NAMES):
            y_hat_lead = y_hat[c]
            y_target_lead = y_target[c]
            mask_lead = mask[c]
            
            mae = ECGMetrics.mae_on_missing(
                y_hat_lead[None], y_target_lead[None], mask_lead[None]
            )
            rmse = ECGMetrics.rmse_on_missing(
                y_hat_lead[None], y_target_lead[None], mask_lead[None]
            )
            rmse_g = ECGMetrics.rmse_global(
                y_hat_lead[None], y_target_lead[None]
            )
            
            results[lead_name] = {
                'mae': mae,
                'rmse': rmse,
                'rmse_global': rmse_g,
            }
        
        return results
    
    @staticmethod
    def group_metrics(
        y_hat: np.ndarray,
        y_target: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Metriken gruppiert nach Limb Leads und Chest Leads.
        
        Args:
            y_hat: (12, T)
            y_target: (12, T)
            mask: (12, T)
            
        Returns:
            dict: {'limb': {...}, 'chest': {...}}
        """
        limb_indices = list(range(6))  # I, II, III, aVR, aVL, aVF
        chest_indices = list(range(6, 12))  # V1-V6
        
        results = {}
        
        for group_name, indices in [('limb', limb_indices), ('chest', chest_indices)]:
            y_hat_group = y_hat[indices]
            y_target_group = y_target[indices]
            mask_group = mask[indices]
            
            mae = ECGMetrics.mae_on_missing(y_hat_group, y_target_group, mask_group)
            rmse = ECGMetrics.rmse_on_missing(y_hat_group, y_target_group, mask_group)
            rmse_g = ECGMetrics.rmse_global(y_hat_group, y_target_group)
            
            results[group_name] = {
                'mae': mae,
                'rmse': rmse,
                'rmse_global': rmse_g,
            }
        
        return results


def plot_overlay(
    x_filled: np.ndarray,
    y_hat: np.ndarray,
    y_target: np.ndarray,
    lead_names: list = None,
    figsize: Tuple = (15, 10),
    save_path: Optional[str] = None,
):
    """
    Overlays visualisieren: x_filled vs y_hat vs y_target.
    
    Args:
        x_filled: (12, T) — Input mit 0en gefüllt
        y_hat: (12, T) — Modell-Vorhersage
        y_target: (12, T) — Ground Truth
        lead_names: Liste von Lead-Namen
        figsize: Figure-Größe
        save_path: Optional: Speicherpfad
    """
    if lead_names is None:
        lead_names = ECGMetrics.LEAD_NAMES
    
    n_leads = x_filled.shape[0]
    n_cols = 2
    n_rows = (n_leads + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    T = x_filled.shape[1]
    t_axis = np.arange(T)
    
    for c in range(n_leads):
        ax = axes[c]
        
        ax.plot(t_axis, x_filled[c], 'o-', label='x_filled (input)', alpha=0.6, markersize=2)
        ax.plot(t_axis, y_hat[c], 's-', label='y_hat (prediction)', alpha=0.7, markersize=2)
        ax.plot(t_axis, y_target[c], 'x-', label='y_target (ground truth)', alpha=0.8, markersize=2)
        
        ax.set_title(f"{lead_names[c]}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Letzten leeren Plot entfernen
    if n_leads < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot gespeichert: {save_path}")
    
    return fig


def evaluate_batch(
    model,
    dataloader,
    device: torch.device,
    num_batches: Optional[int] = None,
) -> Dict:
    """
    Evaluate Modell auf Batch.
    
    Args:
        model: TCN-Modell
        dataloader: Data Loader
        device: torch device
        num_batches: Optional max Anzahl Batches
        
    Returns:
        Dict mit Metriken
    """
    model.eval()
    
    all_mae = []
    all_rmse = []
    all_rmse_global = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break
            
            x_filled = batch['x_filled'].to(device)
            mask = batch['mask'].to(device)
            y_target = batch['y_target'].to(device)
            
            # Vorhersage
            y_hat = model.predict(x_filled, mask)
            
            # CPU für Metriken
            y_hat_np = y_hat.cpu().numpy()
            y_target_np = y_target.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # Pro Batch
            for b in range(y_hat_np.shape[0]):
                mae = ECGMetrics.mae_on_missing(
                    y_hat_np[b], y_target_np[b], mask_np[b]
                )
                rmse = ECGMetrics.rmse_on_missing(
                    y_hat_np[b], y_target_np[b], mask_np[b]
                )
                rmse_g = ECGMetrics.rmse_global(y_hat_np[b], y_target_np[b])
                
                all_mae.append(mae)
                all_rmse.append(rmse)
                all_rmse_global.append(rmse_g)
    
    return {
        'mae_mean': np.mean(all_mae),
        'mae_std': np.std(all_mae),
        'rmse_mean': np.mean(all_rmse),
        'rmse_std': np.std(all_rmse),
        'rmse_global_mean': np.mean(all_rmse_global),
        'rmse_global_std': np.std(all_rmse_global),
        'num_samples': len(all_mae),
    }


if __name__ == '__main__':
    # Test-Code
    y_hat = np.random.randn(12, 500)
    y_target = np.random.randn(12, 500)
    mask = np.random.randint(0, 2, (12, 500))
    
    mae = ECGMetrics.mae_on_missing(y_hat, y_target, mask)
    rmse = ECGMetrics.rmse_on_missing(y_hat, y_target, mask)
    rmse_g = ECGMetrics.rmse_global(y_hat, y_target)
    
    print(f"MAE (missing): {mae:.6f}")
    print(f"RMSE (missing): {rmse:.6f}")
    print(f"RMSE (global): {rmse_g:.6f}")
    
    per_lead = ECGMetrics.per_lead_metrics(y_hat, y_target, mask)
    print("\nPer-Lead Metrics (first 3):")
    for i, (lead, metrics) in enumerate(list(per_lead.items())[:3]):
        print(f"  {lead}: {metrics}")
    
    group = ECGMetrics.group_metrics(y_hat, y_target, mask)
    print("\nGroup Metrics:")
    for group_name, metrics in group.items():
        print(f"  {group_name}: {metrics}")
    
    print("\nTest erfolgreich!")
