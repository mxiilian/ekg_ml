"""
Debugging Script für Konvergenzprobleme.

Testet:
1. Daten-Qualität (NaN-Verteilung, Signal-Stärke)
2. Modell-Ausgaben (Gradient-Flow, Aktivierungen)
3. Loss-Stabilität
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import sys

from data import create_dataloader
from model import build_model
from loss import CombinedMaskedLoss


def check_data_quality(train_loader, num_batches=5):
    """Überprüfe Daten-Qualität."""
    print("\n" + "="*70)
    print("CHECK 1: Data Quality")
    print("="*70)
    
    all_mask_ratios = []
    all_signal_means = []
    all_signal_stds = []
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        x_filled = batch['x_filled'].numpy()
        mask = batch['mask'].numpy()
        y_target = batch['y_target'].numpy()
        
        # Missing ratio
        missing_mask = (mask == 0)
        missing_ratio = missing_mask.mean()
        all_mask_ratios.append(missing_ratio)
        
        # Signal statistics
        valid_x = x_filled[np.isfinite(x_filled)]
        valid_y = y_target[np.isfinite(y_target)]
        
        if valid_x.size > 0:
            all_signal_means.append(np.abs(valid_x).mean())
            all_signal_stds.append(np.abs(valid_x).std())
        
        print(f"Batch {batch_idx}:")
        print(f"  - Shape: x_filled={x_filled.shape}, y_target={y_target.shape}")
        print(f"  - Missing ratio: {missing_ratio:.4f}")
        print(f"  - x_filled valid: {np.isfinite(x_filled).mean():.4f}")
        print(f"  - y_target valid: {np.isfinite(y_target).mean():.4f}")
        if valid_x.size > 0:
            print(f"  - |x_filled| mean: {np.abs(valid_x).mean():.6f}")
            print(f"  - |x_filled| std: {np.abs(valid_x).std():.6f}")
            print(f"  - |y_target| mean: {np.abs(valid_y).mean():.6f}")
            print(f"  - |y_target| std: {np.abs(valid_y).std():.6f}")
    
    print(f"\nSummary:")
    print(f"  - Avg missing ratio: {np.mean(all_mask_ratios):.4f}")
    print(f"  - Avg |signal| mean: {np.mean(all_signal_means):.6f}")
    print(f"  - Avg |signal| std: {np.mean(all_signal_stds):.6f}")
    
    # ⚠️ Warnungen
    if np.mean(all_mask_ratios) < 0.01:
        print("  ⚠️ WARNING: Missing ratio sehr niedrig → Modell hat wenig zu tun!")
    if np.mean(all_signal_means) < 1e-4:
        print("  ⚠️ WARNING: Signal-Amplituden sehr klein → Trainieren könnte schwierig sein!")


def check_gradient_flow(model, train_loader, loss_fn, device, num_batches=3):
    """Überprüfe Gradient-Flow."""
    print("\n" + "="*70)
    print("CHECK 2: Gradient Flow")
    print("="*70)
    
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        x_filled = batch['x_filled'].to(device)
        mask = batch['mask'].to(device)
        y_target = batch['y_target'].to(device)
        
        x_filled = torch.nan_to_num(x_filled, nan=0.0, posinf=0.0, neginf=0.0)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Forward pass
        y_hat = model.predict(x_filled, mask)
        y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Loss
        loss, loss_dict = loss_fn(y_hat, y_target, mask)
        
        print(f"Batch {batch_idx}:")
        print(f"  - Loss: {loss.item():.6e}")
        print(f"  - Loss is finite: {torch.isfinite(loss).item()}")
        print(f"  - y_hat is finite: {torch.isfinite(y_hat).all().item()}")
        
        # Backward
        loss.backward()
        
        # Überprüfe Gradienten
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                grad_norms.append((name, grad_norm))
        
        grad_norms.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  - Top 5 gradient norms:")
        for name, grad_norm in grad_norms[:5]:
            print(f"    {name}: {grad_norm:.6e}")
        
        # Überprüfe auf NaN Gradienten
        nan_grads = sum(1 for name, gn in grad_norms if not np.isfinite(gn))
        if nan_grads > 0:
            print(f"  ⚠️ WARNING: {nan_grads} NaN/Inf gradients detected!")
        
        model.zero_grad()


def check_model_output_scale(model, train_loader, device, num_batches=5):
    """Überprüfe Output-Skala des Modells."""
    print("\n" + "="*70)
    print("CHECK 3: Model Output Scale")
    print("="*70)
    
    model.eval()
    
    delta_means = []
    delta_stds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            x_filled = batch['x_filled'].to(device)
            mask = batch['mask'].to(device)
            
            x_filled = torch.nan_to_num(x_filled, nan=0.0, posinf=0.0, neginf=0.0)
            mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Forward
            delta = model(x_filled, mask)
            y_hat = model.predict(x_filled, mask)
            
            delta_np = delta.detach().cpu().numpy()
            y_hat_np = y_hat.detach().cpu().numpy()
            x_filled_np = x_filled.detach().cpu().numpy()
            
            # Stats
            valid_delta = delta_np[np.isfinite(delta_np)]
            valid_y_hat = y_hat_np[np.isfinite(y_hat_np)]
            valid_x = x_filled_np[np.isfinite(x_filled_np)]
            
            if valid_delta.size > 0:
                delta_mean = np.abs(valid_delta).mean()
                delta_std = np.abs(valid_delta).std()
                delta_means.append(delta_mean)
                delta_stds.append(delta_std)
                
                print(f"Batch {batch_idx}:")
                print(f"  - |Δ| mean: {delta_mean:.6e}")
                print(f"  - |Δ| std: {delta_std:.6e}")
                print(f"  - |Δ| max: {np.abs(valid_delta).max():.6e}")
                print(f"  - |x_filled| mean: {np.abs(valid_x).mean():.6e}")
                print(f"  - |y_hat| mean: {np.abs(valid_y_hat).mean():.6e}")
    
    if delta_means:
        print(f"\nSummary:")
        print(f"  - Avg |Δ| mean: {np.mean(delta_means):.6e}")
        print(f"  - Avg |Δ| std: {np.mean(delta_stds):.6e}")
        
        # Warnungen
        if np.mean(delta_means) < 1e-6:
            print("  ⚠️ WARNING: Δ outputs zu klein! Modell könnte nicht trainiert sein.")
        if np.mean(delta_stds) < 1e-8:
            print("  ⚠️ WARNING: Δ outputs sehr konsistent (constant)! Modell nicht lernend?")


def main():
    print("\n" + "="*70)
    print("CONVERGENCE DEBUGGING FOR EKG IMPUTATION MODEL")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Lade Trainingsdaten
    print("\n1. Loading training data...")
    train_loader = create_dataloader(
        data_dir='./data_split/train',
        batch_size=8,
        window_size=1000,
        overlap=0.5,
        damage_rate=0.1,
        min_damage_abs=1e-6,
        valid_abs_eps=1e-6,
        min_valid_ratio=0.05,
        damage_blocks_min=1,
        damage_blocks_max=3,
        damage_block_min=100,
        damage_block_max=500,
        shuffle=True,
        num_workers=0,
    )
    
    print(f"Datenlader erstellt. Erste Batch-Größe prüfen...")
    
    # Check 1: Data Quality
    check_data_quality(train_loader)
    
    # Build Model
    print("\n2. Building model...")
    model = build_model(device=device)
    print(f"✓ Modell gebaut mit {sum(p.numel() for p in model.parameters()):,} Parametern")
    
    # Check 2: Gradient Flow
    loss_fn = CombinedMaskedLoss(loss_type='mse', w_missing=1.0, w_present=0.001)
    check_gradient_flow(model, train_loader, loss_fn, device)
    
    # Check 3: Model Output Scale
    check_model_output_scale(model, train_loader, device)
    
    print("\n" + "="*70)
    print("DEBUGGING COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
