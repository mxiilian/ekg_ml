"""
Test-Evaluation für trainiertes EKG-Reparatur-Modell.

Lädt Best-Model und evaluiert auf Test-Set mit umfassenden Metriken.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse
import json

from data import create_dataloader
from model import build_model
from eval import ECGMetrics, evaluate_batch


def load_checkpoint(
    checkpoint_path: str,
    model,
    device: torch.device,
) -> Dict:
    """
    Lade trainiertes Modell aus Checkpoint.
    
    Args:
        checkpoint_path: Pfad zu best_model.pt
        model: Model-Instanz
        device: torch.device
        
    Returns:
        Dict mit Checkpoint-Informationen
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    return checkpoint


@torch.no_grad()
def evaluate_test_set(
    model,
    test_loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluiere auf Test-Set.
    
    Returns:
        Dict mit Test-Metriken
    """
    model.eval()
    
    all_mae = []
    all_rmse = []
    per_lead_mae = {i: [] for i in range(12)}
    per_lead_rmse = {i: [] for i in range(12)}
    
    limb_mae = []
    limb_rmse = []
    chest_mae = []
    chest_rmse = []
    
    print("\nEvaluating on test set...")
    for batch_idx, batch in enumerate(test_loader):
        x_filled = batch['x_filled'].to(device)
        mask = batch['mask'].to(device)
        y_target = batch['y_target'].to(device)
        
        # Forward pass
        y_hat = model.predict(x_filled, mask)
        
        # Numpy conversion
        y_hat_np = y_hat.cpu().numpy()
        y_target_np = y_target.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Batch-Metriken
        for b in range(y_hat_np.shape[0]):
            mae = ECGMetrics.mae_on_missing(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            rmse = ECGMetrics.rmse_on_missing(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            all_mae.append(mae)
            all_rmse.append(rmse)
            
            # Per-Lead
            per_lead_dict = ECGMetrics.per_lead_metrics(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            for lead in range(12):
                per_lead_mae[lead].append(per_lead_dict['mae'][lead])
                per_lead_rmse[lead].append(per_lead_dict['rmse'][lead])
            
            # Grouped
            group_dict = ECGMetrics.group_metrics(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            limb_mae.append(group_dict['limb_mae'])
            limb_rmse.append(group_dict['limb_rmse'])
            chest_mae.append(group_dict['chest_mae'])
            chest_rmse.append(group_dict['chest_rmse'])
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(test_loader)}: "
                  f"MAE={np.mean(all_mae):.6f}, RMSE={np.mean(all_rmse):.6f}")
    
    # Zusammenfassung
    results = {
        'overall_mae': float(np.mean(all_mae)),
        'overall_rmse': float(np.mean(all_rmse)),
        'overall_mae_std': float(np.std(all_mae)),
        'overall_rmse_std': float(np.std(all_rmse)),
        
        'limb_mae': float(np.mean(limb_mae)),
        'limb_rmse': float(np.mean(limb_rmse)),
        'chest_mae': float(np.mean(chest_mae)),
        'chest_rmse': float(np.mean(chest_rmse)),
        
        'per_lead_mae': {
            i: float(np.mean(per_lead_mae[i])) for i in range(12)
        },
        'per_lead_rmse': {
            i: float(np.mean(per_lead_rmse[i])) for i in range(12)
        },
    }
    
    return results


def print_results(results: Dict[str, float]):
    """Hübsche Ausgabe der Ergebnisse."""
    
    print("\n" + "="*70)
    print("TEST EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  MAE:  {results['overall_mae']:.6f} ± {results['overall_mae_std']:.6f}")
    print(f"  RMSE: {results['overall_rmse']:.6f} ± {results['overall_rmse_std']:.6f}")
    
    print(f"\nGrouped Performance:")
    print(f"  Limb Leads (I, II, III, aVR, aVL, aVF):")
    print(f"    MAE:  {results['limb_mae']:.6f}")
    print(f"    RMSE: {results['limb_rmse']:.6f}")
    print(f"  Chest Leads (V1-V6):")
    print(f"    MAE:  {results['chest_mae']:.6f}")
    print(f"    RMSE: {results['chest_rmse']:.6f}")
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    print(f"\nPer-Lead Performance:")
    for i, name in enumerate(lead_names):
        mae = results['per_lead_mae'][i]
        rmse = results['per_lead_rmse'][i]
        print(f"  {name:3s}: MAE={mae:.6f}, RMSE={rmse:.6f}")
    
    print("="*70 + "\n")


def main(args):
    """Hauptevaluations-Script."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test-Daten laden
    print("\n1. Loading test data...")
    test_loader = create_dataloader(
        data_dir=args.test_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        overlap=args.overlap,
        damage_rate=args.damage_rate,  # Auch auf Test anwenden
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Modell bauen
    print("\n2. Building model...")
    model = build_model(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        device=device,
    )
    
    # Checkpoint laden
    print("\n3. Loading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    
    # Evaluieren
    print("\n4. Evaluating...")
    results = evaluate_test_set(model, test_loader, device)
    
    # Ergebnisse anzeigen
    print_results(results)
    
    # Ergebnisse speichern
    results_path = Path(args.output_dir) / 'test_results.json'
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate TCN Repair Net on Test Set"
    )
    
    # Daten
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Directory with test EKG CSV files')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Windowing (sollte gleich wie im Training sein)
    parser.add_argument('--window-size', type=int, default=1000)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--damage-rate', type=float, default=0.1)
    
    # Modell
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory for results')
    
    args = parser.parse_args()
    main(args)
