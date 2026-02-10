"""
Test-Evaluation für trainiertes EKG-Reparatur-Modell.

Lädt Best-Model und evaluiert auf Test-Set mit umfassenden Metriken.
"""

import torch
import numpy as np
from torch.serialization import safe_globals
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
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        # PyTorch 2.6+: allowlist numpy types for weights_only loading
        if "Weights only load failed" in str(e) or "WeightsUnpickler" in str(e):
            allowlist = [np.core.multiarray.scalar, np.dtype]
            # Newer numpy uses dtype classes like numpy.dtypes.Float64DType
            float64_dtype = getattr(getattr(np, "dtypes", None), "Float64DType", None)
            if float64_dtype is not None:
                allowlist.append(float64_dtype)
            with safe_globals(allowlist):
                checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            raise
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
    all_snr = []
    missing_snr_digitizer = []
    missing_snr_recon = []
    snr_present_digitizer = []
    snr_present_recon = []
    global_snr_digitizer = []
    global_snr_recon = []
    per_lead_mae = {i: [] for i in range(12)}
    per_lead_rmse = {i: [] for i in range(12)}
    per_lead_snr = {i: [] for i in range(12)}
    
    limb_mae = []
    limb_rmse = []
    chest_mae = []
    chest_rmse = []
    
    print("\nEvaluating on test set...")
    for batch_idx, batch in enumerate(test_loader):
        x_filled = batch['x_filled'].to(device)
        mask = batch['mask'].to(device)
        y_target = batch['y_target'].to(device)
        lead_mean = batch['lead_mean'].to(device)
        lead_std = batch['lead_std'].to(device)
        
        # Forward pass
        y_hat = model.predict(x_filled, mask)
        if not torch.isfinite(y_hat).all():
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        y_hat_denorm = y_hat * lead_std + lead_mean
        y_target_denorm = y_target * lead_std + lead_mean
        x_denorm = x_filled * lead_std + lead_mean
        # Option A: keep digitizer values on present points, only fill missing
        if args.use_digitizer:
            y_recon = torch.where(mask == 1, x_denorm, y_hat_denorm)
        else:
            y_recon = y_hat_denorm
        
        # Numpy conversion
        y_hat_np = y_recon.cpu().numpy()
        y_target_np = y_target_denorm.cpu().numpy()
        mask_np = mask.cpu().numpy()
        x_denorm_np = x_denorm.cpu().numpy()
        
        # Batch-Metriken
        for b in range(y_hat_np.shape[0]):
            mae = ECGMetrics.mae_on_missing(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            rmse = ECGMetrics.rmse_on_missing(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            snr = ECGMetrics.snr_on_missing(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            all_mae.append(mae)
            all_rmse.append(rmse)
            if np.isfinite(snr):
                all_snr.append(snr)

            # Missing SNR (digitizer baseline vs recon)
            if args.use_digitizer:
                snr_md = ECGMetrics.snr_on_missing(
                    x_denorm_np[b], y_target_np[b], mask_np[b]
                )
                snr_mr = ECGMetrics.snr_on_missing(
                    y_hat_np[b], y_target_np[b], mask_np[b]
                )
                if np.isfinite(snr_md):
                    missing_snr_digitizer.append(snr_md)
                if np.isfinite(snr_mr):
                    missing_snr_recon.append(snr_mr)

                snr_gd = ECGMetrics.snr_global(
                    x_denorm_np[b], y_target_np[b]
                )
                snr_gr = ECGMetrics.snr_global(
                    y_hat_np[b], y_target_np[b]
                )
                if np.isfinite(snr_gd):
                    global_snr_digitizer.append(snr_gd)
                if np.isfinite(snr_gr):
                    global_snr_recon.append(snr_gr)

            # Present SNR (digitizer vs GT) and (recon vs GT)
            snr_d = ECGMetrics.snr_on_present(
                x_denorm_np[b], y_target_np[b], mask_np[b]
            )
            snr_r = ECGMetrics.snr_on_present(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            if np.isfinite(snr_d):
                snr_present_digitizer.append(snr_d)
            if np.isfinite(snr_r):
                snr_present_recon.append(snr_r)
            
            # Per-Lead
            per_lead_dict = ECGMetrics.per_lead_metrics(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            for lead_idx, lead_name in enumerate(ECGMetrics.LEAD_NAMES):
                per_lead_mae[lead_idx].append(per_lead_dict[lead_name]['mae'])
                per_lead_rmse[lead_idx].append(per_lead_dict[lead_name]['rmse'])
                snr_lead = ECGMetrics.snr_on_missing(
                    y_hat_np[b][lead_idx:lead_idx+1],
                    y_target_np[b][lead_idx:lead_idx+1],
                    mask_np[b][lead_idx:lead_idx+1],
                )
                if np.isfinite(snr_lead):
                    per_lead_snr[lead_idx].append(snr_lead)
            
            # Grouped
            group_dict = ECGMetrics.group_metrics(
                y_hat_np[b], y_target_np[b], mask_np[b]
            )
            limb_mae.append(group_dict['limb']['mae'])
            limb_rmse.append(group_dict['limb']['rmse'])
            chest_mae.append(group_dict['chest']['mae'])
            chest_rmse.append(group_dict['chest']['rmse'])
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(test_loader)}: "
                  f"MAE={np.mean(all_mae):.6f}, RMSE={np.mean(all_rmse):.6f}")
    
    # Zusammenfassung
    results = {
        'overall_mae': float(np.mean(all_mae)),
        'overall_rmse': float(np.mean(all_rmse)),
        'overall_mae_std': float(np.std(all_mae)),
        'overall_rmse_std': float(np.std(all_rmse)),
        'overall_snr_db': float(np.mean(all_snr)) if all_snr else float('nan'),
        'overall_snr_db_std': float(np.std(all_snr)) if all_snr else float('nan'),
        'missing_snr_db_digitizer': float(np.mean(missing_snr_digitizer)) if missing_snr_digitizer else float('nan'),
        'missing_snr_db_recon': float(np.mean(missing_snr_recon)) if missing_snr_recon else float('nan'),
        'missing_snr_db_improvement': (
            float(np.mean(missing_snr_recon)) - float(np.mean(missing_snr_digitizer))
            if missing_snr_recon and missing_snr_digitizer else float('nan')
        ),
        'present_snr_db_digitizer': float(np.mean(snr_present_digitizer)) if snr_present_digitizer else float('nan'),
        'present_snr_db_recon': float(np.mean(snr_present_recon)) if snr_present_recon else float('nan'),
        'present_snr_db_improvement': (
            float(np.mean(snr_present_recon)) - float(np.mean(snr_present_digitizer))
            if snr_present_recon and snr_present_digitizer else float('nan')
        ),
        'global_snr_db_digitizer': float(np.mean(global_snr_digitizer)) if global_snr_digitizer else float('nan'),
        'global_snr_db_recon': float(np.mean(global_snr_recon)) if global_snr_recon else float('nan'),
        'global_snr_db_improvement': (
            float(np.mean(global_snr_recon)) - float(np.mean(global_snr_digitizer))
            if global_snr_recon and global_snr_digitizer else float('nan')
        ),
        
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
        'per_lead_snr_db': {
            i: float(np.mean(per_lead_snr[i])) if per_lead_snr[i] else float('nan')
            for i in range(12)
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
    if 'overall_snr_db' in results:
        print(f"  SNR (missing):  {results['overall_snr_db']:.2f} dB ± {results['overall_snr_db_std']:.2f}")
    if 'missing_snr_db_digitizer' in results and np.isfinite(results['missing_snr_db_digitizer']):
        print(f"  SNR (missing, digitizer): {results['missing_snr_db_digitizer']:.2f} dB")
        print(f"  SNR (missing, recon):     {results['missing_snr_db_recon']:.2f} dB")
        print(f"  SNR Δ (missing, recon - digitizer): {results['missing_snr_db_improvement']:.2f} dB")
    if 'global_snr_db_digitizer' in results and np.isfinite(results['global_snr_db_digitizer']):
        print(f"  SNR (global, digitizer): {results['global_snr_db_digitizer']:.2f} dB")
        print(f"  SNR (global, recon):     {results['global_snr_db_recon']:.2f} dB")
        print(f"  SNR Δ (global, recon - digitizer): {results['global_snr_db_improvement']:.2f} dB")
    if 'present_snr_db_digitizer' in results:
        print(f"  SNR (present, digitizer): {results['present_snr_db_digitizer']:.2f} dB")
        print(f"  SNR (present, recon):     {results['present_snr_db_recon']:.2f} dB")
        print(f"  SNR Δ (recon - digitizer): {results['present_snr_db_improvement']:.2f} dB")
    
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
        snr = results.get('per_lead_snr_db', {}).get(i, float('nan'))
        if np.isfinite(snr):
            print(f"  {name:3s}: MAE={mae:.6f}, RMSE={rmse:.6f}, SNR={snr:.2f} dB")
        else:
            print(f"  {name:3s}: MAE={mae:.6f}, RMSE={rmse:.6f}")
    
    print("="*70 + "\n")


def _load_history_config(path: Path) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('config', {})


def _parse_variant(record_id: str) -> Optional[str]:
    """
    Extract digitizer variant from record_id when using --digitizer-variant all.
    Expected format: "<id>:<stem>" where stem includes "<id>-0001_..."
    """
    if record_id is None:
        return None
    if ":" not in record_id:
        return None
    base, stem = record_id.split(":", 1)
    token = f"{base}-"
    if token in stem:
        rest = stem.split(token, 1)[1]
        return rest.split("_", 1)[0]
    return None


def _resolve_config(args) -> Dict:
    """
    Merge CLI args with optional training_history.json config.
    CLI args take precedence.
    """
    config = {}
    history_config = None
    if args.run_dir:
        history_path = Path(args.run_dir) / 'training_history.json'
        history_config = _load_history_config(history_path)
    elif args.history:
        history_config = _load_history_config(Path(args.history))

    if history_config:
        config.update(history_config)

    # CLI args override history config when provided
    for key in [
        'batch_size',
        'num_workers',
        'window_size',
        'overlap',
        'damage_rate',
        'min_damage_abs',
        'valid_abs_eps',
        'min_valid_ratio',
        'damage_blocks_min',
        'damage_blocks_max',
        'damage_block_min',
        'damage_block_max',
        'max_missing_leads_per_timestep',
        'protect_leads',
        'hidden_channels',
        'num_blocks',
        'dropout',
    ]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    return config


def main(args):
    """Hauptevaluations-Script."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Checkpoint resolve (optional via run dir)
    if args.checkpoint is None:
        if args.run_dir:
            args.checkpoint = str(Path(args.run_dir) / 'best_model.pt')
        else:
            raise ValueError(
                "Missing --checkpoint. Provide --checkpoint or use --run-dir "
                "with a best_model.pt inside."
            )

    config = _resolve_config(args)
    model_id = None
    if args.run_dir:
        model_id = Path(args.run_dir).name
    elif args.checkpoint:
        model_id = Path(args.checkpoint).stem
    else:
        model_id = "unknown"

    # Test-Daten laden
    print("\n1. Loading test data...")
    test_loader = create_dataloader(
        data_dir=args.test_dir,
        batch_size=config.get('batch_size', 32),
        window_size=config.get('window_size', 1000),
        overlap=config.get('overlap', 0.5),
        damage_rate=config.get('damage_rate', 0.1),  # Auch auf Test anwenden
        min_damage_abs=config.get('min_damage_abs', 1e-6),
        valid_abs_eps=config.get('valid_abs_eps', 1e-6),
        min_valid_ratio=config.get('min_valid_ratio', 0.05),
        damage_blocks_min=config.get('damage_blocks_min', 1),
        damage_blocks_max=config.get('damage_blocks_max', 3),
        damage_block_min=config.get('damage_block_min', 100),
        damage_block_max=config.get('damage_block_max', 500),
        max_missing_leads_per_timestep=config.get('max_missing_leads_per_timestep', None),
        protect_leads=config.get('protect_leads', None),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        x_dir=args.x_dir,
        use_digitizer=args.use_digitizer,
        digitizer_variant=args.digitizer_variant,
    )
    
    # Modell bauen
    print("\n2. Building model...")
    model = build_model(
        hidden_channels=config.get('hidden_channels', 64),
        num_blocks=config.get('num_blocks', 4),
        dropout=config.get('dropout', 0.2),
        device=device,
    )
    
    # Checkpoint laden
    print("\n3. Loading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    
    # Evaluieren
    print("\n4. Evaluating...")
    results = evaluate_test_set(model, test_loader, device)
    # Attach metadata to results for traceability
    results['meta'] = {
        'model_id': model_id,
        'checkpoint': args.checkpoint,
        'run_dir': args.run_dir,
        'test_dir': args.test_dir,
        'x_dir': args.x_dir,
        'use_digitizer': bool(args.use_digitizer),
        'digitizer_variant': args.digitizer_variant,
    }
    
    # Ergebnisse anzeigen
    print_results(results)
    
    # Ergebnisse speichern
    suffix = "_digitizer" if args.use_digitizer else "_synthetic"
    variant_tag = f"_{args.digitizer_variant}" if args.digitizer_variant else ""
    results_filename = f"test_results_{model_id}{suffix}{variant_tag}.json"
    results_path = Path(args.output_dir) / results_filename
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Optional: detailed aggregates for analysis
    if args.detailed_output:
        details = {
            "per_record": {},
            "per_variant": {},
            "per_variant_lead": {},
        }

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x_filled = batch['x_filled'].to(device)
                mask = batch['mask'].to(device)
                y_target = batch['y_target'].to(device)
                lead_mean = batch['lead_mean'].to(device)
                lead_std = batch['lead_std'].to(device)
                record_ids = batch['record_id']

                y_hat = model.predict(x_filled, mask)
                if not torch.isfinite(y_hat).all():
                    y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
                y_hat_denorm = y_hat * lead_std + lead_mean
                y_target_denorm = y_target * lead_std + lead_mean
                x_denorm = x_filled * lead_std + lead_mean
                if args.use_digitizer:
                    y_recon = torch.where(mask == 1, x_denorm, y_hat_denorm)
                else:
                    y_recon = y_hat_denorm

                y_hat_np = y_recon.cpu().numpy()
                y_target_np = y_target_denorm.cpu().numpy()
                mask_np = mask.cpu().numpy()
                x_denorm_np = x_denorm.cpu().numpy()

                for b in range(y_hat_np.shape[0]):
                    rid = str(record_ids[b])
                    variant = _parse_variant(rid)
                    mae = ECGMetrics.mae_on_missing(y_hat_np[b], y_target_np[b], mask_np[b])
                    rmse = ECGMetrics.rmse_on_missing(y_hat_np[b], y_target_np[b], mask_np[b])
                    snr_m = ECGMetrics.snr_on_missing(y_hat_np[b], y_target_np[b], mask_np[b])
                    snr_g = ECGMetrics.snr_global(y_hat_np[b], y_target_np[b])
                    snr_gd = ECGMetrics.snr_global(x_denorm_np[b], y_target_np[b]) if args.use_digitizer else float('nan')

                    rec = details["per_record"].setdefault(rid, {"mae": [], "rmse": [], "snr_missing": [], "snr_global": []})
                    rec["mae"].append(mae)
                    rec["rmse"].append(rmse)
                    if np.isfinite(snr_m):
                        rec["snr_missing"].append(snr_m)
                    if np.isfinite(snr_g):
                        rec["snr_global"].append(snr_g)

                    if variant:
                        var = details["per_variant"].setdefault(variant, {"mae": [], "rmse": [], "snr_missing": [], "snr_global": [], "snr_global_digitizer": []})
                        var["mae"].append(mae)
                        var["rmse"].append(rmse)
                        if np.isfinite(snr_m):
                            var["snr_missing"].append(snr_m)
                        if np.isfinite(snr_g):
                            var["snr_global"].append(snr_g)
                        if np.isfinite(snr_gd):
                            var["snr_global_digitizer"].append(snr_gd)

                        per_lead_dict = ECGMetrics.per_lead_metrics(y_hat_np[b], y_target_np[b], mask_np[b])
                        for i, name in enumerate(ECGMetrics.LEAD_NAMES):
                            key = f"{variant}:{name}"
                            pl = details["per_variant_lead"].setdefault(key, {"mae": [], "rmse": []})
                            pl["mae"].append(per_lead_dict[name]["mae"])
                            pl["rmse"].append(per_lead_dict[name]["rmse"])

        def _reduce(d):
            out = {}
            for k, v in d.items():
                out[k] = {kk: (float(np.mean(vv)) if vv else float('nan')) for kk, vv in v.items()}
            return out

        details["per_record"] = _reduce(details["per_record"])
        details["per_variant"] = _reduce(details["per_variant"])
        details["per_variant_lead"] = _reduce(details["per_variant_lead"])

        detailed_path = Path(args.detailed_output)
        detailed_path.parent.mkdir(exist_ok=True)
        with open(detailed_path, "w") as f:
            json.dump(details, f, indent=2)
        print(f"Detailed results saved to: {detailed_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate TCN Repair Net on Test Set"
    )
    
    # Run config (optional)
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Run directory containing training_history.json and best_model.pt')
    parser.add_argument('--history', type=str, default=None,
                        help='Path to training_history.json (optional)')
    
    # Daten
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Directory with test EKG CSV files')
    parser.add_argument('--x-dir', type=str, default=None,
                        help='Optional directory with digitizer outputs (real missing data)')
    parser.add_argument('--use-digitizer', action='store_true',
                        help='Use digitizer outputs from --x-dir instead of synthetic masking')
    parser.add_argument('--digitizer-variant', type=str, default=None,
                        help='Prefer digitizer variant like 0001, 0002, ... when available; use \"all\" to evaluate all variants')
    parser.add_argument('--detailed-output', type=str, default=None,
                        help='Write detailed aggregates (per record/variant/lead) to JSON')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    
    # Windowing (sollte gleich wie im Training sein)
    parser.add_argument('--window-size', type=int, default=None)
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--damage-rate', type=float, default=None)
    parser.add_argument('--min-damage-abs', type=float, default=None)
    parser.add_argument('--valid-abs-eps', type=float, default=None)
    parser.add_argument('--min-valid-ratio', type=float, default=None)
    parser.add_argument('--damage-blocks-min', type=int, default=None)
    parser.add_argument('--damage-blocks-max', type=int, default=None)
    parser.add_argument('--damage-block-min', type=int, default=None)
    parser.add_argument('--damage-block-max', type=int, default=None)
    parser.add_argument('--max-missing-leads-per-timestep', type=int, default=None)
    parser.add_argument('--protect-leads', type=str, default=None,
                        help='Comma-separated lead names to protect, e.g. "I,II,V1"')
    
    # Modell
    parser.add_argument('--hidden-channels', type=int, default=None)
    parser.add_argument('--num-blocks', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to best_model.pt')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory for results')
    
    args = parser.parse_args()
    if args.protect_leads is not None:
        args.protect_leads = [s.strip() for s in args.protect_leads.split(',') if s.strip()]
    main(args)
