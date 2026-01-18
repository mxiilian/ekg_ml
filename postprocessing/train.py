"""
Trainingsloop für EKG-NaN-Imputation TCN.

Workflow:
1. Daten laden (ground truth + synthetische Beschädigung)
2. Modell initialisieren
3. Training mit Masked Loss
4. Validierung
5. Checkpoints speichern
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import Optional, Dict, Tuple
import argparse

from data import create_dataloader
from model import build_model
from loss import CombinedMaskedLoss
from eval import evaluate_batch, ECGMetrics


class Trainer:
    """
    Trainer für EKG-Reparatur-Modell.
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: torch.device,
        checkpoint_dir: Path = None,
        grad_clip_norm: float = 5.0,
        run_id: str = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.checkpoint_dir = checkpoint_dir or Path('./checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.checkpoint_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'learning_rate': [],
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self._warned_nonfinite = False
        self.grad_clip_norm = grad_clip_norm
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Eine Trainings-Epoch.
        
        Returns:
            Dict mit durchschnittlichen Loss-Metriken
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_loss_missing = 0.0
        epoch_loss_present = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            x_filled = batch['x_filled'].to(self.device)
            mask = batch['mask'].to(self.device)
            y_target = batch['y_target'].to(self.device)
            lead_mean = batch['lead_mean'].to(self.device)
            lead_std = batch['lead_std'].to(self.device)
            
            # Safety: Inputs muessen endlich sein
            x_filled = torch.nan_to_num(x_filled, nan=0.0, posinf=0.0, neginf=0.0)
            mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Debug: Anteil gueltiger Missing-Stellen im Batch
            valid_mask = torch.isfinite(y_target) & torch.isfinite(x_filled)
            missing_mask = (mask == 0) & valid_mask
            if (batch_idx + 1) % 20 == 0:
                missing_ratio = missing_mask.float().mean().item()
                if missing_mask.any():
                    tgt_abs = y_target[missing_mask].abs().mean().item()
                    x_abs = x_filled[missing_mask].abs().mean().item()
                    print(
                        "    Debug: missing_ratio="
                        f"{missing_ratio:.6f} "
                        f"tgt_abs={tgt_abs:.6e} "
                        f"x_abs={x_abs:.6e}"
                    )
                else:
                    print(
                        f"    Debug: missing_ratio={missing_ratio:.6f} "
                        "keine gueltigen Missing-Stellen"
                    )

            # Forward pass
            y_hat = self.model.predict(x_filled, mask)
            if not torch.isfinite(y_hat).all():
                if not self._warned_nonfinite:
                    print("    Warning: non-finite y_hat; replacing NaN/Inf with 0")
                    self._warned_nonfinite = True
                y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
            y_hat_denorm = y_hat * lead_std + lead_mean
            y_target_denorm = y_target * lead_std + lead_mean
            
            # Debug: Vorhersage-NaNs und Loss auf Missing-Stellen
            if (batch_idx + 1) % 20 == 0:
                pred_finite = torch.isfinite(y_hat)
                pred_nan_ratio = 1.0 - pred_finite.float().mean().item()
                debug_valid = torch.isfinite(y_target_denorm) & pred_finite
                debug_missing = (mask == 0) & debug_valid
                if debug_missing.any():
                    debug_mse = ((y_hat_denorm[debug_missing] - y_target_denorm[debug_missing]) ** 2).mean().item()
                    print(
                        "    Debug: pred_nan_ratio="
                        f"{pred_nan_ratio:.6f} "
                        f"missing_mse={debug_mse:.6e}"
                    )
                else:
                    print(
                        "    Debug: pred_nan_ratio="
                        f"{pred_nan_ratio:.6f} missing_mse=n/a"
                    )
            
            # Loss berechnen
            loss, loss_dict = self.loss_fn(y_hat_denorm, y_target_denorm, mask)
            if not torch.isfinite(loss):
                if not self._warned_nonfinite:
                    print("    Warning: non-finite loss; skipping batch")
                    self._warned_nonfinite = True
                self.optimizer.zero_grad(set_to_none=True)
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm,
            )
            self.optimizer.step()
            
            epoch_loss += loss_dict['loss_total']
            epoch_loss_missing += loss_dict['loss_missing']
            epoch_loss_present += loss_dict['loss_present']
            num_batches += 1
            
            if (batch_idx + 1) % 20 == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"loss={loss_dict['loss_total']:.6e} "
                    f"missing={loss_dict['loss_missing']:.6e} "
                    f"present={loss_dict['loss_present']:.6e}"
                )
        
        if num_batches == 0:
            print("    Warning: no valid batches; returning zero losses")
            return {
                'train_loss': 0.0,
                'train_loss_missing': 0.0,
                'train_loss_present': 0.0,
            }
        return {
            'train_loss': epoch_loss / num_batches,
            'train_loss_missing': epoch_loss_missing / num_batches,
            'train_loss_present': epoch_loss_present / num_batches,
        }
    
    @torch.no_grad()
    def validate(self, val_loader, num_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Validierungsloop.
        
        Returns:
            Dict mit Metriken
        """
        self.model.eval()
        
        val_loss_total = 0.0
        num_batches_processed = 0
        
        all_mae = []
        all_rmse = []
        
        for batch_idx, batch in enumerate(val_loader):
            if num_batches and batch_idx >= num_batches:
                break
            
            x_filled = batch['x_filled'].to(self.device)
            mask = batch['mask'].to(self.device)
            y_target = batch['y_target'].to(self.device)
            lead_mean = batch['lead_mean'].to(self.device)
            lead_std = batch['lead_std'].to(self.device)
            
            x_filled = torch.nan_to_num(x_filled, nan=0.0, posinf=0.0, neginf=0.0)
            mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Forward pass
            y_hat = self.model.predict(x_filled, mask)
            if not torch.isfinite(y_hat).all():
                y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
            y_hat_denorm = y_hat * lead_std + lead_mean
            y_target_denorm = y_target * lead_std + lead_mean
            
            # Loss
            loss, loss_dict = self.loss_fn(y_hat_denorm, y_target_denorm, mask)
            val_loss_total += loss_dict['loss_total']
            
            if batch_idx == 0:
                valid_mask = torch.isfinite(y_target_denorm) & torch.isfinite(y_hat_denorm)
                missing_mask = (mask == 0) & valid_mask
                missing_ratio = missing_mask.float().mean().item()
                print(f"    Val debug: missing_ratio={missing_ratio:.6f}")
            
            # Metriken (CPU)
            y_hat_np = y_hat_denorm.cpu().numpy()
            y_target_np = y_target_denorm.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for b in range(y_hat_np.shape[0]):
                mae = ECGMetrics.mae_on_missing(
                    y_hat_np[b], y_target_np[b], mask_np[b]
                )
                rmse = ECGMetrics.rmse_on_missing(
                    y_hat_np[b], y_target_np[b], mask_np[b]
                )
                all_mae.append(mae)
                all_rmse.append(rmse)
            
            num_batches_processed += 1
        
        return {
            'val_loss': val_loss_total / num_batches_processed,
            'val_mae': np.mean(all_mae),
            'val_rmse': np.mean(all_rmse),
        }
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Checkpoint speichern.
        
        Args:
            epoch: Epoch-Nummer
            val_loss: Validation Loss
            is_best: Ob dies das beste Modell ist
        """
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }
        
        # Regular checkpoint
        ckpt_path = self.run_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, ckpt_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.run_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved: {best_path}")
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        lr_scheduler = None,
    ):
        """
        Vollständiges Training.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Anzahl Epochs
            lr_scheduler: Optional Learning Rate Scheduler
        """
        print(f"\n{'='*70}")
        print(f"Starting training: {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Training...")
            
            train_metrics = self.train_epoch(train_loader)
            print(f"    Loss: {train_metrics['train_loss']:.6e}")
            
            print(f"  Validation...")
            val_metrics = self.validate(val_loader)
            print(f"    Loss: {val_metrics['val_loss']:.6e}")
            print(f"    MAE (missing): {val_metrics['val_mae']:.6e}")
            print(f"    RMSE (missing): {val_metrics['val_rmse']:.6e}")
            
            # History aktualisieren
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_mae'].append(val_metrics['val_mae'])
            self.history['val_rmse'].append(val_metrics['val_rmse'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Learning Rate anpassen
            if lr_scheduler:
                lr_scheduler.step(val_metrics['val_loss'])
            
            # Checkpoint speichern
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch + 1
            
            self.save_checkpoint(epoch + 1, val_metrics['val_loss'], is_best=is_best)
            print()
        
        print(f"{'='*70}")
        print(f"Training completed!")
        print(f"Best epoch: {self.best_epoch} (val_loss={self.best_val_loss:.6e})")
        print(f"{'='*70}\n")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Lade Modell und Optimizer von Checkpoint (zum Fortsetzen).
        
        Args:
            checkpoint_path: Pfad zu checkpoint.pt
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['val_loss']
        
        print(f"✓ Checkpoint loaded from: {checkpoint_path}")
        print(f"  Resuming from epoch: {checkpoint['epoch']}")
        print(f"  Best val_loss so far: {self.best_val_loss:.6f}")
    
    def save_history(self, save_path: Path = None, config: dict = None):
        """Training History speichern."""
        if save_path is None:
            save_path = self.run_dir / 'training_history.json'
        if config:
            payload = {
                'config': config,
                'history': self.history,
            }
        else:
            payload = self.history
        
        with open(save_path, 'w') as f:
            json.dump(payload, f, indent=2)
        
        print(f"History saved: {save_path}")


def main(args):
    """
    Haupttrainings-Script.
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Daten laden
    print("\n1. Loading data...")
    train_loader = create_dataloader(
        data_dir=args.train_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        overlap=args.overlap,
        damage_rate=args.damage_rate,
        min_damage_abs=args.min_damage_abs,
        valid_abs_eps=args.valid_abs_eps,
        min_valid_ratio=args.min_valid_ratio,
        damage_blocks_min=args.damage_blocks_min,
        damage_blocks_max=args.damage_blocks_max,
        damage_block_min=args.damage_block_min,
        damage_block_max=args.damage_block_max,
        max_missing_leads_per_timestep=args.max_missing_leads_per_timestep,
        protect_leads=args.protect_leads,
        mask_debug=args.mask_debug,
        mask_debug_every=args.mask_debug_every,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = None
    if args.val_dir:
        val_loader = create_dataloader(
            data_dir=args.val_dir,
            batch_size=args.batch_size,
            window_size=args.window_size,
            overlap=args.overlap,
            damage_rate=args.damage_rate,
            min_damage_abs=args.min_damage_abs,
            valid_abs_eps=args.valid_abs_eps,
            min_valid_ratio=args.min_valid_ratio,
            damage_blocks_min=args.damage_blocks_min,
            damage_blocks_max=args.damage_blocks_max,
            damage_block_min=args.damage_block_min,
            damage_block_max=args.damage_block_max,
            max_missing_leads_per_timestep=args.max_missing_leads_per_timestep,
            protect_leads=args.protect_leads,
            mask_debug=args.mask_debug,
            mask_debug_every=args.mask_debug_every,
            shuffle=False,
            num_workers=args.num_workers,
        )
    
    if val_loader is None:
        val_loader = train_loader
        print("  Using training data for validation (no val_dir provided)")
    
    # Modell
    print("\n2. Building model...")
    model = build_model(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        device=device,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Loss und Optimizer
    print("\n3. Setting up loss and optimizer...")
    loss_fn = CombinedMaskedLoss(
        loss_type=args.loss_type,
        w_missing=1.0,
        w_present=args.w_present,
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,  # More aggressive LR reduction
        min_lr=1e-6,
    )
    
    # Trainer
    checkpoint_dir = Path(args.checkpoint_dir)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        grad_clip_norm=args.grad_clip_norm,
        run_id=args.run_id,
    )
    
    # Resume vom Checkpoint?
    if args.resume_checkpoint:
        print("\n4. Resuming from checkpoint...")
        trainer.load_checkpoint(args.resume_checkpoint)
        start_epoch = trainer.history['train_loss'].__len__()
    else:
        start_epoch = 0
    
    # Training
    print("\n5. Starting training...\n")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr_scheduler=lr_scheduler,
    )
    
    # History speichern
    config = vars(args).copy()
    trainer.save_history(config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train TCN Repair Net for EKG NaN Imputation"
    )
    
    # Daten
    parser.add_argument('--train-dir', type=str, default='./data/train',
                        help='Directory with training EKG CSV files')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='Directory with validation EKG CSV files')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Windowing
    parser.add_argument('--window-size', type=int, default=1000,
                        help='Temporal window size (samples)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Window overlap factor (0-1)')
    parser.add_argument('--damage-rate', type=float, default=0.1,
                        help='Fraction of NaNs to inject synthetically')
    parser.add_argument('--min-damage-abs', type=float, default=1e-6,
                        help='Only inject damage where |y| exceeds this threshold')
    parser.add_argument('--valid-abs-eps', type=float, default=1e-6,
                        help='Values with |y| <= eps are treated as uninformative')
    parser.add_argument('--min-valid-ratio', type=float, default=0.05,
                        help='Minimum fraction of informative samples to keep a lead')
    parser.add_argument('--damage-blocks-min', type=int, default=1,
                        help='Minimum number of missing blocks per lead')
    parser.add_argument('--damage-blocks-max', type=int, default=3,
                        help='Maximum number of missing blocks per lead')
    parser.add_argument('--damage-block-min', type=int, default=100,
                        help='Minimum block length in samples')
    parser.add_argument('--damage-block-max', type=int, default=500,
                        help='Maximum block length in samples')
    parser.add_argument('--max-missing-leads-per-timestep', type=int, default=None,
                        help='Max number of leads missing at the same time (synthetic)')
    parser.add_argument('--protect-leads', type=str, default="",
                        help='Comma-separated lead names to never mask (e.g. II)')
    parser.add_argument('--mask-debug', action='store_true',
                        help='Enable mask debug stats in dataset')
    parser.add_argument('--mask-debug-every', type=int, default=0,
                        help='Print mask stats every N samples (0/1 = always)')
    
    # Modell
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Loss
    parser.add_argument('--loss-type', type=str, default='mse',
                        choices=['mse', 'huber'])
    parser.add_argument('--w-present', type=float, default=0.001,
                        help='Weight of loss on present values (stability, smaller = focus on missing)')
    
    # Optimizer
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                        help='Initial learning rate (recommend 5e-4 to 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--grad-clip-norm', type=float, default=5.0,
                        help='Max norm for gradient clipping')
    
    # Checkpoint
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Optional run name (default: timestamp)')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    if args.protect_leads:
        args.protect_leads = [s.strip() for s in args.protect_leads.split(",") if s.strip()]
    else:
        args.protect_leads = []
    main(args)
