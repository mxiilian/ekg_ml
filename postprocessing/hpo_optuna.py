"""
Hyperparameter-Optimierung fuer das TCN-Repair-Net mit Optuna.

Beispiel:
python /workspace/postprocessing/hpo_optuna.py \
  --train-dir /workspace/data_split/train \
  --val-dir /workspace/data_split/val \
  --checkpoint-dir /workspace/postprocessing/checkpoints \
  --n-trials 10
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import optuna
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import create_dataloader
from loss import CombinedMaskedLoss
from model import build_model
from train import Trainer


def run_trial(
    trial: optuna.Trial,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Run a single trial and return best val_loss."""
    learning_rate = trial.suggest_float(
        "learning_rate", args.lr_min, args.lr_max, log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", args.wd_min, args.wd_max, log=True
    )
    dropout = trial.suggest_float("dropout", args.dropout_min, args.dropout_max)
    hidden_channels = trial.suggest_categorical(
        "hidden_channels", args.hidden_channels
    )
    num_blocks = trial.suggest_int("num_blocks", args.num_blocks_min, args.num_blocks_max)
    w_present = trial.suggest_float("w_present", 0.0, args.w_present_max)

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

    model = build_model(
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dropout=dropout,
        device=device,
    )

    loss_fn = CombinedMaskedLoss(
        loss_type=args.loss_type,
        w_missing=1.0,
        w_present=w_present,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    run_id = f"hpo_trial_{trial.number}"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
        grad_clip_norm=args.grad_clip_norm,
        run_id=run_id,
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.num_epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        trainer.history['train_loss'].append(train_metrics['train_loss'])
        trainer.history['val_loss'].append(val_metrics['val_loss'])
        trainer.history['val_mae'].append(val_metrics['val_mae'])
        trainer.history['val_rmse'].append(val_metrics['val_rmse'])
        trainer.history['learning_rate'].append(
            optimizer.param_groups[0]['lr']
        )

        lr_scheduler.step(val_metrics['val_loss'])

        current_val = val_metrics['val_loss']
        if current_val + args.es_min_delta < best_val_loss:
            best_val_loss = current_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        trial.report(current_val, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if args.early_stop and epochs_no_improve >= args.es_patience:
            break

    config: Dict[str, object] = vars(args).copy()
    config.update(trial.params)
    config['trial_number'] = trial.number
    config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save_history(config=config)

    return best_val_loss


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for TCN Repair Net"
    )

    # Data
    parser.add_argument('--train-dir', type=str, default='./data/train')
    parser.add_argument('--val-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)

    # Windowing / damage
    parser.add_argument('--window-size', type=int, default=2000)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--damage-rate', type=float, default=0.12)
    parser.add_argument('--min-damage-abs', type=float, default=0.0)
    parser.add_argument('--valid-abs-eps', type=float, default=0.0)
    parser.add_argument('--min-valid-ratio', type=float, default=0.05)
    parser.add_argument('--damage-blocks-min', type=int, default=1)
    parser.add_argument('--damage-blocks-max', type=int, default=3)
    parser.add_argument('--damage-block-min', type=int, default=20)
    parser.add_argument('--damage-block-max', type=int, default=120)
    parser.add_argument('--max-missing-leads-per-timestep', type=int, default=None)
    parser.add_argument('--protect-leads', type=str, default="")
    parser.add_argument('--mask-debug', action='store_true')
    parser.add_argument('--mask-debug-every', type=int, default=0)

    # Model
    parser.add_argument('--hidden-channels', type=int, nargs='+', default=[32, 64, 96, 128])
    parser.add_argument('--num-blocks-min', type=int, default=3)
    parser.add_argument('--num-blocks-max', type=int, default=5)
    parser.add_argument('--dropout-min', type=float, default=0.0)
    parser.add_argument('--dropout-max', type=float, default=0.4)

    # Loss
    parser.add_argument('--loss-type', type=str, default='mse', choices=['mse', 'huber'])
    parser.add_argument('--w-present-max', type=float, default=0.01)

    # Optimizer
    parser.add_argument('--lr-min', type=float, default=1e-4)
    parser.add_argument('--lr-max', type=float, default=5e-4)
    parser.add_argument('--wd-min', type=float, default=1e-6)
    parser.add_argument('--wd-max', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--grad-clip-norm', type=float, default=5.0)

    # HPO
    parser.add_argument('--n-trials', type=int, default=10)
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--es-patience', type=int, default=5)
    parser.add_argument('--es-min-delta', type=float, default=0.0)

    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--study-name', type=str, default='tcn_hpo')

    args = parser.parse_args()

    if args.protect_leads:
        args.protect_leads = [s.strip() for s in args.protect_leads.split(',') if s.strip()]
    else:
        args.protect_leads = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: run_trial(trial, args, device),
        n_trials=args.n_trials,
    )

    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")


if __name__ == '__main__':
    main()
