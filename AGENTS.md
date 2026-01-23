# AGENTS.md - Coding Agent Guidelines

This document provides guidelines for AI coding agents working in this repository.

## Project Overview

EKG/ECG image digitization project using PyTorch. The model transforms noisy EKG images (0002-0012) into clean target images (0001) using a Lightweight Attention U-Net architecture with depthwise separable convolutions and CBAM attention.

## Project Structure

```
├── src/                    # Main source code
│   ├── main.py            # Entry point - CLI for training
│   ├── model.py           # LightweightAttentionUNet model definition
│   ├── data.py            # EKGDataset class and preprocessing utilities
│   ├── train.py           # Training loop, validation, logging
│   └── util.py            # CombinedLoss function, utilities
├── data/                   # Data directory (not in git)
├── points/                # Training logs and checkpoints (not in git)
├── requirements.txt       # Python dependencies
└── flake.nix              # Nix development environment
```

## Environment Setup

```bash
# Using Nix (recommended)
nix develop

# Using virtualenv
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Python version:** 3.13 | **Key deps:** PyTorch 2.5.1+cu121, numpy, Pillow, matplotlib, tqdm

## Build/Run Commands

```bash
# Default training
python src/main.py --data-dir data/train

# Custom training
python src/main.py --data-dir data/train --epochs 50 --batch-size 8
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/train` | Training data directory |
| `--epochs` | `30` | Training epochs |
| `--batch-size` | `4` | Batch size |

## Testing

**No test framework configured.** If adding tests:

```bash
pip install pytest
pytest                              # Run all tests
pytest tests/test_model.py          # Single file
pytest tests/test_model.py::test_fn # Single function
```

## Linting

**No linting tools configured.** Recommended setup:

```bash
pip install ruff
ruff check src/       # Lint
ruff format src/      # Format
ruff check --fix src/ # Auto-fix
```

## Code Style Guidelines

### Import Organization
Stdlib/third-party mixed, local imports separated by blank line:

```python
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from model import LightweightAttentionUNet
```

### Naming Conventions
- **Functions/variables:** `snake_case` (`remove_grid_lines`, `train_loss`)
- **Classes:** `PascalCase` (`EKGDataset`, `LightweightAttentionUNet`)
- **Private methods:** `_prefix` (`_prepare_dataset`, `_init_weights`)

### Type Hints
**Not used** in this codebase. Optional for new code.

### Docstrings (Google-style)
```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Run one training epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
    
    Returns:
        Average training loss for the epoch
    """
```

### Error Handling
- Defensive checks with `print()` warnings, not exceptions
- Try/except for compatibility only

```python
if not data_dir.exists():
    print(f"Warning: Data directory {data_dir} does not exist")
    return default_value
```

### Logging
Use custom `log()` from `train.py` (writes to console + markdown file):
```python
from train import log
log("Message to console and log file")
```

## PyTorch Patterns

```python
# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training
model.train()
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Validation
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

## Model Architecture

`LightweightAttentionUNet`:
- Input/Output: 1-channel grayscale (1024x1024)
- Features: Depthwise separable convs, CBAM attention, attention gates
- Parameters: ~1.5-2M (vs ~30M standard U-Net)
- Output activation: Tanh (values -1 to 1)

## Data Pipeline

1. Load from subject directories
2. Convert to grayscale
3. Remove grid lines (threshold-based, targets only)
4. Resize with padding to 1024x1024
5. Normalize to [-1, 1]

## Checkpoints

Saved to `points/`:
- `best_model.pth` - Best weights
- `training_log.md` - Training log
- `samples/` - Prediction visualizations
