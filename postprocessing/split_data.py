"""
Automatischer Datensplit für EKG-Training.

Kopiert CSVs aus Source-Directory in Train/Val/Test-Verzeichnisse (z.B. 60/20/20 split).
"""

import numpy as np
from pathlib import Path
from typing import Tuple
import shutil
from tqdm import tqdm
import argparse


def split_data(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[int, int, int]:
    """
    Split EKG-Dateien in Train/Val/Test-Verzeichnisse.
    
    Args:
        source_dir: Verzeichnis mit EKG-CSVs (z.B. train/[id]/[id].csv)
        output_dir: Output-Verzeichnis (wird erstellt)
        train_ratio: Anteil für Training (default 0.6)
        val_ratio: Anteil für Validierung (default 0.2)
        test_ratio: Anteil für Testing (default 0.2)
        random_seed: Für reproduzierbare Splits
        
    Returns:
        (num_train, num_val, num_test) – Anzahl Dateien in jedem Split
        
    Raises:
        ValueError: Falls Ratios sich nicht zu 1.0 summieren
    """
    
    # Validiere Ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Ratios müssen sich zu 1.0 summieren, "
            f"aber {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )
    
    np.random.seed(random_seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Ausgabe-Verzeichnisse erstellen
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # CSVs finden (hierarchisch: train/[id]/[id].csv oder flat)
    csv_files = list(source_dir.glob('*/*.csv')) or list(source_dir.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien in {source_dir} gefunden")
    
    print(f"Gefunden: {len(csv_files)} EKG-Dateien")
    
    # Shufflen und splitten
    indices = np.arange(len(csv_files))
    np.random.shuffle(indices)
    
    train_end = int(len(csv_files) * train_ratio)
    val_end = train_end + int(len(csv_files) * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Helper-Funktion zum Kopieren
    def copy_files(indices, target_dir, split_name):
        print(f"\nKopiere zu {split_name} ({len(indices)} Dateien)...")
        for idx in tqdm(indices):
            src = csv_files[idx]
            # Struktur beibehalten oder flach kopieren
            if src.parent.name != source_dir.name:
                # Hierarchische Struktur: train/[id]/[id].csv
                id_dir = target_dir / src.parent.name
                id_dir.mkdir(exist_ok=True)
                dst = id_dir / src.name
            else:
                # Flat: nur Dateinamen
                dst = target_dir / src.name
            shutil.copy2(src, dst)
    
    copy_files(train_indices, train_dir, "Train")
    copy_files(val_indices, val_dir, "Val")
    copy_files(test_indices, test_dir, "Test")
    
    print(f"\n{'='*60}")
    print(f"Split abgeschlossen!")
    print(f"Train: {len(train_indices)} Dateien ({100*train_ratio:.0f}%) → {train_dir}")
    print(f"Val:   {len(val_indices)} Dateien ({100*val_ratio:.0f}%) → {val_dir}")
    print(f"Test:  {len(test_indices)} Dateien ({100*test_ratio:.0f}%) → {test_dir}")
    print(f"{'='*60}")
    
    return len(train_indices), len(val_indices), len(test_indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split EKG-Daten in Train/Val/Test"
    )
    parser.add_argument('--source-dir', type=str, required=True,
                        help='Source-Verzeichnis mit EKG-CSVs')
    parser.add_argument('--output-dir', type=str, default='./data_split',
                        help='Output-Verzeichnis')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                        help='Anteil für Training (0-1)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Anteil für Validierung (0-1)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Anteil für Testing (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed für Reproduzierbarkeit')
    
    args = parser.parse_args()
    
    split_data(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )