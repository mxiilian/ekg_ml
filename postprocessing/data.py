"""
Datenmodul für EKG-NaN-Imputation
- Laden von Ground-Truth-Daten (12-Lead-EKG)
- Synthetische Beschädigung (NaN-Erzeugung)
- Windowing mit Overlap
- Maskenerzeugung
- Resampling basiert auf Open-ECG-Digitizer
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample  # ← NEU: wie im Open-ECG-Digitizer


def resample_to_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """
    EKG Resampling mit scipy.signal.resample (wie im Open-ECG-Digitizer).
    
    Args:
        signal: (C, T) – EKG-Daten (C=Leads, T=Samples)
        target_length: Ziel-Länge in Samples
        
    Returns:
        Resampled data: (C, target_length)
    """
    leads, _ = signal.shape

    out = []
    for lead in range(leads):
        x = signal[lead]
        finite = np.isfinite(x)
        if finite.sum() == 0:
            out.append(np.full(target_length, np.nan, dtype=np.float32))
            continue

        # Interpolate missing values for resampling
        x_filled = x.astype(np.float64).copy()
        if not finite.all():
            idx = np.arange(x.size)
            x_filled[~finite] = np.interp(idx[~finite], idx[finite], x_filled[finite])

        y_resampled = resample(x_filled, target_length)

        # Resample finite mask and restore NaNs outside valid regions
        mask = finite.astype(np.float64)
        mask_resampled = resample(mask, target_length)
        y_resampled[mask_resampled < 0.5] = np.nan

        out.append(y_resampled.astype(np.float32))

    return np.stack(out)


class ECGDataset(Dataset):
    """
    Dataset für EKG-NaN-Imputation.
    
    Input: x_filled (12, T), mask m (12, T)
    Output: y_target (12, T)
    
    Modell-Input: concat(x_filled, mask) → (24, T)
    """
    
    def __init__(
        self,
        x_list: List[np.ndarray],
        y_list: List[np.ndarray],
        id_list: Optional[List[str]] = None,
        window_size: int = 1000,
        overlap: float = 0.5,
        sample_rate: int = 500,
        damage_rate: float = 0.1,
        min_damage_abs: float = 1e-6,
        valid_abs_eps: float = 1e-6,
        min_valid_ratio: float = 0.05,
        damage_blocks_min: int = 1,
        damage_blocks_max: int = 3,
        damage_block_min: int = 100,
        damage_block_max: int = 500,
        max_missing_leads_per_timestep: Optional[int] = None,
        protect_leads: Optional[List[str]] = None,
        mask_debug: bool = False,
        mask_debug_every: int = 0,
    ):
        """
        Args:
            x_list: Liste von beschädigten EKGs (optional), oder None → synthetisch beschädigen
            y_list: Liste von Ground-Truth-EKGs (12, T)
            window_size: Zeitfenster in Samples (default: 1000 = 2s bei 500Hz)
            overlap: Überlapfaktor (0.5 = 50%)
            sample_rate: Sampling-Rate in Hz
            damage_rate: Anteil NaN-Werte in synthetischen Schäden (0.0-1.0)
        """
        self.y_list = y_list  # Ground Truth
        self.x_list = x_list  # Optional: echte Digitizer-Outputs
        self.id_list = id_list
        self.window_size = window_size
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.damage_rate = damage_rate
        self.min_damage_abs = min_damage_abs
        self.valid_abs_eps = valid_abs_eps
        self.min_valid_ratio = min_valid_ratio
        self.damage_blocks_min = damage_blocks_min
        self.damage_blocks_max = damage_blocks_max
        self.damage_block_min = damage_block_min
        self.damage_block_max = damage_block_max
        self.max_missing_leads_per_timestep = max_missing_leads_per_timestep
        self.protect_leads = protect_leads or []
        self.mask_debug = mask_debug
        self.mask_debug_every = mask_debug_every
        self._mask_debug_count = 0
        
        self.windows = []
        self._create_windows()
    
    def _create_windows(self):
        """Erstelle Fenster mit Overlap aus allen EKGs."""
        stride = int(self.window_size * (1 - self.overlap))
        
        for idx, y in enumerate(self.y_list):
            # y shape: (12, T)
            T = y.shape[1]
            
            # Fenster erstellen
            for start in range(0, T - self.window_size + 1, stride):
                end = start + self.window_size
                y_window = y[:, start:end]
                
                self.windows.append({
                    'y_idx': idx,
                    'start': start,
                    'end': end,
                    'y_window': y_window,
                    'record_id': self.id_list[idx] if self.id_list else None,
                })
    
    def _create_damage(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthetische Beschädigung mit zusammenhängenden Blöcken (realistisch).
        
        Args:
            y: Ground-Truth (12, T)
            
        Returns:
            x_filled: EKG mit NaNs als 0 gefüllt
            mask: Binary Mask, 1=vorhanden, 0=fehlt
            y_target: Ground Truth mit NaN an uninformative Stellen
        """
        x = y.copy()
        # Per-lead normalization (z-score) for stable training
        lead_mean = np.nanmean(x, axis=1, keepdims=True)
        lead_std = np.nanstd(x, axis=1, keepdims=True)
        lead_mean = np.nan_to_num(lead_mean, nan=0.0)
        lead_std = np.nan_to_num(lead_std, nan=1.0)
        lead_std = np.where(lead_std < 1e-6, 1.0, lead_std)
        x = (x - lead_mean) / lead_std
        y_norm = x.copy()
        # Informative Stellen: nur finite Werte (0 mV bleibt erhalten)
        informative_mask = np.isfinite(x)

        # Targets: uninformative Stellen als NaN markieren, damit sie nicht im Loss landen
        y_target = y_norm.copy()
        y_target[~informative_mask] = np.nan
        
        # Mask: 1 = vorhandener informativer GT-Wert, 0 = fehlend/invalid
        mask = informative_mask.astype(np.float32)
        
        C, T = x.shape  # 12 Leads, T Samples
        lead_name_map = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        protected_idx = {i for i, name in enumerate(lead_name_map) if name in self.protect_leads}
        allowed_mask = informative_mask.copy()
        for idx in protected_idx:
            if idx < C:
                allowed_mask[idx] = False

        target_total = int(round(self.damage_rate * allowed_mask.sum()))
        if target_total <= 0:
            x_filled = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            mask = np.where(np.isfinite(mask), mask, 0.0).astype(np.float32)
            return x_filled, mask, y_target

        masked = np.zeros_like(allowed_mask, dtype=bool)
        remaining = target_total

        if np.random.rand() < 0.05:
            lead_candidates = [i for i in range(C) if allowed_mask[i].any()]
            if lead_candidates:
                c = np.random.choice(lead_candidates)
                _mask_block(
                    allowed_mask,
                    masked,
                    x,
                    mask,
                    y,
                    c,
                    remaining,
                    long_block=True,
                )
                remaining = target_total - masked.sum()

        max_iters = 200
        iters = 0
        while remaining > 0 and iters < max_iters:
            iters += 1
            lead_candidates = [i for i in range(C) if allowed_mask[i].any()]
            if not lead_candidates:
                break
            c = np.random.choice(lead_candidates)
            _mask_block(
                allowed_mask,
                masked,
                x,
                mask,
                y,
                c,
                remaining,
                long_block=False,
            )
            remaining = target_total - masked.sum()
        
        if self.mask_debug:
            self._mask_debug_count += 1
            if self.mask_debug_every <= 1 or self._mask_debug_count % self.mask_debug_every == 0:
                _log_mask_stats(mask, informative_mask, lead_name_map)
    
        # NaNs/Inf durch 0 ersetzen fuer Modell (harte Absicherung)
        x_filled = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.where(np.isfinite(mask), mask, 0.0).astype(np.float32)
        
        return x_filled, mask, y_target, lead_mean, lead_std
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'x_filled': (12, T) – EKG mit NaNs als 0 gefüllt
                'mask': (12, T) – Binary Mask (1=vorhanden, 0=fehlt)
                'y_target': (12, T) – Ground Truth
            }
        """
        window_info = self.windows[idx]
        y_window = window_info['y_window']
        
        # Beschädigung erstellen
        x_filled, mask, y_target, lead_mean, lead_std = self._create_damage(y_window)
        
        # Zu Tensoren konvertieren
        x_filled_t = torch.from_numpy(x_filled).float()
        mask_t = torch.from_numpy(mask).float()
        y_target_t = torch.from_numpy(y_target).float()
        
        return {
            'x_filled': x_filled_t,
            'mask': mask_t,
            'y_target': y_target_t,
            'record_id': window_info.get('record_id'),
            'window_start': int(window_info.get('start', 0)),
            'lead_mean': torch.from_numpy(lead_mean).float(),
            'lead_std': torch.from_numpy(lead_std).float(),
        }


def load_ecg_csv(
    filepath: str,
    sample_names: List[str] = None,
    target_length: Optional[int] = None,
) -> np.ndarray:
    """
    EKG aus CSV laden mit optionalem Resampling.
    
    Args:
        filepath: Pfad zur CSV-Datei
        sample_names: Optionale Spaltennamen (12 Leads)
        target_length: Optionale Ziel-Länge für Resampling
        
    Returns:
        Array (12, T) - möglicherweise resampled
    """
    # Zwei Möglichkeiten: pandas oder numpy (wie Open-ECG-Digitizer)
    df = pd.read_csv(filepath)
    
    if sample_names is None:
        sample_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    available_names = [name for name in sample_names if name in df.columns]
    data = df[available_names].values.T  # (T, C) -> (C, T)
    data = data.astype(np.float32)
    
    # Resampling falls gewünscht
    if target_length is not None and data.shape[1] != target_length:
        data = resample_to_length(data, target_length)
    
    return data


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    window_size: int = 1000,
    overlap: float = 0.5,
    damage_rate: float = 0.1,
    min_damage_abs: float = 1e-6,
    valid_abs_eps: float = 1e-6,
    min_valid_ratio: float = 0.05,
    damage_blocks_min: int = 1,
    damage_blocks_max: int = 3,
    damage_block_min: int = 100,
    damage_block_max: int = 500,
    max_missing_leads_per_timestep: Optional[int] = None,
    protect_leads: Optional[List[str]] = None,
    mask_debug: bool = False,
    mask_debug_every: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    target_length: Optional[int] = 10000,  # Default fallback length
    target_fs: int = 1000,
    metadata_path: Optional[str] = None,
) -> DataLoader:
    """
    DataLoader mit Resampling auf Ziel-FS (default 1000 Hz).
    
    Args:
        data_dir: Verzeichnis mit EKG-CSV-Dateien (hierarchisch oder flat)
        batch_size: Batch-Größe
        window_size: Zeitfenster
        overlap: Overlap-Faktor
        damage_rate: NaN-Anteil bei synthetischer Beschädigung
        shuffle: Ob Daten mischen
        num_workers: Anzahl Worker für Datenladen
        target_length: Optionale Ziel-Länge (Fallback)
        target_fs: Ziel-Samplingrate in Hz
        metadata_path: Optionaler Pfad zu meta_data.csv
        protect_leads: Liste von Lead-Namen, die nicht maskiert werden sollen
        mask_debug: Debug-Stats fuer Maskierung aktivieren
        mask_debug_every: Alle N Samples Debug ausgeben (0/1 = immer)
    """
    data_path = Path(data_dir)

    meta_map = None
    if metadata_path is None:
        candidate = data_path.parent / "meta_data.csv"
        if candidate.exists():
            metadata_path = str(candidate)
    if metadata_path is not None and Path(metadata_path).exists():
        meta_df = pd.read_csv(metadata_path)
        meta_map = {
            str(row["id"]): (float(row["fs"]), int(row["sig_len"]))
            for _, row in meta_df.iterrows()
        }
    
    # Suche hierarchisch (train/[id]/[id].csv) ODER flat (*.csv)
    csv_files = sorted(data_path.glob('*/*.csv'))
    if not csv_files:
        csv_files = sorted(data_path.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien in {data_dir} gefunden")
    
    print(f"Lade {len(csv_files)} EKG-Dateien von {data_path}...")
    y_list = []
    id_list = []
    for csv_file in csv_files:
        try:
            if csv_file.parent != data_path:
                record_id = csv_file.parent.name
            else:
                record_id = csv_file.stem

            per_record_target = target_length
            if meta_map is not None and record_id in meta_map:
                fs, sig_len = meta_map[record_id]
                per_record_target = int(round(sig_len * (target_fs / fs)))

            y = load_ecg_csv(str(csv_file), target_length=per_record_target)  # ← Resampling hier
            y_list.append(y)
            id_list.append(record_id)
        except Exception as e:
            print(f"  ⚠ Fehler beim Laden {csv_file}: {e}")
    
    print(f"  ✓ Erfolgreich geladen: {len(y_list)}/{len(csv_files)} EKGs")
    
    dataset = ECGDataset(
        x_list=None,  # Synthetisch beschädigen
        y_list=y_list,
        id_list=id_list,
        window_size=window_size,
        overlap=overlap,
        damage_rate=damage_rate,
        min_damage_abs=min_damage_abs,
        valid_abs_eps=valid_abs_eps,
        min_valid_ratio=min_valid_ratio,
        damage_blocks_min=damage_blocks_min,
        damage_blocks_max=damage_blocks_max,
        damage_block_min=damage_block_min,
        damage_block_max=damage_block_max,
        max_missing_leads_per_timestep=max_missing_leads_per_timestep,
        protect_leads=protect_leads,
        mask_debug=mask_debug,
        mask_debug_every=mask_debug_every,
    )
    
    print(f"  ✓ Erstellt: {len(dataset)} Trainings-Fenster")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


if __name__ == '__main__':
    # Test-Code
    print("Data module loaded successfully!")


def _sample_block_len(long_block: bool = False) -> int:
    if long_block:
        return np.random.randint(250, 801)
    r = np.random.rand()
    if r < 0.70:
        return np.random.randint(10, 41)
    if r < 0.95:
        return np.random.randint(40, 121)
    return np.random.randint(120, 301)


def _mask_block(
    allowed_mask: np.ndarray,
    masked: np.ndarray,
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    lead_idx: int,
    remaining: int,
    long_block: bool = False,
):
    C, T = allowed_mask.shape
    avail = allowed_mask[lead_idx] & ~masked[lead_idx]
    if not avail.any():
        return
    start = np.random.choice(np.where(avail)[0])
    max_len = 1
    while start + max_len < T and allowed_mask[lead_idx, start + max_len] and not masked[lead_idx, start + max_len]:
        max_len += 1
    block_len = min(_sample_block_len(long_block), remaining, max_len)
    if block_len <= 0:
        return
    end = start + block_len
    masked[lead_idx, start:end] = True
    x[lead_idx, start:end] = np.nan
    mask[lead_idx, start:end] = 0.0


def _log_mask_stats(mask: np.ndarray, informative_mask: np.ndarray, lead_names: List[str]):
    valid = informative_mask.astype(bool)
    if valid.any():
        missing_ratio_total = ((mask == 0) & valid).sum() / valid.sum()
    else:
        missing_ratio_total = 0.0
    per_lead = []
    for lead in range(mask.shape[0]):
        v = valid[lead]
        if v.any():
            per_lead.append(((mask[lead] == 0) & v).sum() / v.sum())
        else:
            per_lead.append(0.0)
    missing_ratio_per_lead = per_lead
    max_gaps = []
    for lead in range(mask.shape[0]):
        run = 0
        max_run = 0
        for v in (mask[lead] == 0):
            if v:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        max_gaps.append(max_run)
    print(
        f"Mask debug: total={missing_ratio_total:.4f} "
        f"per_lead={[round(v,4) for v in missing_ratio_per_lead]} "
        f"max_gap={max_gaps}"
    )
