import numpy as np
from pathlib import Path
import csv

import pandas as pd

from data import create_dataloader, load_ecg_csv, ECGDataset


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def debug_samples(
    out_dir: str = "debug_out",
    num_batches: int = 1,
    batch_size: int = 2,
    train_dir: str = "../data_split/train",
    window_size: int = 2000,
    overlap: float = 0.5,
    damage_rate: float = 0.1,
    sample_rate: float = 1000.0,
    plot: bool = True,
    layout_plot: bool = True,
    per_quarter: bool = False,
    record_id: str | None = None,
    metadata_path: str | None = None,
    max_missing_leads_per_timestep: int | None = None,
):
    if per_quarter:
        _debug_record_quarters(
            train_dir=train_dir,
            out_dir=out_dir,
            record_id=record_id,
            window_size=window_size,
            overlap=overlap,
            damage_rate=damage_rate,
            sample_rate=sample_rate,
            metadata_path=metadata_path,
            max_missing_leads_per_timestep=max_missing_leads_per_timestep,
        )
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    loader = create_dataloader(
        data_dir=train_dir,
        batch_size=batch_size,
        window_size=window_size,
        overlap=overlap,
        damage_rate=damage_rate,
        max_missing_leads_per_timestep=max_missing_leads_per_timestep,
        num_workers=0,
        shuffle=False,
    )

    for b_idx, batch in enumerate(loader):
        if b_idx >= num_batches:
            break

        x_filled = batch["x_filled"].numpy()
        mask = batch["mask"].numpy()
        y_target = batch["y_target"].numpy()
        record_ids = batch.get("record_id")
        window_starts = batch.get("window_start")

        for i in range(x_filled.shape[0]):
            record_id = None
            if record_ids is not None:
                record_id = record_ids[i]
            if not record_id:
                record_id = f"sample{i}"
            _save_csv(out_path / f"{record_id}_batch{b_idx}_x_filled.csv", x_filled[i])
            _save_csv(out_path / f"{record_id}_batch{b_idx}_mask.csv", mask[i])
            _save_csv(out_path / f"{record_id}_batch{b_idx}_y_target.csv", y_target[i])

            missing_ratio = (mask[i] == 0).mean()
            finite_ratio = np.isfinite(y_target[i]).mean()
            y_abs_mean = np.nanmean(np.abs(y_target[i]))
            print(
                f"sample {i}: missing_ratio={missing_ratio:.4f} "
                f"finite_ratio={finite_ratio:.4f} "
                f"y_abs_mean={y_abs_mean:.6f}"
            )

            if plot:
                if plt is None:
                    print("matplotlib not available; skipping plots")
                    plot = False
                    continue
                start_idx = int(window_starts[i]) if window_starts is not None else 0
                _plot_sample(
                    x_filled[i],
                    y_target[i],
                    mask[i],
                    sample_rate,
                    start_idx,
                    out_path / f"{record_id}_batch{b_idx}.png",
                )
            if layout_plot:
                if plt is None:
                    print("matplotlib not available; skipping layout plots")
                    layout_plot = False
                    continue
                start_idx = int(window_starts[i]) if window_starts is not None else 0
                _plot_layout_comparison(
                    y_target[i],
                    x_filled[i],
                    sample_rate,
                    start_idx,
                    out_path / f"{record_id}_batch{b_idx}_layout.png",
                )
            # per_quarter handled above for full-record plotting


def _plot_sample(
    x_filled: np.ndarray,
    y_target: np.ndarray,
    mask: np.ndarray,
    sample_rate: float,
    start_idx: int,
    out_path: Path,
    lead_indices: np.ndarray | None = None,
):
    leads, T = x_filled.shape
    if lead_indices is None:
        lead_indices = np.arange(leads)
    cols = 1
    rows = leads
    fig_width = 12
    fig_height = max(4.0, 2.4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True)
    axes = np.array(axes).reshape(-1)

    t = (np.arange(T) + start_idx) / sample_rate
    for lead in range(leads):
        ax = axes[lead]
        ax.plot(t, y_target[lead], color="gray", linewidth=0.8, label="y_target")
        ax.plot(t, x_filled[lead], color="blue", linewidth=0.8, label="x_filled")
        missing = mask[lead] == 0
        if missing.any():
            ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1], where=missing, color="red", alpha=0.1)
        lead_idx = int(lead_indices[lead])
        name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f"Lead {lead_idx+1}"
        ax.set_title(name)
        ax.grid(alpha=0.2)
        if lead == 0:
            ax.legend(fontsize=8, loc="upper right")
        if lead >= leads - cols:
            ax.set_xlabel("t (s)")

    for k in range(leads, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_layout_comparison(
    y_target: np.ndarray,
    x_filled: np.ndarray,
    sample_rate: float,
    start_idx: int,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    _plot_ecg_layout(axes[0], y_target, sample_rate, start_idx, title="Ground Truth")
    _plot_ecg_layout(axes[1], x_filled, sample_rate, start_idx, title="Masked")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_ecg_layout(
    ax,
    signal: np.ndarray,
    sample_rate: float,
    start_idx: int,
    title: str,
):
    lead_order = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]
    lead_index = {name: i for i, name in enumerate(LEAD_NAMES)}

    leads, T = signal.shape
    cols = 4
    segment_len = max(1, T // cols)
    segment_time = segment_len / sample_rate

    row_height = 2.0
    total_rows = 4
    y_min = -0.5 * row_height
    y_max = (total_rows - 1) * row_height + 0.5 * row_height
    t_offset = start_idx / sample_rate
    t_total = T / sample_rate

    _ecg_paper(ax, t_offset, t_offset + t_total, y_min, y_max)

    for r, row in enumerate(lead_order):
        y_base = (total_rows - 1 - r) * row_height
        for c, lead_name in enumerate(row):
            idx = lead_index.get(lead_name)
            if idx is None or idx >= leads:
                continue
            start = c * segment_len
            end = min(start + segment_len, T)
            if end <= start:
                continue
            t = (np.arange(end - start) / sample_rate) + c * segment_time + t_offset
            ax.plot(t, signal[idx, start:end] + y_base, color="black", linewidth=0.8)
            ax.text(
                c * segment_time + 0.02,
                y_base + 0.6,
                lead_name,
                fontsize=9,
                va="bottom",
                ha="left",
                color="black",
            )

    rhythm_name = "II"
    rhythm_idx = lead_index.get(rhythm_name, None)
    if rhythm_idx is not None and rhythm_idx < leads:
        y_base = 0.0
        t = (np.arange(T) + start_idx) / sample_rate
        ax.plot(t, signal[rhythm_idx] + y_base, color="black", linewidth=0.8)
        ax.text(0.02, y_base + 0.6, rhythm_name, fontsize=9, va="bottom", ha="left")

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("mV")
    ax.set_xlabel("t (s)")
    ax.set_yticks([])


def _ecg_paper(ax, t_min: float, t_max: float, y_min: float, y_max: float):
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(y_min, y_max)
    ax.set_facecolor("#fff7f7")

    minor_time = 0.04
    major_time = 0.2
    minor_mv = 0.1
    major_mv = 0.5

    ax.set_xticks(np.arange(t_min, t_max + major_time, major_time))
    ax.set_xticks(np.arange(t_min, t_max + minor_time, minor_time), minor=True)
    ax.set_yticks(np.arange(y_min, y_max + major_mv, major_mv))
    ax.set_yticks(np.arange(y_min, y_max + minor_mv, minor_mv), minor=True)

    ax.grid(which="minor", color="#ffcccc", linewidth=0.3)
    ax.grid(which="major", color="#ff6666", linewidth=0.6)
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _debug_record_quarters(
    train_dir: str,
    out_dir: str,
    record_id: str | None,
    window_size: int,
    overlap: float,
    damage_rate: float,
    sample_rate: float,
    metadata_path: str | None,
    max_missing_leads_per_timestep: int | None,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    record_path, actual_id = _resolve_record_path(train_dir, record_id)
    if record_path is None:
        print("No record found to plot")
        return

    y = _load_record_signal(record_path, actual_id, sample_rate, metadata_path)
    if y is None:
        print(f"Failed to load record: {record_path}")
        return

    dataset = ECGDataset(
        x_list=None,
        y_list=[y],
        window_size=window_size,
        overlap=0.0,
        damage_rate=damage_rate,
        max_missing_leads_per_timestep=max_missing_leads_per_timestep,
    )

    T = y.shape[1]
    quarter = T // 4
    if quarter <= 0:
        print("Record too short for quarter plots")
        return

    for q in range(4):
        s = q * quarter
        e = T if q == 3 else (q + 1) * quarter
        if e - s < window_size:
            print(f"Quarter {q+1} too short for window_size={window_size}; skipping")
            continue
        start_idx = s
        if start_idx + window_size > e:
            start_idx = e - window_size
        y_window = y[:, start_idx:start_idx + window_size]
        x_filled, mask, y_target, _, _ = dataset._create_damage(y_window)

        lead_indices = np.where(np.isfinite(y_window).any(axis=1))[0]
        if lead_indices.size == 0:
            print(f"Quarter {q+1} has no finite leads; skipping plot")
            continue
        x_plot = x_filled[lead_indices]
        y_plot = y_target[lead_indices]
        mask_plot = mask[lead_indices]

        _save_csv(out_path / f"{actual_id}_quarter{q+1}_x_filled.csv", x_filled)
        _save_csv(out_path / f"{actual_id}_quarter{q+1}_mask.csv", mask)
        _save_csv(out_path / f"{actual_id}_quarter{q+1}_y_target.csv", y_target)

        if plt is None:
            continue
        _plot_sample(
            x_plot,
            y_plot,
            mask_plot,
            sample_rate,
            start_idx,
            out_path / f"{actual_id}_quarter{q+1}.png",
            lead_indices=lead_indices,
        )
        _plot_quarter_windows(
            y,
            sample_rate,
            s,
            e,
            window_size,
            overlap,
            out_path / f"{actual_id}_quarter{q+1}_windows.png",
            lead_idx=int(lead_indices[0]),
        )


def _resolve_record_path(train_dir: str, record_id: str | None):
    data_path = Path(train_dir)
    if record_id:
        nested = data_path / record_id / f"{record_id}.csv"
        flat = data_path / f"{record_id}.csv"
        if nested.exists():
            return nested, record_id
        if flat.exists():
            return flat, record_id
        return None, None

    nested_files = sorted(data_path.glob("*/*.csv"))
    if nested_files:
        rec = nested_files[0]
        return rec, rec.parent.name
    flat_files = sorted(data_path.glob("*.csv"))
    if flat_files:
        rec = flat_files[0]
        return rec, rec.stem
    return None, None


def _load_record_signal(
    record_path: Path,
    record_id: str,
    target_fs: float,
    metadata_path: str | None,
):
    if metadata_path is None:
        candidate = record_path.parent.parent / "meta_data.csv"
        if candidate.exists():
            metadata_path = str(candidate)

    target_length = None
    if metadata_path is not None and Path(metadata_path).exists():
        meta_df = pd.read_csv(metadata_path)
        match = meta_df[meta_df["id"].astype(str) == str(record_id)]
        if not match.empty:
            fs = float(match.iloc[0]["fs"])
            sig_len = int(match.iloc[0]["sig_len"])
            target_length = int(round(sig_len * (target_fs / fs)))

    return load_ecg_csv(str(record_path), target_length=target_length)


def _plot_quarter_windows(
    signal: np.ndarray,
    sample_rate: float,
    quarter_start: int,
    quarter_end: int,
    window_size: int,
    overlap: float,
    out_path: Path,
    lead_idx: int | None = None,
):
    if plt is None:
        return
    stride = max(1, int(window_size * (1 - overlap)))
    if quarter_end - quarter_start < window_size:
        return

    if lead_idx is None:
        lead_idx = 1 if signal.shape[0] > 1 else 0
    t = np.arange(quarter_start, quarter_end) / sample_rate
    y = signal[lead_idx, quarter_start:quarter_end]

    fig, ax = plt.subplots(figsize=(12, 3))
    _ecg_paper(ax, t[0], t[-1], np.nanmin(y) - 0.1, np.nanmax(y) + 0.1)
    ax.plot(t, y, color="black", linewidth=0.8, label=LEAD_NAMES[lead_idx])

    for start in range(quarter_start, quarter_end - window_size + 1, stride):
        end = start + window_size
        ax.axvspan(start / sample_rate, end / sample_rate, color="blue", alpha=0.08)
        ax.axvline(start / sample_rate, color="blue", alpha=0.2, linewidth=0.5)
        ax.axvline(end / sample_rate, color="blue", alpha=0.2, linewidth=0.5)

    ax.set_title("Quarter windows")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("mV")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_csv(path: Path, arr: np.ndarray):
    header = ["t"] + LEAD_NAMES[: arr.shape[0]]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t_idx in range(arr.shape[1]):
            row = [str(t_idx)]
            for val in arr[:, t_idx]:
                if not np.isfinite(val):
                    row.append("")
                else:
                    row.append(_format_value(val))
            writer.writerow(row)


def _format_value(val: float) -> str:
    if val == 0:
        return "0"
    text = f"{val:.10f}".rstrip("0").rstrip(".")
    return text if text else "0"


if __name__ == "__main__":
    debug_samples(per_quarter=True, max_missing_leads_per_timestep=1)
