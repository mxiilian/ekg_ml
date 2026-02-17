import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importiere deine Funktionen/Klassen (Pfad ggf. anpassen)
from data import load_ecg_csv, ECGDataset  # <- anpassen, falls Datei anders heißt

def plot_one_window(
    y_target,
    x_filled,
    mask,
    leads_to_plot=None,
    title="ECG mask debug",
    valid_abs_eps=1e-6,
    min_valid_ratio=0.05,
):
    """
    y_target, x_filled, mask: (12, T)
    mask: 1=vorhanden, 0=fehlt
    """
    T = y_target.shape[1]
    t = np.arange(T)

    if leads_to_plot is None:
        leads_to_plot = tuple(np.where(np.isfinite(y_target).any(axis=1))[0].tolist())
    else:
        leads_to_plot = tuple([c for c in leads_to_plot if np.isfinite(y_target[c]).any()])

    n = len(leads_to_plot)
    if n == 0:
        print("No leads with finite data in this window; skipping plot")
        return None
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5*n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, leads_to_plot):
        y = y_target[c]
        x = x_filled[c]
        m = mask[c].astype(bool)
        finite_mask = np.isfinite(y)
        informative_mask = finite_mask & (np.abs(y) > valid_abs_eps)
        informative_ratio = informative_mask.mean()
        lead_invalid = informative_ratio < min_valid_ratio

        # Ground truth
        ax.plot(t, y, linewidth=1.0, label="y_target (GT)")

        # Beschädigtes Signal
        ax.plot(t, x, linewidth=1.0, label="x_filled (input)")

        # Missing-Stellen markieren (mask==0)
        missing_idx = np.where(~m)[0]
        if missing_idx.size > 0:
            ax.scatter(missing_idx, y[missing_idx], s=10, label="missing positions (mask==0) [GT values]")
            ax.scatter(missing_idx, x[missing_idx], s=10, label="missing positions in x_filled")

        status = "invalid" if lead_invalid else "valid"
        ax.set_title(
            f"Lead {c} | missing_ratio={np.mean(~m):.4f} "
            f"info_ratio={informative_ratio:.4f} ({status})"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.suptitle(title)
    plt.tight_layout()
    return fig

def main():
    # Passe diese Pfade an:
    train_dir = Path("../data_split/train")

    # Nimm irgendeine CSV aus dem Train-Ordner (hierarchisch oder flat)
    csv_files = sorted(train_dir.glob("*/*.csv"))
    if not csv_files:
        csv_files = sorted(train_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Keine CSVs in {train_dir}")

    csv_path = csv_files[0]
    print("Using:", csv_path)

    # Lade ein EKG (resample wie in deinem loader)
    y = load_ecg_csv(str(csv_path), target_length=10000)  # target_length ggf. anpassen

    # Erzeuge Dataset mit exakt deinen Parametern
    ds = ECGDataset(
        x_list=None,
        y_list=[y],
        window_size=1000,
        overlap=0.5,
        damage_rate=0.1,
        valid_abs_eps=1e-6,
        min_valid_ratio=0.05,
        damage_blocks_min=1,
        damage_blocks_max=3,
        damage_block_min=100,
        damage_block_max=500,
    )

    # Zieh ein Fenster
    sample = ds[0]
    y_target = sample["y_target"].numpy()
    x_filled = sample["x_filled"].numpy()
    mask = sample["mask"].numpy()

    print("Shapes:", y_target.shape, x_filled.shape, mask.shape)
    print("Missing ratio overall:", np.mean(mask == 0))
    for c in range(y_target.shape[0]):
        finite_ratio = np.isfinite(y_target[c]).mean()
        informative_ratio = (np.isfinite(y_target[c]) & (np.abs(y_target[c]) > 1e-6)).mean()
        var = np.nanvar(y_target[c])
        print(
            f"Lead {c}: finite_ratio={finite_ratio:.4f} "
            f"info_ratio={informative_ratio:.4f} var={var:.6e}"
        )

    fig = plot_one_window(
        y_target, x_filled, mask,
        leads_to_plot=None,
        title=f"Mask debug | {csv_path.name}",
        valid_abs_eps=1e-6,
        min_valid_ratio=0.05,
    )
    if fig is not None:
        out = "mask_debug.png"
        fig.savefig(out, dpi=150)
        print("Saved plot to:", out)

if __name__ == "__main__":
    main()
