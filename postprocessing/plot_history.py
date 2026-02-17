import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def plot_history(history_path: Path, out_path: Path, loss_scale: str):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")

    with history_path.open() as f:
        payload = json.load(f)

    history = payload.get("history", payload)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_mae = history.get("val_mae", [])
    val_rmse = history.get("val_rmse", [])
    lr = history.get("learning_rate", [])

    epochs = list(range(1, max(len(train_loss), len(val_loss), len(val_mae), len(val_rmse), len(lr)) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)

    axes[0].plot(epochs[:len(train_loss)], train_loss, label="train_loss")
    axes[0].plot(epochs[:len(val_loss)], val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    if loss_scale == "log":
        axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(epochs[:len(val_mae)], val_mae, label="val_mae")
    axes[1].set_title("MAE (missing)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].plot(epochs[:len(val_rmse)], val_rmse, label="val_rmse")
    axes[2].set_title("RMSE (missing)")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    axes[3].plot(epochs[:len(lr)], lr, label="learning_rate")
    axes[3].set_title("Learning Rate")
    axes[3].set_xlabel("epoch")
    axes[3].legend()
    axes[3].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory containing training_history.json",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="checkpoints/training_history.json",
        help="Path to training_history.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output plot path (defaults to run dir when --run-dir is used)",
    )
    parser.add_argument(
        "--loss-scale",
        type=str,
        default="log",
        choices=("log", "linear"),
        help="Scale for loss axis (log or linear)",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        history_path = run_dir / "training_history.json"
        if args.out:
            out_path = Path(args.out)
        else:
            out_path = run_dir / "training_history.png"
    else:
        history_path = Path(args.history)
        if args.out:
            out_path = Path(args.out)
        else:
            out_path = Path("checkpoints/training_history.png")

    plot_history(history_path, out_path, args.loss_scale)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
