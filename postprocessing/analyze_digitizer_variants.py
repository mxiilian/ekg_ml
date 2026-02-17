import argparse
import csv
import json
import re
import statistics
from pathlib import Path


VARIANT_RE = re.compile(r"-(\d{4})_timeseries_canonical")


def extract_variant(record_key: str) -> str | None:
    match = VARIANT_RE.search(record_key)
    if not match:
        return None
    return f"-{match.group(1)}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def analyze(input_path: Path) -> dict[str, dict[str, float]]:
    with input_path.open() as f:
        payload = json.load(f)

    per_record = payload.get("per_record", {})
    by_variant: dict[str, dict[str, list[float]]] = {}

    for record_key, metrics in per_record.items():
        variant = extract_variant(record_key)
        if not variant:
            continue
        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        if mae is None or rmse is None:
            continue
        by_variant.setdefault(variant, {"mae": [], "rmse": []})
        by_variant[variant]["mae"].append(float(mae))
        by_variant[variant]["rmse"].append(float(rmse))

    results: dict[str, dict[str, float]] = {}
    for variant, series in by_variant.items():
        mae_values = series["mae"]
        rmse_values = series["rmse"]
        results[variant] = {
            "count": len(mae_values),
            "mae_mean": mean(mae_values),
            "mae_median": statistics.median(mae_values),
            "rmse_mean": mean(rmse_values),
            "rmse_median": statistics.median(rmse_values),
        }

    return results


def write_csv(results: dict[str, dict[str, float]], output_path: Path) -> None:
    rows = []
    for variant, stats in results.items():
        rows.append(
            {
                "variant": variant,
                "count": stats["count"],
                "mae_mean": fmt(stats["mae_mean"]),
                "mae_median": fmt(stats["mae_median"]),
                "rmse_mean": fmt(stats["rmse_mean"]),
                "rmse_median": fmt(stats["rmse_median"]),
            }
        )

    def sort_key(item: dict[str, str | int]) -> tuple[int, str]:
        variant = str(item["variant"])
        if variant.startswith("-") and variant[1:].isdigit():
            return (int(variant[1:]), variant)
        return (10**9, variant)

    rows.sort(key=sort_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "count",
                "mae_mean",
                "mae_median",
                "rmse_mean",
                "rmse_median",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(
    results: dict[str, dict[str, float]],
    input_path: Path,
    csv_path: Path,
    output_path: Path,
) -> None:
    lines = [
        "% Digitizer scan variant analysis",
        f"% Input: {input_path}",
        f"% CSV: {csv_path}",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Variant & Count & MAE mean & MAE median & RMSE mean & RMSE median \\\\",
        "\\hline",
    ]

    def sort_key(item: tuple[str, dict[str, float]]) -> tuple[int, str]:
        variant = item[0]
        if variant.startswith("-") and variant[1:].isdigit():
            return (int(variant[1:]), variant)
        return (10**9, variant)

    for variant, stats in sorted(results.items(), key=sort_key):
        lines.append(
            "{variant} & {count} & {mae_mean} & {mae_median} & {rmse_mean} & {rmse_median} \\\\".format(
                variant=variant,
                count=stats["count"],
                mae_mean=fmt(stats["mae_mean"]),
                mae_median=fmt(stats["mae_median"]),
                rmse_mean=fmt(stats["rmse_mean"]),
                rmse_median=fmt(stats["rmse_median"]),
            )
        )

    lines.extend(["\\hline", "\\end{tabular}"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze digitizer scan variants")
    parser.add_argument(
        "--input",
        type=str,
        default="results/test_details_norm_ref_w2000_digitizer_all.json",
        help="Path to test_details_*.json with per_record entries",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/tables/digitizer_variant_analysis_norm_ref_w2000.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--tex",
        type=str,
        default="results/digitizer_variant_analysis_norm_ref_w2000.tex",
        help="Output LaTeX table path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    results = analyze(input_path)

    csv_path = Path(args.csv)
    tex_path = Path(args.tex)

    write_csv(results, csv_path)
    write_latex_table(results, input_path, csv_path, tex_path)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
