from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "3.5_df_weather_wind_feature" / "df_weather_강릉_강원특별자치도_preprocessed.csv"
OUT_DIR = BASE_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)


def spearman_rho(series_x: pd.Series, series_y: pd.Series) -> float:
    valid = series_x.notna() & series_y.notna()
    if valid.sum() < 2:
        return float("nan")
    x_rank = series_x.loc[valid].rank()
    y_rank = series_y.loc[valid].rank()
    return float(x_rank.corr(y_rank, method="pearson"))


def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    target = "TA_AVG"

    numeric_df = df.drop(columns=["DATE", "STN"], errors="ignore").copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    candidate_records = []
    for col in numeric_df.columns:
        if col == target:
            continue
        rho = spearman_rho(numeric_df[col], numeric_df[target])
        if pd.notna(rho):
            candidate_records.append((col, rho))

    top_two = sorted(candidate_records, key=lambda item: abs(item[1]), reverse=True)[:2]
    lowest_two = sorted(candidate_records, key=lambda item: abs(item[1]))[:2]
    selected = []
    seen = set()
    for col, rho in top_two + lowest_two:
        if col not in seen:
            selected.append((col, rho))
            seen.add(col)
    if len(selected) < 4:
        raise RuntimeError(f"선택할 변수 수가 부족합니다. selected={selected}")

    print("Selected variables:")
    for col, rho in selected:
        print(f"{col}: rho={rho:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (col, rho) in zip(axes, selected):
        x = numeric_df[col]
        y = numeric_df[target]
        valid = x.notna() & y.notna()
        ax.scatter(x[valid], y[valid], alpha=0.6, s=12)
        ax.set_title(f"{col} vs {target}\nρ={rho:.4f}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = OUT_DIR / "TA_AVG_selected_scatter.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
