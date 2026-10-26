from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "3.5_df_weather_wind_feature"
OUTPUT_DIR = BASE_DIR / "4_correlation_review"

TARGET_COLUMNS = [
    "WS_AVG",
    "TA_AVG",
    "TA_MAX",
    "TA_MIN",
    "RN_DAY",
]
METADATA_COLUMNS = ["DATE", "STN"]


def get_region_name(csv_path: Path) -> str:
    name = csv_path.stem
    if name.startswith("df_weather_"):
        name = name.removeprefix("df_weather_")
    if name.endswith("_preprocessed"):
        name = name.removesuffix("_preprocessed")
    return name


def format_rho(value: float) -> str:
    return f"{value:.4f}" if pd.notna(value) else "계산 불가"


def build_target_report(
    numeric_df: pd.DataFrame,
    target_name: str,
) -> str:
    target = numeric_df[target_name]
    correlations = []

    for feature_name in numeric_df.columns:
        if feature_name == target_name:
            continue

        feature = numeric_df[feature_name]
        valid_rows = target.notna() & feature.notna()
        sample_count = int(valid_rows.sum())

        if sample_count >= 2:
            target_ranks = target.loc[valid_rows].rank()
            feature_ranks = feature.loc[valid_rows].rank()
            rho = target_ranks.corr(feature_ranks, method="pearson")
        else:
            rho = float("nan")
        correlations.append(
            {
                "feature": feature_name,
                "rho": rho,
                "sample_count": sample_count,
            }
        )

    correlations.sort(
        key=lambda item: (
            pd.isna(item["rho"]),
            -abs(item["rho"]) if pd.notna(item["rho"]) else 0,
        )
    )

    valid_correlations = [
        item for item in correlations if pd.notna(item["rho"])
    ]
    if valid_correlations:
        strongest = valid_correlations[0]
        summary = (
            f"- 가장 강한 상관: `{strongest['feature']}` "
            f"(ρ = {format_rho(strongest['rho'])}, "
            f"n = {strongest['sample_count']})"
        )
    else:
        summary = "- 유효한 상관계수를 계산할 수 있는 변수가 없습니다."

    lines = [
        f"### {target_name}",
        "",
        summary,
        "",
        "| 변수 | Spearman ρ | 유효 표본 수 |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {item['feature']} | {format_rho(item['rho'])} "
        f"| {item['sample_count']} |"
        for item in correlations
    )
    if not correlations:
        lines.append("| 계산 가능한 다른 변수가 없습니다. | - | - |")
    lines.append("")

    return "\n".join(lines)


def analyze_file(csv_path: Path) -> Path:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    analysis_df = df.drop(columns=METADATA_COLUMNS, errors="ignore").copy()

    missing_targets = [
        target for target in TARGET_COLUMNS if target not in analysis_df.columns
    ]
    if missing_targets:
        raise ValueError(
            f"{csv_path.name}에 필수 타깃 컬럼이 없습니다: {missing_targets}"
        )

    for column in analysis_df.columns:
        analysis_df[column] = pd.to_numeric(
            analysis_df[column],
            errors="coerce",
        )

    target_sections = [
        build_target_report(analysis_df, target)
        for target in TARGET_COLUMNS
    ]

    report = [
        f"# Spearman 상관분석: {get_region_name(csv_path)}",
        "",
        f"- 원본 파일: `{csv_path.name}`",
        f"- 데이터 행 수: {len(df):,}",
        "- 상관계수: Spearman ρ",
        "- DATE, STN 및 현재 분석 타깃 자체는 상관분석에서 제외했습니다.",
        "- 각 변수 쌍에서 값이 모두 존재하는 행만 사용했습니다.",
        "- 상관계수의 절대값이 큰 순서로 정렬했습니다.",
        "",
        *target_sections,
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"REVIEW_{get_region_name(csv_path)}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {INPUT_DIR}")

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"입력 폴더에 CSV 파일이 없습니다: {INPUT_DIR}")

    for csv_path in csv_files:
        report_path = analyze_file(csv_path)
        print(f"보고서 저장 완료: {report_path}")

    print(
        f"\n총 {len(csv_files)}개 CSV의 분석을 완료했습니다. "
        f"저장 폴더: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
