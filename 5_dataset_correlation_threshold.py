from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "3.5_df_weather_wind_feature"
REVIEW_DIR = BASE_DIR / "4_correlation_review"
OUTPUT_DIR = BASE_DIR / "5_dataset_correlation_threshold"

TARGET_COLUMNS = [
    "WS_AVG",
    "TA_AVG",
    "TA_MAX",
    "TA_MIN",
    "RN_DAY",
]
METADATA_COLUMNS = ["DATE", "STN"]
CORRELATION_THRESHOLD = 0.1
UNAVAILABLE_RHO_VALUES = {"계산 불가", "N/A", "NaN", ""}


def get_region_name(csv_path: Path) -> str:
    name = csv_path.stem
    if name.startswith("df_weather_"):
        name = name.removeprefix("df_weather_")
    if name.endswith("_preprocessed"):
        name = name.removesuffix("_preprocessed")
    return name


def parse_review(review_path: Path) -> dict[str, dict[str, float | None]]:
    sections: dict[str, dict[str, float | None]] = {}
    current_target: str | None = None

    for line_number, line in enumerate(
        review_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        heading = re.fullmatch(r"###\s+(\S+)\s*", line)
        if heading:
            current_target = heading.group(1)
            if current_target in sections:
                raise ValueError(
                    f"{review_path.name}:{line_number}: "
                    f"중복 타깃 섹션 {current_target}"
                )
            sections[current_target] = {}
            continue

        if current_target is None or not line.strip().startswith("|"):
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] == "변수" or set(cells[0]) <= {"-", ":"}:
            continue

        feature_name, rho_text = cells[0], cells[1]
        if rho_text in UNAVAILABLE_RHO_VALUES:
            rho = None
        else:
            try:
                rho = float(rho_text)
            except ValueError as exc:
                raise ValueError(
                    f"{review_path.name}:{line_number}: "
                    f"상관계수를 숫자로 읽을 수 없습니다: {rho_text!r}"
                ) from exc

        if feature_name in sections[current_target]:
            raise ValueError(
                f"{review_path.name}:{line_number}: "
                f"{current_target}에 중복 변수 {feature_name}"
            )
        sections[current_target][feature_name] = rho

    missing_targets = [
        target for target in TARGET_COLUMNS if target not in sections
    ]
    if missing_targets:
        raise ValueError(
            f"{review_path.name}에 타깃 섹션이 없습니다: {missing_targets}"
        )

    for target in TARGET_COLUMNS:
        if not sections[target]:
            raise ValueError(
                f"{review_path.name}의 {target} 섹션에 상관계수 표가 없습니다."
            )

    return sections


def create_target_dataset(
    df: pd.DataFrame,
    target: str,
    correlations: dict[str, float | None],
    csv_path: Path,
) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"{csv_path.name}에 타깃 컬럼이 없습니다: {target}")

    expected_features = {
        column
        for column in df.columns
        if column not in METADATA_COLUMNS and column != target
    }
    unlisted_features = expected_features - correlations.keys()
    if unlisted_features:
        raise ValueError(
            f"{csv_path.name}의 {target}에 대한 상관분석 결과에 "
            f"컬럼이 없습니다: {sorted(unlisted_features)}"
        )

    kept_columns = [
        column
        for column in df.columns
        if (
            column in METADATA_COLUMNS
            or column == target
            or (
                correlations[column] is not None
                and abs(correlations[column]) >= CORRELATION_THRESHOLD
            )
        )
    ]
    kept_columns = [
        column for column in kept_columns if column != target
    ] + [target]
    return df[kept_columns].copy()


def process_file(csv_path: Path) -> int:
    region_name = get_region_name(csv_path)
    review_path = REVIEW_DIR / f"REVIEW_{region_name}.md"
    if not review_path.is_file():
        raise FileNotFoundError(
            f"{csv_path.name}에 대응하는 상관분석 파일이 없습니다: "
            f"{review_path}"
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    correlations_by_target = parse_review(review_path)
    region_output_dir = OUTPUT_DIR / region_name
    region_output_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGET_COLUMNS:
        target_df = create_target_dataset(
            df,
            target,
            correlations_by_target[target],
            csv_path,
        )
        output_path = region_output_dir / f"{target}.csv"
        target_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return len(TARGET_COLUMNS)


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {INPUT_DIR}")
    if not REVIEW_DIR.is_dir():
        raise FileNotFoundError(
            f"상관분석 Markdown 폴더를 찾을 수 없습니다: {REVIEW_DIR}"
        )

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"입력 폴더에 CSV 파일이 없습니다: {INPUT_DIR}")

    dataset_count = 0
    for csv_path in csv_files:
        dataset_count += process_file(csv_path)

    print(
        f"\n총 {len(csv_files)}개 지역, {dataset_count}개 타깃 데이터셋을 "
        f"저장했습니다: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
