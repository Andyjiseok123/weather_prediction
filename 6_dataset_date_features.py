from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "5_dataset_correlation_threshold"
OUTPUT_DIR = BASE_DIR / "6_dataset"

DATE_COLUMN = "DATE"
STATION_COLUMN = "STN"
DATE_FEATURE_COLUMNS = [
    "MONTH_SIN",
    "MONTH_COS",
    "DOY_SIN",
    "DOY_COS",
]


def add_date_features(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    required_columns = [DATE_COLUMN, STATION_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{csv_path.name}에 필수 컬럼이 없습니다: {missing_columns}"
        )

    date_values = pd.to_datetime(df[DATE_COLUMN], errors="raise")
    month = date_values.dt.month
    day_of_year = date_values.dt.day_of_year

    result = df.drop(columns=[DATE_COLUMN, STATION_COLUMN]).copy()
    for column in DATE_FEATURE_COLUMNS:
        if column in result.columns:
            raise ValueError(
                f"{csv_path.name}에 생성할 날짜 파생변수 컬럼이 "
                f"이미 있습니다: {column}"
            )

    date_features = pd.DataFrame(
        {
            "MONTH_SIN": np.sin(2 * np.pi * (month - 1) / 12),
            "MONTH_COS": np.cos(2 * np.pi * (month - 1) / 12),
            "DOY_SIN": np.sin(2 * np.pi * (day_of_year - 1) / 365.25),
            "DOY_COS": np.cos(2 * np.pi * (day_of_year - 1) / 365.25),
        },
        index=df.index,
    )

    return pd.concat([date_features, result], axis=1)


def process_file(csv_path: Path) -> Path:
    relative_path = csv_path.relative_to(INPUT_DIR)
    output_path = OUTPUT_DIR / relative_path.parent / f"LSTM_{csv_path.name}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    transformed_df = add_date_features(df, csv_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return output_path


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {INPUT_DIR}")

    csv_files = sorted(INPUT_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"입력 폴더에 CSV 파일이 없습니다: {INPUT_DIR}")

    for csv_path in csv_files:
        output_path = process_file(csv_path)
        print(f"저장 완료: {output_path}")

    print(
        f"\n총 {len(csv_files)}개 CSV를 변환했습니다. "
        f"저장 폴더: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
