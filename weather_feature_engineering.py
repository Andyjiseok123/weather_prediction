from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. 파일 경로
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "df_weather_preprocessed"
OUTPUT_DIR = BASE_DIR / "df_weather_featured"


# ==================================================
# 2. 기본 설정
# ==================================================

DATE_COLUMN = "DATE"
STATION_COLUMN = "STN"

TARGET_COLUMNS = [
    "WS_AVG",
    "TA_MAX",
    "TA_MIN",
    "TA_AVG",
    "RN_DAY",
]

WEATHER_COLUMNS = [
    "WS_AVG",
    "TA_MAX",
    "TA_MIN",
    "TA_AVG",
    "RN_DAY",
]

LAG_DAYS = [1, 3, 7, 14, 30, 365]
ROLLING_WINDOWS = [7, 14, 30]

EXCLUDED_COLUMNS = [
    "CA_TOT",
    "RN_60M_MAX",
    "FG_DUR",
    "SS_CMB",
]


def engineer_file(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    required_columns = [
        DATE_COLUMN,
        STATION_COLUMN,
        *TARGET_COLUMNS,
    ]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"{input_path.name}의 필수 컬럼이 누락되었습니다: {missing_columns}"
        )

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="raise")
    df = df.sort_values(
        [STATION_COLUMN, DATE_COLUMN]
    ).reset_index(drop=True)

    if df.duplicated([STATION_COLUMN, DATE_COLUMN]).any():
        raise ValueError(
            f"{input_path.name}에 동일 관측소의 중복 날짜가 있습니다. "
            "중복 데이터를 먼저 확인하세요."
        )

    print(f"\n처리 파일: {input_path.name}")
    print("원본 데이터 크기:", df.shape)

    # 날짜 기반 Feature
    df["YEAR"] = df[DATE_COLUMN].dt.year
    df["MONTH"] = df[DATE_COLUMN].dt.month
    df["DAY_OF_YEAR"] = df[DATE_COLUMN].dt.dayofyear
    df["DAY_OF_WEEK"] = df[DATE_COLUMN].dt.dayofweek
    df["MONTH_SIN"] = np.sin(
        2 * np.pi * (df["MONTH"] - 1) / 12
    )
    df["MONTH_COS"] = np.cos(
        2 * np.pi * (df["MONTH"] - 1) / 12
    )
    df["DOY_SIN"] = np.sin(
        2 * np.pi * (df["DAY_OF_YEAR"] - 1) / 365.25
    )
    df["DOY_COS"] = np.cos(
        2 * np.pi * (df["DAY_OF_YEAR"] - 1) / 365.25
    )
    df["YEAR_TREND"] = df["YEAR"] - df["YEAR"].min()

    # 관측소별 시차 Feature
    grouped = df.groupby(STATION_COLUMN, sort=False)
    for column in WEATHER_COLUMNS:
        for lag in LAG_DAYS:
            df[f"{column}_LAG_{lag}"] = grouped[column].shift(lag)

    # 과거 일교차 Feature
    df["TA_DIF_OBSERVED"] = df["TA_MAX"] - df["TA_MIN"]
    df["TA_DIF_LAG_1"] = (
        df.groupby(STATION_COLUMN)["TA_DIF_OBSERVED"].shift(1)
    )
    df["TA_DIF_LAG_7"] = (
        df.groupby(STATION_COLUMN)["TA_DIF_OBSERVED"].shift(7)
    )
    df = df.drop(columns=["TA_DIF_OBSERVED"])

    # 과거 데이터만 사용하는 이동 통계량 Feature
    for column in WEATHER_COLUMNS:
        past_values = df.groupby(STATION_COLUMN)[column].shift(1)
        grouped_past = past_values.groupby(
            df[STATION_COLUMN],
            sort=False,
        )

        for window in ROLLING_WINDOWS:
            df[f"{column}_ROLL_MEAN_{window}"] = grouped_past.transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )
            df[f"{column}_ROLL_STD_{window}"] = grouped_past.transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=window,
                ).std()
            )

    # 강수 관련 Feature
    rain = df["RN_DAY"]
    df["WET_DAY"] = np.where(
        rain.isna(),
        np.nan,
        (rain > 0).astype(float),
    )

    for window in ROLLING_WINDOWS:
        df[f"RN_DAY_ROLL_SUM_{window}"] = (
            df.groupby(STATION_COLUMN)["RN_DAY"].transform(
                lambda series: series.shift(1).rolling(
                    window=window,
                    min_periods=window,
                ).sum()
            )
        )
        df[f"WET_DAYS_{window}"] = (
            df.groupby(STATION_COLUMN)["WET_DAY"].transform(
                lambda series: series.shift(1).rolling(
                    window=window,
                    min_periods=window,
                ).sum()
            )
        )

    if not all(column in df.columns for column in TARGET_COLUMNS):
        raise AssertionError("예측 대상 컬럼이 결과에 없습니다.")

    remaining_excluded = [
        column for column in EXCLUDED_COLUMNS if column in df.columns
    ]
    if remaining_excluded:
        raise ValueError(
            f"{input_path.name}에 제외 대상 컬럼이 남아 있습니다: "
            f"{remaining_excluded}"
        )

    print("생성된 데이터 크기:", df.shape)
    print("생성된 Feature 개수:", len(df.columns) - len(TARGET_COLUMNS))
    feature_columns = [
        column
        for column in df.columns
        if "_LAG_" in column
        or "_ROLL_" in column
        or column.startswith("WET_DAYS_")
    ]
    print("\n시차·이동 Feature의 결측치:")
    print(
        df[feature_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )
    print("\n예측 대상 결측치:")
    print(df[TARGET_COLUMNS].isna().sum())

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    print("저장 경로:", output_path)


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"전처리 파일 폴더를 찾을 수 없습니다: {INPUT_DIR}\n"
            "먼저 weather_preprocessing.py를 실행하세요."
        )

    input_files = sorted(INPUT_DIR.glob("*.csv"))
    if not input_files:
        raise FileNotFoundError(
            f"전처리 파일 폴더에 CSV 파일이 없습니다: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in input_files:
        output_path = OUTPUT_DIR / f"{input_path.stem}_featured.csv"
        engineer_file(input_path, output_path)

    print(
        f"\n총 {len(input_files)}개 파일의 Feature Engineering을 완료했습니다. "
        f"저장 폴더: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
