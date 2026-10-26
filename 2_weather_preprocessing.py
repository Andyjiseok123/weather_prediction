# 1단계 데이터 전처리 ex) 결측치 처리, 특수값 처리, 컬럼 제외 등
from pathlib import Path
import pandas as pd


# ==================================================
# 1. 파일 경로 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "1_df_weather_bf_process"
OUTPUT_DIR = BASE_DIR / "2_df_weather_preprocessed"


# ==================================================
# 2. 전처리 정책
# ==================================================

MISSING_VALUE_MAP = {
    "TA_AVG": -99,
    "TA_MIN": -9,
}

RAIN_COLUMN = "RN_DAY"
RAIN_MISSING_CODE = -9

EXCLUDE_COLUMNS = [
    "CA_TOT",
    "RN_60M_MAX",
    "FG_DUR",
    "SS_CMB",
]

DATE_COLUMN = "# YYMMDD"
REQUIRED_COLUMNS = [
    DATE_COLUMN,
    "STN",
    "WS_AVG",
    "TA_MAX",
    "TA_MIN",
    "TA_AVG",
    RAIN_COLUMN,
]


def preprocess_file(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    print("=" * 60)
    print(f"[1] 원본 데이터: {input_path.name}")
    print("=" * 60)
    print(f"데이터 크기: {df.shape}")
    print(f"컬럼 수: {len(df.columns)}")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(
            f"{input_path.name}에 필수 컬럼이 없습니다: {missing_columns}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN].astype("string"),
        format="%Y%m%d",
        errors="raise",
    )
    df = df.rename(columns={DATE_COLUMN: "DATE"})
    df = df.sort_values("DATE").reset_index(drop=True)

    print("\n" + "=" * 60)
    print(f"[2] 특수값 처리: {input_path.name}")
    print("=" * 60)

    for column, sentinel in MISSING_VALUE_MAP.items():
        mask = df[column].eq(sentinel)
        count = int(mask.sum())
        df.loc[mask, column] = float("nan")
        print(f"{column}: {sentinel} -> NaN ({count}개)")

    rain_mask = df[RAIN_COLUMN].eq(RAIN_MISSING_CODE)
    rain_count = int(rain_mask.sum())
    df.loc[rain_mask, RAIN_COLUMN] = 0
    print(f"{RAIN_COLUMN}: {RAIN_MISSING_CODE} -> 0 ({rain_count}개)")

    existing_exclude_columns = [
        column for column in EXCLUDE_COLUMNS if column in df.columns
    ]
    missing_exclude_columns = [
        column for column in EXCLUDE_COLUMNS if column not in df.columns
    ]
    df = df.drop(columns=existing_exclude_columns)

    print("\n" + "=" * 60)
    print(f"[3] 컬럼 제외: {input_path.name}")
    print("=" * 60)
    print(f"제외된 컬럼: {existing_exclude_columns}")
    if missing_exclude_columns:
        print(
            "원본에 없어 제외 작업을 건너뛴 컬럼: "
            f"{missing_exclude_columns}"
        )

    numeric_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column != "STN"
    ]
    missing_before_fill = df[numeric_columns].isna().sum()
    df[numeric_columns] = (
        df[numeric_columns]
        .interpolate(method="linear", limit_area="inside")
        .bfill()
        .ffill()
    )
    filled_counts = missing_before_fill - df[numeric_columns].isna().sum()

    print("\n" + "=" * 60)
    print(f"[4] 수치형 결측값 보간: {input_path.name}")
    print("=" * 60)
    for column, count in filled_counts.items():
        if count > 0:
            print(f"{column}: {int(count)}개 채움")

    remaining_missing = df[numeric_columns].isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if not remaining_missing.empty:
        print(
            "앞뒤에 사용할 값이 없어 채우지 못한 수치형 결측값: "
            f"{remaining_missing.to_dict()}"
        )

    print("\n" + "=" * 60)
    print(f"[5] 전처리 결과 검증: {input_path.name}")
    print("=" * 60)
    print(f"최종 데이터 크기: {df.shape}")
    print(f"기간: {df['DATE'].min()} ~ {df['DATE'].max()}")

    duplicate_dates = int(df["DATE"].duplicated().sum())
    print(f"중복 날짜 수: {duplicate_dates}")

    for column, sentinel in MISSING_VALUE_MAP.items():
        remaining = int(df[column].eq(sentinel).sum())
        print(f"{column}의 {sentinel} 잔여 개수: {remaining}")

    remaining_rain_code = int(df[RAIN_COLUMN].eq(RAIN_MISSING_CODE).sum())
    print(f"RN_DAY의 -9 잔여 개수: {remaining_rain_code}")

    target_columns = [
        "WS_AVG",
        "TA_MAX",
        "TA_MIN",
        "TA_AVG",
        "RN_DAY",
    ]
    print("\n주요 변수 결측치:")
    print(df[target_columns].isna().sum())

    missing_values = df.isna().sum()
    print("\n결측치가 존재하는 전체 컬럼:")
    print(missing_values[missing_values > 0])

    expected_dates = pd.date_range(
        start=df["DATE"].min(),
        end=df["DATE"].max(),
        freq="D",
    )
    missing_dates = expected_dates.difference(df["DATE"])
    print(f"\n누락된 날짜 수: {len(missing_dates)}")

    if duplicate_dates > 0:
        print("주의: 중복 날짜가 있습니다.")
    if len(missing_dates) > 0:
        print("주의: 누락된 날짜가 있습니다.")

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print("\n" + "=" * 60)
    print(f"[6] 저장 완료: {input_path.name}")
    print("=" * 60)
    print(f"저장 경로: {output_path}")
    print(df.head())


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"입력 폴더를 찾을 수 없습니다: {INPUT_DIR}")

    input_files = sorted(INPUT_DIR.glob("*.csv"))
    if not input_files:
        raise FileNotFoundError(f"입력 폴더에 CSV 파일이 없습니다: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for input_path in input_files:
        output_path = OUTPUT_DIR / f"{input_path.stem}_preprocessed.csv"
        preprocess_file(input_path, output_path)

    print(
        f"\n총 {len(input_files)}개 파일의 전처리를 완료했습니다. "
        f"저장 폴더: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
