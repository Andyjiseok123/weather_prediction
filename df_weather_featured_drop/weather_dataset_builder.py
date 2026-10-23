
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. 경로 및 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BASE_DIR
    / "df_weather_강릉_강원특별자치도_preprocessed_featured.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "df_weather_강릉_30day_dataset.csv"
)

HORIZON_DAYS = 30

# 목표 날짜에 예측할 기상 변수
TARGET_COLUMNS = [
    "WS_AVG",
    "TA_AVG",
    "TA_MAX",
    "TA_MIN",
    "RN_DAY",
]


# =========================================================
# 2. 데이터 불러오기 및 검증
# =========================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"입력 파일이 없습니다: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "DATE",
    "STN",
    *TARGET_COLUMNS,
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_columns}"
    )

df["DATE"] = pd.to_datetime(
    df["DATE"],
    errors="raise",
)

df = df.sort_values(
    ["STN", "DATE"]
).reset_index(drop=True)

if df.duplicated(["STN", "DATE"]).any():
    raise ValueError(
        "동일 관측소에 중복 날짜가 있습니다."
    )

# 날짜 단위의 Lag를 사용하므로 관측소별 날짜 누락을 검사한다.
for stn, group in df.groupby("STN"):
    dates = group["DATE"].sort_values()

    expected_dates = pd.date_range(
        dates.min(),
        dates.max(),
        freq="D",
    )

    missing_dates = expected_dates.difference(dates)

    if len(missing_dates) > 0:
        raise ValueError(
            f"관측소 {stn}에 날짜 누락이 있습니다. "
            f"예시: {missing_dates[:5].tolist()}"
        )

print("=" * 60)
print("입력 데이터")
print("=" * 60)
print("행 수:", len(df))
print("컬럼 수:", len(df.columns))
print("관측소 수:", df["STN"].nunique())
print("시작일:", df["DATE"].min().date())
print("종료일:", df["DATE"].max().date())


# =========================================================
# 3. 기준일(ORIGIN_DATE)의 입력 Feature 선정
# =========================================================

# 기준일에 관측이 완료되어 있다고 가정하고,
# 해당 날짜까지 확인할 수 있는 원본 기상값을 사용한다.
ORIGIN_WEATHER_COLUMNS = [
    "WS_AVG",
    "TA_AVG",
    "TA_MAX",
    "TA_MIN",
    "RN_DAY",
]

# 기존 Feature Engineering에서 생성한 과거 통계 변수
HISTORICAL_FEATURES = [
    col
    for col in df.columns
    if (
        "_LAG_" in col
        or "_ROLL_MEAN_" in col
        or "_ROLL_STD_" in col
        or col.startswith("RN_DAY_ROLL_SUM_")
        or col.startswith("WET_DAYS_")
    )
]

# WET_DAY는 현재 행의 강수 여부이므로
# 별도의 모델 입력으로 사용하지 않는다.
HISTORICAL_FEATURES = [
    col
    for col in HISTORICAL_FEATURES
    if col != "WET_DAY"
]

# 기준일의 원본 관측값과 과거 Feature를 결합한다.
ORIGIN_FEATURE_COLUMNS = list(
    dict.fromkeys(
        ORIGIN_WEATHER_COLUMNS
        + HISTORICAL_FEATURES
    )
)

# 입력 Feature와 Label의 의미가 섞이지 않도록
# 기준일 원본 관측값에 _ORIGIN 접미사를 붙인다.
origin_table = df[
    ["STN", "DATE", *ORIGIN_FEATURE_COLUMNS]
].copy()

origin_table = origin_table.rename(
    columns={
        "DATE": "ORIGIN_DATE",
        **{
            col: f"{col}_ORIGIN"
            for col in ORIGIN_WEATHER_COLUMNS
        },
    }
)


# =========================================================
# 4. 목표 날짜의 정답 Label 테이블 생성
# =========================================================

label_table = df[
    ["STN", "DATE", *TARGET_COLUMNS]
].copy()

label_table = label_table.rename(
    columns={
        "DATE": "TARGET_DATE",
        "WS_AVG": "Y_WS_AVG",
        "TA_AVG": "Y_TA_AVG",
        "TA_MAX": "Y_TA_MAX",
        "TA_MIN": "Y_TA_MIN",
        "RN_DAY": "Y_RN_DAY",
    }
)

# 강수 관측값이 존재하는 경우에만 강수 여부를 정의한다.
label_table["Y_RAIN"] = np.where(
    label_table["Y_RN_DAY"].notna(),
    (label_table["Y_RN_DAY"] > 0).astype(float),
    np.nan,
)


# =========================================================
# 5. 기준일 + 30일 = 목표 날짜
# =========================================================

# 예측 기준일을 중심으로 학습 행을 생성한다.
dataset = origin_table.copy()

dataset["TARGET_DATE"] = (
    dataset["ORIGIN_DATE"]
    + pd.to_timedelta(HORIZON_DAYS, unit="D")
)

dataset["HORIZON_DAYS"] = HORIZON_DAYS

# 목표 날짜의 실제 관측값을 정답으로 연결한다.
dataset = dataset.merge(
    label_table,
    on=["STN", "TARGET_DATE"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

unmatched_count = int(
    dataset["_merge"].ne("both").sum()
)

# 미래 30일의 실제 관측값이 없는 행은 학습에 사용할 수 없다.
dataset = dataset.loc[
    dataset["_merge"].eq("both")
].drop(columns="_merge").copy()


# =========================================================
# 6. 목표 날짜의 달력 Feature 생성
# =========================================================

target_dates = dataset["TARGET_DATE"]

dataset["TARGET_YEAR"] = target_dates.dt.year
dataset["TARGET_MONTH"] = target_dates.dt.month
dataset["TARGET_DAY_OF_YEAR"] = target_dates.dt.dayofyear
dataset["TARGET_DAY_OF_WEEK"] = target_dates.dt.dayofweek

# 월의 계절 주기
dataset["TARGET_MONTH_SIN"] = np.sin(
    2 * np.pi
    * (dataset["TARGET_MONTH"] - 1)
    / 12
)

dataset["TARGET_MONTH_COS"] = np.cos(
    2 * np.pi
    * (dataset["TARGET_MONTH"] - 1)
    / 12
)

# 윤년을 반영한 연중 계절 주기
days_in_year = np.where(
    target_dates.dt.is_leap_year,
    366,
    365,
)

dataset["TARGET_DOY_SIN"] = np.sin(
    2 * np.pi
    * (dataset["TARGET_DAY_OF_YEAR"] - 1)
    / days_in_year
)

dataset["TARGET_DOY_COS"] = np.cos(
    2 * np.pi
    * (dataset["TARGET_DAY_OF_YEAR"] - 1)
    / days_in_year
)

dataset["TARGET_YEAR_TREND"] = (
    dataset["TARGET_YEAR"] - 2016
)


# =========================================================
# 7. Label 검증 및 정리
# =========================================================

LABEL_COLUMNS = [
    "Y_WS_AVG",
    "Y_TA_AVG",
    "Y_TA_MAX",
    "Y_TA_MIN",
    "Y_RN_DAY",
    "Y_RAIN",
]

missing_label_mask = dataset[
    LABEL_COLUMNS
].isna().any(axis=1)

missing_label_count = int(
    missing_label_mask.sum()
)

dataset = dataset.loc[
    ~missing_label_mask
].copy()

# 기준일과 목표 날짜의 간격이 정확히 30일인지 확인한다.
actual_horizon = (
    dataset["TARGET_DATE"] - dataset["ORIGIN_DATE"]
).dt.days

assert actual_horizon.eq(HORIZON_DAYS).all()

# 목표 날짜와 관측소 조합이 중복되지 않는지 확인한다.
assert not dataset.duplicated(
    ["STN", "ORIGIN_DATE", "TARGET_DATE"]
).any()

# 기온의 물리적 순서 검사
temperature_order_errors = (
    (dataset["Y_TA_MIN"] > dataset["Y_TA_AVG"])
    | (dataset["Y_TA_AVG"] > dataset["Y_TA_MAX"])
)

print("\n" + "=" * 60)
print("학습 데이터 생성 결과")
print("=" * 60)
print("예측 기간:", HORIZON_DAYS, "일")
print("기준일 범위:",
      dataset["ORIGIN_DATE"].min().date(),
      "~",
      dataset["ORIGIN_DATE"].max().date())
print("목표 날짜 범위:",
      dataset["TARGET_DATE"].min().date(),
      "~",
      dataset["TARGET_DATE"].max().date())
print("미래 관측값이 없어 제외한 행:", unmatched_count)
print("Label 결측으로 제외한 행:", missing_label_count)
print("최종 행 수:", len(dataset))
print("기온 순서 이상 행:",
      int(temperature_order_errors.sum()))


# =========================================================
# 8. 최종 컬럼 구성
# =========================================================

METADATA_COLUMNS = [
    "ORIGIN_DATE",
    "TARGET_DATE",
    "STN",
    "HORIZON_DAYS",
]

CALENDAR_FEATURES = [
    "TARGET_YEAR",
    "TARGET_MONTH",
    "TARGET_DAY_OF_YEAR",
    "TARGET_DAY_OF_WEEK",
    "TARGET_MONTH_SIN",
    "TARGET_MONTH_COS",
    "TARGET_DOY_SIN",
    "TARGET_DOY_COS",
    "TARGET_YEAR_TREND",
]

FEATURE_COLUMNS = (
    CALENDAR_FEATURES
    + [
        f"{col}_ORIGIN"
        for col in ORIGIN_WEATHER_COLUMNS
    ]
    + HISTORICAL_FEATURES
)

OUTPUT_COLUMNS = (
    METADATA_COLUMNS
    + FEATURE_COLUMNS
    + LABEL_COLUMNS
)

dataset = dataset[OUTPUT_COLUMNS].copy()

# 예측 기준일의 날짜와 목표 날짜는 메타데이터다.
# 다음 학습 단계에서는 FEATURE_COLUMNS만 X로 사용한다.
dataset.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d",
)

print("\n저장 완료:", OUTPUT_PATH)
print("최종 데이터 크기:", dataset.shape)
print("메타데이터 컬럼 수:", len(METADATA_COLUMNS))
print("입력 Feature 수:", len(FEATURE_COLUMNS))
print("Label 수:", len(LABEL_COLUMNS))

print("\n생성 결과 예시:")
print(
    dataset[
        [
            "ORIGIN_DATE",
            "TARGET_DATE",
            "STN",
            "TA_AVG_ORIGIN",
            "Y_TA_AVG",
            "Y_TA_MAX",
            "Y_TA_MIN",
            "Y_RN_DAY",
            "Y_RAIN",
        ]
    ].head(5).to_string(index=False)
)

print("\n입력 Feature 결측치 상위 10개:")
print(
    dataset[FEATURE_COLUMNS]
    .isna()
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
