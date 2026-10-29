from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 기본 설정
# =========================================================

LOOKBACK = 14
RANDOM_SEED = 42

tf.keras.utils.set_random_seed(RANDOM_SEED)


# =========================================================
# CSV 및 날짜 처리
# =========================================================

DATE_COLUMN_CANDIDATES = [
    "DATE", "date", "Date",
    "DATETIME", "datetime", "timestamp",
    "일시", "날짜", "일자", "관측일",
]


def load_dataset(csv_path, start_date=None, date_column=None):
    """
    CSV를 로드한다.

    규칙:
    - 마지막 열: Y
    - 날짜 열: X에서 제외
    - 나머지 열: X
    - 날짜 열이 없다면 start_date부터 일 단위 날짜를 생성한다.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"데이터가 비어 있습니다: {csv_path}")

    if len(df.columns) < 2:
        raise ValueError(f"X와 Y를 구분할 열이 부족합니다: {csv_path}")

    # 명시한 날짜 열을 우선 사용하고, 없으면 이름으로 자동 탐지
    if date_column is not None:
        if date_column not in df.columns:
            raise ValueError(
                f"지정한 날짜 열 '{date_column}'이 없습니다: {csv_path}"
            )
        detected_date_column = date_column
    else:
        candidate_map = {
            str(col).strip().casefold(): col
            for col in df.columns
        }

        detected_date_column = None

        for candidate in DATE_COLUMN_CANDIDATES:
            key = candidate.strip().casefold()

            if key in candidate_map:
                detected_date_column = candidate_map[key]
                break

    # 마지막 CSV 열을 Y로 사용
    target_column = df.columns[-1]

    if target_column == detected_date_column:
        raise ValueError(
            f"마지막 열이 날짜 열입니다. Y 열을 확인하세요: {csv_path}"
        )

    # 실제 날짜 열 또는 설정한 시작 날짜로 날짜 배열 구성
    if detected_date_column is not None:
        dates = pd.to_datetime(
            df[detected_date_column],
            errors="coerce"
        )

        if dates.isna().any():
            raise ValueError(
                f"날짜 열을 날짜 형식으로 변환할 수 없습니다: {csv_path}"
            )

        dates = pd.DatetimeIndex(dates).normalize()

        # 날짜 순서대로 정렬
        order = np.argsort(dates.values)
        df = df.iloc[order].reset_index(drop=True)
        dates = dates[order]

    else:
        if start_date is None:
            raise ValueError(
                f"{csv_path}에 날짜 열이 없습니다.\n"
                "학습할 때 --start-date YYYY-MM-DD를 지정하세요."
            )

        dates = pd.date_range(
            start=pd.Timestamp(start_date).normalize(),
            periods=len(df),
            freq="D"
        )

    # 이 모델은 일 단위 시계열을 가정한다.
    # 날짜 누락이 있으면 행 간격이 실제 날짜 간격과 달라지므로 중단한다.
    if len(dates) > 1:
        date_diffs = np.diff(dates.values)

        if not np.all(date_diffs == np.timedelta64(1, "D")):
            raise ValueError(
                f"{csv_path}의 날짜가 연속된 일 단위 데이터가 아닙니다.\n"
                "누락 날짜 또는 중복 날짜를 먼저 확인하세요."
            )

    # 날짜 열과 Y를 제외한 나머지 열이 X
    x_columns = [
        col for col in df.columns
        if col not in {detected_date_column, target_column}
    ]

    if not x_columns:
        raise ValueError(f"X 특성이 없습니다: {csv_path}")

    # 숫자 문자열도 숫자로 변환
    X_df = df[x_columns].apply(pd.to_numeric, errors="coerce")
    y_series = pd.to_numeric(df[target_column], errors="coerce")

    # 결측치는 시계열 보간 후 앞/뒤 값으로 보완
    X_df = X_df.interpolate(
        method="linear",
        limit_direction="both"
    )
    y_series = y_series.interpolate(
        method="linear",
        limit_direction="both"
    )

    if X_df.isna().any().any():
        bad_cols = X_df.columns[X_df.isna().any()].tolist()
        raise ValueError(
            f"숫자로 변환할 수 없거나 값이 없는 X 열이 있습니다: {bad_cols}"
        )

    if y_series.isna().any():
        raise ValueError(
            f"Y 열 '{target_column}'에 처리되지 않은 결측값이 있습니다."
        )

    X = X_df.to_numpy(dtype=np.float32)
    y = y_series.to_numpy(dtype=np.float32).reshape(-1, 1)

    return {
        "X": X,
        "y": y,
        "dates": pd.DatetimeIndex(dates),
        "x_columns": list(x_columns),
        "target_column": target_column,
        "date_column": detected_date_column,
    }


# =========================================================
# 날짜 기반 특성 재계산
# =========================================================

def update_calendar_features(x_row, x_columns, date):
    """
    X에 다음 특성이 존재하면 미래 날짜에 맞춰 다시 계산한다.
    - MONTH_SIN
    - MONTH_COS
    - DOY_SIN
    - DOY_COS

    해당 열이 없으면 아무것도 추가하지 않는다.
    """

    row = np.asarray(x_row, dtype=np.float32).copy()
    date = pd.Timestamp(date)

    month_angle = 2.0 * np.pi * (date.month - 1) / 12.0

    day_of_year = date.dayofyear
    year_length = 366.0 if date.is_leap_year else 365.0
    doy_angle = 2.0 * np.pi * (day_of_year - 1) / year_length

    calendar_values = {
        "MONTH_SIN": np.sin(month_angle),
        "MONTH_COS": np.cos(month_angle),
        "DOY_SIN": np.sin(doy_angle),
        "DOY_COS": np.cos(doy_angle),
    }

    column_lookup = {
        str(col).strip().upper(): idx
        for idx, col in enumerate(x_columns)
    }

    for name, value in calendar_values.items():
        if name in column_lookup:
            row[column_lookup[name]] = value

    return row

# =========================================================
# LightGBM feature 생성
# =========================================================

def create_lightgbm_features(X_raw, x_columns):
    """
    원본 X와 파생 Feature를 결합해 LightGBM 입력을 생성한다.

    Feature:
    - TA_RANGE = TA_MAX - TA_MIN
    - 모든 X의 1일/365일 LAG
    - 모든 X의 7일 Rolling Mean/Std
    """

    X_raw = np.asarray(X_raw, dtype=np.float32)

    if X_raw.ndim != 2 or X_raw.shape[1] != len(x_columns):
        raise ValueError(
            "X_raw의 shape과 x_columns 길이가 일치하지 않습니다."
        )

    base_df = pd.DataFrame(X_raw, columns=list(x_columns))
    features = base_df.copy()

    # 1. 일교차
    column_lookup = {
        str(col).strip().upper(): col
        for col in base_df.columns
    }

    ta_max_col = column_lookup.get("TA_MAX")
    ta_min_col = column_lookup.get("TA_MIN")

    if ta_max_col is not None and ta_min_col is not None:
        range_name = "TA_RANGE"

        if range_name in features.columns:
            range_name = "TA_MAX_MINUS_TA_MIN"

        features[range_name] = (
            base_df[ta_max_col] - base_df[ta_min_col]
        )

    # 2. LAG 및 3. Rolling
    for col in x_columns:
        features[f"{col}_LAG_1"] = base_df[col].shift(1)
        features[f"{col}_LAG_365"] = base_df[col].shift(365)

        features[f"{col}_ROLL_MEAN_7"] = (
            base_df[col].rolling(
                window=7,
                min_periods=7
            ).mean()
        )

        features[f"{col}_ROLL_STD_7"] = (
            base_df[col].rolling(
                window=7,
                min_periods=7
            ).std()
        )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return features


# =========================================================
# 시계열 데이터 생성
# =========================================================

def create_x_sequences(X_scaled, lookback):
    """
    Model A:
    과거 lookback일의 X -> 다음 날의 X
    """

    X_seq = []
    y_seq = []

    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(X_scaled[i])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
    )


def create_y_sequences(X_scaled, y_scaled, lookback):
    """
    Model B:
    목표 날짜를 포함한 최근 lookback일의 X -> 해당 날짜의 Y
    """

    X_seq = []
    y_seq = []

    for i in range(lookback - 1, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback + 1:i + 1])
        y_seq.append(y_scaled[i])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
    )


# =========================================================
# BiLSTM 모델
# Functional API 사용
# =========================================================

def build_x_model(lookback, n_features):
    """
    Model A: 미래 X 예측 모델
    """

    inputs = Input(shape=(lookback, n_features), name="x_sequence")

    x = Bidirectional(
        LSTM(64, return_sequences=True)
    )(inputs)

    x = Dropout(0.2)(x)

    x = Bidirectional(
        LSTM(32)
    )(x)

    x = Dense(64, activation="relu")(x)

    outputs = Dense(
        n_features,
        activation="linear",
        name="next_x"
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="X_BiLSTM")

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mse"]
    )

    return model


def build_y_model(lookback, n_features):
    """
    Model B: 목표 날짜의 Y 예측 모델
    """

    inputs = Input(shape=(lookback, n_features), name="x_sequence")

    x = Bidirectional(
        LSTM(64, return_sequences=True)
    )(inputs)

    x = Dropout(0.2)(x)

    x = Bidirectional(
        LSTM(32)
    )(x)

    x = Dense(32, activation="relu")(x)

    outputs = Dense(
        1,
        activation="linear",
        name="target_y"
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="Y_BiLSTM")

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mse"]
    )

    return model


# =========================================================
# 모델 학습
# =========================================================

def build_dense_ensemble_model(n_inputs=2):
    """Bi-LSTM과 LightGBM의 예측값 2개를 결합하는 Dense stacking 모델."""
    inputs = Input(shape=(n_inputs,), name="base_model_predictions")
    x = Dense(16, activation="relu", name="ensemble_dense_1")(inputs)
    x = Dense(8, activation="relu", name="ensemble_dense_2")(x)
    outputs = Dense(1, activation="linear", name="ensemble_y")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Dense_Stacking_Ensemble")
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


def fit_model(
    model, X_train, y_train, epochs=50, batch_size=32,
    validation_data=None, return_history=False, early_stopping=True
):
    """모델을 학습하고 필요하면 epoch별 loss 이력을 반환한다."""
    callbacks = []
    validation_split = 0.1 if validation_data is None and len(X_train) >= 20 else 0.0

    if early_stopping and (validation_data is not None or validation_split > 0):
        callbacks.append(EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ))

    fit_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": callbacks,
        "shuffle": False,
        "verbose": 0,
    }
    if validation_data is not None:
        fit_kwargs["validation_data"] = validation_data
    else:
        fit_kwargs["validation_split"] = validation_split

    history = model.fit(X_train, y_train, **fit_kwargs)
    if return_history:
        return model, history.history
    return model


# =========================================================
# 미래 X 재귀 예측
# =========================================================

def recursive_forecast_x(
    model_x,
    x_scaler,
    x_columns,
    history_X_raw,
    future_dates,
    lookback
):
    """
    미래 X를 하루씩 반복해서 예측한다.

    반환값:
        미래 날짜에 대한 예측 X (원래 단위)
    """

    history_X_raw = np.asarray(history_X_raw, dtype=np.float32)
    future_dates = pd.DatetimeIndex(future_dates)

    if len(history_X_raw) < lookback:
        raise ValueError(
            f"X 이력이 최소 {lookback}행 필요합니다."
        )

    scaled_history = x_scaler.transform(history_X_raw)
    predicted_rows = []

    for future_date in future_dates:
        window = scaled_history[-lookback:]
        window = window.reshape(1, lookback, len(x_columns))

        # 다음 날짜의 X를 정규화된 공간에서 예측
        pred_scaled = model_x.predict(window, verbose=0)

        # 원래 단위로 복원
        pred_raw = x_scaler.inverse_transform(pred_scaled)[0]

        # 날짜 기반 특성은 모델 예측값 대신 정확한 날짜 값으로 교체
        pred_raw = update_calendar_features(
            pred_raw,
            x_columns,
            future_date
        )

        # 재귀 예측에 사용할 정규화 값 재계산
        pred_scaled_corrected = x_scaler.transform(
            pred_raw.reshape(1, -1)
        )[0]

        predicted_rows.append(pred_raw.copy())

        # 다음 날 예측 입력에 현재 예측값을 추가
        scaled_history = np.vstack([
            scaled_history,
            pred_scaled_corrected
        ])

    if not predicted_rows:
        return np.empty((0, len(x_columns)), dtype=np.float32)

    return np.asarray(predicted_rows, dtype=np.float32)