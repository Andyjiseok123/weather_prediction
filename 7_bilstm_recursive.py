
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 1. 설정
# =========================================================

DATA_START_DATE = "2016-01-01"

LOOKBACK = 14   #time steps for the LSTM input sequence
EPOCHS = 50
BATCH_SIZE = 32
TRAIN_RATIO = 0.8


# =========================================================
# 2. CSV 로드 및 전처리
# =========================================================

def load_data(csv_path, data_start_date):

    df = pd.read_csv(csv_path)

    if df.shape[1] < 2:
        raise ValueError("CSV에는 X와 Y 열이 각각 하나 이상 필요합니다.")

    # 마지막 열이 Y
    target_col = df.columns[-1]

    # 나머지 열이 X
    feature_cols = list(df.columns[:-1])

    # 날짜가 CSV에 별도 열로 없다는 전제
    dates = pd.date_range(
        start=data_start_date,
        periods=len(df),
        freq="D"
    )

    # 숫자형으로 변환
    df = df.apply(pd.to_numeric, errors="coerce")

    # 무한대 -> 결측치
    df = df.replace([np.inf, -np.inf], np.nan)

    # 시계열 순서에 따라 결측치 보간
    df = df.interpolate(
        method="linear",
        limit_direction="both"
    )

    df = df.ffill().bfill()

    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"숫자로 변환할 수 없거나 결측치가 남은 열: {bad_cols}"
        )

    # X, Y 분리
    X = df[feature_cols].to_numpy(dtype=np.float32)
    Y = df[[target_col]].to_numpy(dtype=np.float32)

    return df, dates, X, Y, feature_cols, target_col


# =========================================================
# 3. 함수형 BiLSTM 모델 A: 미래 X 예측
# =========================================================

def build_x_model(lookback, n_features):

    inputs = Input(
        shape=(lookback, n_features),
        name="past_X"
    )

    x = Bidirectional(
        LSTM(64, return_sequences=True)
    )(inputs)

    x = Dropout(0.2)(x)

    x = Bidirectional(
        LSTM(32)
    )(x)

    x = Dense(32, activation="relu")(x)

    # X 열 개수에 따라 출력 차원 자동 결정
    outputs = Dense(
        n_features,
        name="next_day_X"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="BiLSTM_X_Forecaster"
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# =========================================================
# 4. 함수형 BiLSTM 모델 B: WS_AVG 예측
# =========================================================

def build_y_model(lookback, n_features):

    inputs = Input(
        shape=(lookback, n_features),
        name="past_X_for_Y"
    )

    x = Bidirectional(
        LSTM(64, return_sequences=True)
    )(inputs)

    x = Dropout(0.2)(x)

    x = Bidirectional(
        LSTM(32)
    )(x)

    x = Dense(32, activation="relu")(x)

    # Y는 WS_AVG 하나
    outputs = Dense(1, name="WS_AVG")(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="BiLSTM_Y_Forecaster"
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# =========================================================
# 5. 시퀀스 생성
# =========================================================

def create_x_sequences(X, lookback):

    sequences = []
    targets = []
    positions = []

    for i in range(lookback, len(X)):

        sequences.append(X[i - lookback:i])
        targets.append(X[i])
        positions.append(i)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(positions)
    )


def create_y_sequences(X, Y, lookback):

    sequences = []
    targets = []
    positions = []

    # X 시퀀스의 마지막 날짜와 Y의 날짜를 일치시킴
    for i in range(lookback - 1, len(X)):

        sequences.append(X[i - lookback + 1:i + 1])
        targets.append(Y[i])
        positions.append(i)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(positions)
    )


# =========================================================
# 6. 검증으로 적절한 학습 epoch 결정
# =========================================================

def find_best_epoch(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size
):

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )

    best_epoch = int(
        np.argmin(history.history["val_loss"]) + 1
    )

    print("선택된 epoch:", best_epoch)

    return best_epoch


# =========================================================
# 7. 미래 날짜의 달력 특성 갱신
# =========================================================

def update_calendar_features(
    raw_row,
    feature_cols,
    forecast_date
):
    """
    CSV에 해당 열이 존재할 때만 갱신한다.
    나머지 X는 모델 A의 예측값을 사용한다.
    """

    row = raw_row.copy()

    month = forecast_date.month
    day_of_year = forecast_date.dayofyear

    calendar_values = {
        "MONTH_SIN": np.sin(
            2 * np.pi * (month - 1) / 12
        ),
        "MONTH_COS": np.cos(
            2 * np.pi * (month - 1) / 12
        ),
        "DOY_SIN": np.sin(
            2 * np.pi * (day_of_year - 1) / 365
        ),
        "DOY_COS": np.cos(
            2 * np.pi * (day_of_year - 1) / 365
        )
    }

    for col, value in calendar_values.items():

        if col in feature_cols:
            idx = feature_cols.index(col)
            row[idx] = value

    return row


# =========================================================
# 8. 전체 학습 및 목표 날짜 예측
# =========================================================

def forecast_ws_avg(
    csv_path,
    target_date,
    data_start_date=DATA_START_DATE,
    lookback=LOOKBACK,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    train_ratio=TRAIN_RATIO
):

    tf.keras.utils.set_random_seed(42)

    # -----------------------------
    # 데이터 로드
    # -----------------------------

    df, dates, X_raw, Y_raw, feature_cols, target_col = (
        load_data(csv_path, data_start_date)
    )

    n_rows, n_features = X_raw.shape

    last_date = dates[-1]
    target_date = pd.Timestamp(target_date)

    if target_date <= last_date:
        raise ValueError(
            f"목표 날짜는 마지막 관측일({last_date.date()})보다 "
            "뒤여야 합니다."
        )

    if n_rows <= lookback + 10:
        raise ValueError("학습 데이터가 부족합니다.")

    split_row = int(n_rows * train_ratio)

    if split_row <= lookback:
        raise ValueError("학습 구간이 너무 짧습니다.")

    print("\n========== 데이터 정보 ==========")
    print("CSV shape:", df.shape)
    print("X 변수 개수:", n_features)
    print("Y 변수:", target_col)
    print("최초 관측일:", dates[0].date())
    print("마지막 관측일:", last_date.date())
    print("목표 날짜:", target_date.date())

    # -----------------------------
    # 검증용 스케일러
    # 학습 구간만으로 적합
    # -----------------------------

    x_scaler_val = MinMaxScaler()
    y_scaler_val = MinMaxScaler()

    x_scaler_val.fit(X_raw[:split_row])
    y_scaler_val.fit(Y_raw[:split_row])

    X_val_scaled = x_scaler_val.transform(X_raw)
    Y_val_scaled = y_scaler_val.transform(Y_raw)

    # -----------------------------
    # 모델 A의 학습 데이터
    # 과거 X -> 다음 날 X
    # -----------------------------

    XA, yA, posA = create_x_sequences(
        X_val_scaled,
        lookback
    )

    train_A = posA < split_row
    valid_A = posA >= split_row

    # -----------------------------
    # 모델 B의 학습 데이터
    # 최근 14일 X -> 해당 날짜 Y
    # -----------------------------

    XB, yB, posB = create_y_sequences(
        X_val_scaled,
        Y_val_scaled,
        lookback
    )

    train_B = posB < split_row
    valid_B = posB >= split_row

    # -----------------------------
    # 모델 A: 검증 및 epoch 선택
    # -----------------------------

    model_A_temp = build_x_model(
        lookback,
        n_features
    )

    print("\n========== 모델 A 검증 ==========")

    best_epoch_A = find_best_epoch(
        model_A_temp,
        XA[train_A],
        yA[train_A],
        XA[valid_A],
        yA[valid_A],
        epochs,
        batch_size
    )

    del model_A_temp

    # -----------------------------
    # 모델 B: 검증 및 epoch 선택
    # -----------------------------

    model_B_temp = build_y_model(
        lookback,
        n_features
    )

    print("\n========== 모델 B 검증 ==========")

    best_epoch_B = find_best_epoch(
        model_B_temp,
        XB[train_B],
        yB[train_B],
        XB[valid_B],
        yB[valid_B],
        epochs,
        batch_size
    )

    del model_B_temp

    # =====================================================
    # 최종 모델은 전체 과거 데이터로 다시 학습
    # =====================================================

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X_raw)
    Y_scaled = y_scaler.fit_transform(Y_raw)

    # -----------------------------
    # 최종 모델 A
    # -----------------------------

    XA_all, yA_all, _ = create_x_sequences(
        X_scaled,
        lookback
    )

    model_A = build_x_model(
        lookback,
        n_features
    )

    print("\n========== 모델 A 최종 학습 ==========")

    model_A.fit(
        XA_all,
        yA_all,
        epochs=best_epoch_A,
        batch_size=batch_size,
        shuffle=False,
        verbose=1
    )

    # -----------------------------
    # 최종 모델 B
    # -----------------------------

    XB_all, yB_all, _ = create_y_sequences(
        X_scaled,
        Y_scaled,
        lookback
    )

    model_B = build_y_model(
        lookback,
        n_features
    )

    print("\n========== 모델 B 최종 학습 ==========")

    model_B.fit(
        XB_all,
        yB_all,
        epochs=best_epoch_B,
        batch_size=batch_size,
        shuffle=False,
        verbose=1
    )

    # =====================================================
    # 미래 X의 재귀적 예측
    # =====================================================

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        end=target_date,
        freq="D"
    )

    # 마지막 14일 실제 X
    window = X_scaled[-lookback:, :].copy()

    future_X_scaled = []
    future_X_raw = []

    print("\n========== 미래 X 재귀 예측 ==========")

    for forecast_date in future_dates:

        # 다음 날 X 예측
        predicted_X_scaled = model_A.predict(
            window[np.newaxis, :, :],
            verbose=0
        )[0]

        # 역변환하여 원래 단위의 X로 복원
        predicted_X_raw = x_scaler.inverse_transform(
            predicted_X_scaled.reshape(1, -1)
        )[0]

        # 날짜 기반 변수는 실제 미래 날짜로 계산
        predicted_X_raw = update_calendar_features(
            predicted_X_raw,
            feature_cols,
            forecast_date
        )

        # 날짜 기반 변수를 포함하여 다시 스케일링
        predicted_X_scaled = x_scaler.transform(
            predicted_X_raw.reshape(1, -1)
        )[0]

        future_X_raw.append(predicted_X_raw.copy())
        future_X_scaled.append(predicted_X_scaled.copy())

        # 예측값을 다음 입력 시퀀스에 추가
        window = np.concatenate(
            [
                window[1:, :],
                predicted_X_scaled[np.newaxis, :]
            ],
            axis=0
        )

    future_X_raw = np.asarray(future_X_raw)
    future_X_scaled = np.asarray(future_X_scaled)

    # -----------------------------
    # 미래 X 저장
    # -----------------------------

    future_x_df = pd.DataFrame(
        future_X_raw,
        columns=feature_cols
    )

    future_x_df.insert(
        0,
        "DATE",
        future_dates
    )

    x_output_path = (
        csv_path.rsplit(".", 1)[0]
        + "_future_X.csv"
    )

    future_x_df.to_csv(
        x_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 모델 B로 목표 날짜의 WS_AVG 예측
    # =====================================================

    # window에는 목표 날짜까지의 X가 들어 있음
    target_X_sequence = window[np.newaxis, :, :]

    predicted_Y_scaled = model_B.predict(
        target_X_sequence,
        verbose=0
    )

    predicted_Y = y_scaler.inverse_transform(
        predicted_Y_scaled
    )[0, 0]

    # -----------------------------
    # 최종 결과 저장
    # -----------------------------

    result = pd.DataFrame({
        "target_date": [target_date.strftime("%Y-%m-%d")],
        target_col: [predicted_Y]
    })

    output_path = (
        csv_path.rsplit(".", 1)[0]
        + "_WS_AVG_prediction.csv"
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== 최종 예측 ==========")
    print(result)

    print("\n미래 X 저장:", x_output_path)
    print("최종 예측 저장:", output_path)

    return model_A, model_B, result, future_x_df


# =========================================================
# 9. 실행
# =========================================================

if __name__ == "__main__":

    model_A, model_B, result, future_X = forecast_ws_avg(
        csv_path="LSTM_WS_AVG.csv",
        target_date="2026-11-05",
        data_start_date="2016-01-01",
        lookback=14,
        epochs=50,
        batch_size=32
    )
