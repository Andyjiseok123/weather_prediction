import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from weather_model import (
    load_dataset,
    recursive_forecast_x,
)


def predict_one_dataset(
    model_id,
    target_date,
    data_dir,
    model_dir,
):
    model_path = model_dir / model_id

    metadata_path = model_path / "metadata.pkl"
    x_model_path = model_path / "x_model.keras"
    y_model_path = model_path / "y_model.keras"

    # -----------------------------------------------------
    # 1. 저장된 메타데이터 로드
    # -----------------------------------------------------

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"메타데이터가 없습니다: {metadata_path}"
        )

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    lookback = metadata["lookback"]
    x_columns = metadata["x_columns"]
    target_column = metadata["target_column"]

    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]

    # -----------------------------------------------------
    # 2. 원본 CSV 로드
    # -----------------------------------------------------

    csv_path = data_dir / metadata["relative_csv_path"]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"원본 CSV를 찾을 수 없습니다: {csv_path}"
        )

    data = load_dataset(
        csv_path,
        start_date=metadata["start_date"],
        date_column=metadata["date_column"],
    )

    X_raw = data["X"]
    dates = data["dates"]

    # 학습 당시와 특성 열의 순서가 같은지 확인
    if data["x_columns"] != x_columns:
        raise ValueError(
            "학습 당시와 현재 CSV의 X 열 또는 열 순서가 다릅니다.\n"
            "원본 CSV가 변경되었는지 확인하세요."
        )

    if data["target_column"] != target_column:
        raise ValueError(
            "학습 당시와 현재 CSV의 Y 열이 다릅니다."
        )

    last_observed_date = pd.Timestamp(
        metadata["last_observed_date"]
    ).normalize()

    if dates[-1] != last_observed_date:
        raise ValueError(
            "현재 CSV의 마지막 관측일이 학습 당시와 다릅니다.\n"
            f"학습 당시: {last_observed_date.date()}\n"
            f"현재 CSV: {dates[-1].date()}\n"
            "데이터가 갱신되었다면 모델을 다시 학습하세요."
        )

    # -----------------------------------------------------
    # 3. 목표 날짜 확인
    # -----------------------------------------------------

    target_date = pd.Timestamp(target_date).normalize()

    if target_date <= last_observed_date:
        raise ValueError(
            "target_date는 마지막 관측일보다 미래여야 합니다.\n"
            f"마지막 관측일: {last_observed_date.date()}\n"
            f"요청 날짜: {target_date.date()}"
        )

    future_dates = pd.date_range(
        start=last_observed_date + pd.Timedelta(days=1),
        end=target_date,
        freq="D"
    )

    # -----------------------------------------------------
    # 4. 모델 로드
    # -----------------------------------------------------

    if not x_model_path.exists() or not y_model_path.exists():
        raise FileNotFoundError(
            f"저장된 모델 파일이 없습니다: {model_path}"
        )

    model_x = tf.keras.models.load_model(x_model_path)
    model_y = tf.keras.models.load_model(y_model_path)

    # -----------------------------------------------------
    # 5. 목표 날짜까지 X 재귀 예측
    # -----------------------------------------------------

    future_X_raw = recursive_forecast_x(
        model_x=model_x,
        x_scaler=x_scaler,
        x_columns=x_columns,
        history_X_raw=X_raw,
        future_dates=future_dates,
        lookback=lookback,
    )

    # 실제 과거 X + 예측 미래 X
    combined_X_raw = np.vstack([
        X_raw,
        future_X_raw
    ])

    # -----------------------------------------------------
    # 6. 목표 날짜의 X 시퀀스로 Y 예측
    # -----------------------------------------------------

    combined_X_scaled = x_scaler.transform(combined_X_raw)

    X_target_sequence = combined_X_scaled[-lookback:]
    X_target_sequence = X_target_sequence.reshape(
        1,
        lookback,
        len(x_columns)
    )

    y_pred_scaled = model_y.predict(
        X_target_sequence,
        verbose=0
    )

    y_pred = y_scaler.inverse_transform(y_pred_scaled)[0, 0]

    # 미래 X 중 목표 날짜에 해당하는 마지막 행
    target_X = future_X_raw[-1]

    result = {
        "model_id": model_id,
        "target_date": str(target_date.date()),
        "last_observed_date": str(last_observed_date.date()),
        "target_column": target_column,
        "predicted_y": float(y_pred),
        "predicted_x": {
            col: float(value)
            for col, value in zip(x_columns, target_X)
        },
        "forecast_days": len(future_dates),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="저장된 BiLSTM 모델로 목표 날짜의 Y 예측"
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="학습 시 출력된 모델 ID"
    )

    parser.add_argument(
        "--target-date",
        required=True,
        help="예측할 날짜. 예: 2026-11-05"
    )

    parser.add_argument(
        "--data-dir",
        default="6_dataset",
        help="원본 데이터 폴더"
    )

    parser.add_argument(
        "--model-dir",
        default="saved_models",
        help="저장된 모델 폴더"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()

    result = predict_one_dataset(
        model_id=args.model_id,
        target_date=args.target_date,
        data_dir=data_dir,
        model_dir=model_dir,
    )

    print("\n" + "=" * 60)
    print("기상 예측 결과")
    print("=" * 60)
    print(f"모델 ID       : {result['model_id']}")
    print(f"목표 날짜     : {result['target_date']}")
    print(f"마지막 관측일 : {result['last_observed_date']}")
    print(f"예측 대상 Y   : {result['target_column']}")
    print(f"예측 Y 값     : {result['predicted_y']:.6f}")
    print(f"재귀 예측 일수: {result['forecast_days']}일")

    print("\n[목표 날짜의 예측 X]")
    for column, value in result["predicted_x"].items():
        print(f"{column}: {value:.6f}")


if __name__ == "__main__":
    main()