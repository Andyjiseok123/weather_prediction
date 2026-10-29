# Run: python train_models.py --start-date 2016-01-01
import argparse
import pickle
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from lightgbm import LGBMRegressor, log_evaluation, record_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from weather_model import (
    LOOKBACK,
    load_dataset,
    create_x_sequences,
    create_y_sequences,
    build_x_model,
    build_y_model,
    build_dense_ensemble_model,
    fit_model,
    recursive_forecast_x,
    create_lightgbm_features,
)


def make_model_id(csv_path, data_root):
    relative_path = csv_path.relative_to(data_root)
    return "__".join(relative_path.with_suffix("").parts)

def is_model_complete(model_dir):
    """모델 및 학습 결과물이 모두 저장된 경우 완료로 판단."""
    required_files = [
        "metadata.pkl",
        "x_model.keras",
        "y_model.keras",
        "dense_ensemble_model.keras",
        "lightgbm_model.pkl",
        "ensemble_input_scaler.pkl",
        "training_history/loss_history.csv",
        "training_history/loss_overview.png",
    ]

    return all((model_dir / name).is_file() for name in required_files)


def calc_metrics(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def make_lgbm_model(n_estimators=500):
    return LGBMRegressor(
        objective="regression",
        metric="l2",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )


def valid_feature_mask(features):
    values = features.to_numpy(dtype=np.float64)
    return features.notna().all(axis=1).to_numpy(dtype=bool) & np.isfinite(values).all(axis=1)


def make_validation_y_sequences(combined_X_raw, X_scaler, split_index, n_rows, lookback, n_features):
    combined_X_scaled = X_scaler.transform(combined_X_raw)
    sequences = []
    for i in range(split_index, n_rows):
        start = i - lookback + 1
        sequences.append(combined_X_scaled[start:i + 1])
    return np.asarray(sequences, dtype=np.float32).reshape(-1, lookback, n_features)


def save_history_artifacts(model_dir, model_id, histories, lgbm_history):
    """Write one long-form CSV and one loss plot per model."""
    history_dir = model_dir / "training_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name, history in histories.items():
        train_values = history.get("loss", [])
        val_values = history.get("val_loss", [])
        for i, train_loss in enumerate(train_values, start=1):
            rows.append({
                "model": model_name,
                "step_type": "epoch",
                "step": i,
                "train_mse": float(train_loss),
                "val_mse": float(val_values[i - 1]) if i <= len(val_values) else np.nan,
            })
            # Per-model plot (epoch based)
        if train_values:
            fig, ax = plt.subplots(figsize=(9, 5.2))
            ax.plot(range(1, len(train_values) + 1), train_values, label="Train MSE", linewidth=2)
            if val_values:
                ax.plot(range(1, len(val_values) + 1), val_values, label="Validation MSE", linewidth=2)
            ax.set_title(f"{model_id} | {model_name} Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE loss")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(history_dir / f"{model_name.lower()}_loss.png", dpi=180)
            plt.close(fig)

    # LightGBM records the MSE for each boosting iteration.
    train_values = lgbm_history.get("training", {}).get("l2", [])
    val_values = lgbm_history.get("validation", {}).get("l2", [])
    # Compatibility with LightGBM versions that call the metric "mse".
    if not train_values:
        train_values = lgbm_history.get("training", {}).get("mse", [])
    if not val_values:
        val_values = lgbm_history.get("validation", {}).get("mse", [])

    for i, train_loss in enumerate(train_values, start=1):
        rows.append({
            "model": "stage2_lightgbm",
            "step_type": "boosting_iteration",
            "step": i,
            "train_mse": float(train_loss),
            "val_mse": float(val_values[i - 1]) if i <= len(val_values) else np.nan,
        })

    if train_values:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.plot(range(1, len(train_values) + 1), train_values, label="Train MSE", linewidth=2)
        if val_values:
            ax.plot(range(1, len(val_values) + 1), val_values, label="Validation MSE", linewidth=2)
        ax.set_title(f"{model_id} | Stage 2 LightGBM Loss")
        ax.set_xlabel("Boosting iteration")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(history_dir / "stage2_lightgbm_loss.png", dpi=180)
        plt.close(fig)

    history_df = pd.DataFrame(rows, columns=["model", "step_type", "step", "train_mse", "val_mse"])
    history_df.to_csv(history_dir / "loss_history.csv", index=False, encoding="utf-8-sig")

    # Combined overview for presentation. Since LSTM epochs and LightGBM iterations
    # are different units, the x-axis is intentionally shown separately per panel.
    model_names = list(histories.keys()) + (["stage2_lightgbm"] if train_values else [])
    if model_names:
        fig, axes = plt.subplots(len(model_names), 1, figsize=(10, 3.8 * len(model_names)), squeeze=False)
        for ax, name in zip(axes.ravel(), model_names):
            if name == "stage2_lightgbm":
                x_train = train_values
                x_val = val_values
                xlabel = "Boosting iteration"
            else:
                h = histories[name]
                x_train = h.get("loss", [])
                x_val = h.get("val_loss", [])
                xlabel = "Epoch"
            if x_train:
                ax.plot(range(1, len(x_train) + 1), x_train, label="Train MSE", linewidth=1.8)
            if x_val:
                ax.plot(range(1, len(x_val) + 1), x_val, label="Validation MSE", linewidth=1.8)
            ax.set_title(name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("MSE")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"Training Loss Overview: {model_id}", y=1.002)
        fig.tight_layout()
        fig.savefig(history_dir / "loss_overview.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def validate_pipeline(X_raw, y_raw, dates, split_index, x_columns, lookback, epochs, batch_size):
    """Chronological holdout validation. Returns metrics and training histories."""
    X_train_raw = X_raw[:split_index]
    y_train_raw = y_raw[:split_index]
    X_scaler = MinMaxScaler().fit(X_train_raw)
    y_scaler = MinMaxScaler().fit(y_train_raw)
    X_train_scaled = X_scaler.transform(X_train_raw)
    y_train_scaled = y_scaler.transform(y_train_raw)
    n_features = X_raw.shape[1]

    X_x_train, y_x_train = create_x_sequences(X_train_scaled, lookback)
    X_y_train, y_y_train = create_y_sequences(X_train_scaled, y_train_scaled, lookback)
    model_x = build_x_model(lookback, n_features)
    model_y = build_y_model(lookback, n_features)
    model_x, hist_x = fit_model(model_x, X_x_train, y_x_train, epochs, batch_size, return_history=True)
    model_y, hist_y = fit_model(model_y, X_y_train, y_y_train, epochs, batch_size, return_history=True)

    validation_dates = dates[split_index:]
    predicted_X_raw = recursive_forecast_x(
        model_x=model_x, x_scaler=X_scaler, x_columns=x_columns,
        history_X_raw=X_train_raw, future_dates=validation_dates, lookback=lookback,
    )
    combined_X_raw = np.vstack([X_train_raw, predicted_X_raw])
    X_y_validation = make_validation_y_sequences(
        combined_X_raw, X_scaler, split_index, len(X_raw), lookback, n_features
    )
    y_pred_bilstm_scaled = model_y.predict(X_y_validation, verbose=0)
    y_pred_bilstm = y_scaler.inverse_transform(y_pred_bilstm_scaled).ravel()
    y_true = y_raw[split_index:].ravel()

    # LightGBM training and per-iteration evaluation history.
    train_features = create_lightgbm_features(X_train_raw, x_columns)
    train_mask = valid_feature_mask(train_features)
    if train_mask.sum() < 20:
        raise ValueError("LightGBM 학습에 유효한 Feature가 부족합니다. 365일 LAG와 데이터 길이를 확인하세요.")
    all_features = create_lightgbm_features(combined_X_raw, x_columns)
    validation_features = all_features.iloc[split_index:].copy().replace([np.inf, -np.inf], np.nan)
    if validation_features.isna().any().any():
        raise ValueError("검증 Feature에 결측치가 있습니다. 365일 LAG 및 관측 이력을 확인하세요.")

    lgbm_model = make_lgbm_model()
    lgbm_history = {}
    lgbm_model.fit(
        train_features.loc[train_mask], y_train_raw.ravel()[train_mask],
        eval_set=[(train_features.loc[train_mask], y_train_raw.ravel()[train_mask]),
                  (validation_features, y_true)],
        eval_names=["training", "validation"],
        eval_metric="l2",
        callbacks=[record_evaluation(lgbm_history), log_evaluation(period=0)],
    )
    y_pred_lgbm = lgbm_model.predict(validation_features)

    # Train the stacking model on aligned in-sample base-model predictions from
    # the training period, then evaluate it on the chronological holdout period.
    train_bilstm_scaled = model_y.predict(X_y_train, verbose=0)
    train_bilstm_raw = y_scaler.inverse_transform(train_bilstm_scaled).ravel()
    train_indices_for_lstm = np.arange(lookback - 1, len(X_train_raw))
    train_lgbm_pred = lgbm_model.predict(train_features)
    meta_mask = train_mask.copy()
    meta_mask[:lookback - 1] = False
    meta_indices = np.flatnonzero(meta_mask)
    lstm_positions = meta_indices - (lookback - 1)
    meta_train_raw = np.column_stack([train_bilstm_raw[lstm_positions], train_lgbm_pred[meta_indices]])
    meta_y_scaled = y_scaler.transform(y_train_raw[meta_indices]).ravel()
    meta_scaler = StandardScaler().fit(meta_train_raw)
    meta_train = meta_scaler.transform(meta_train_raw).astype(np.float32)
    meta_validation_raw = np.column_stack([y_pred_bilstm, y_pred_lgbm])
    meta_validation = meta_scaler.transform(meta_validation_raw).astype(np.float32)

    dense_model = build_dense_ensemble_model(2)
    dense_model, hist_dense = fit_model(
        dense_model, meta_train, meta_y_scaled.reshape(-1, 1), epochs, batch_size,
        validation_data=(meta_validation, y_scaler.transform(y_raw[split_index:]).astype(np.float32)),
        return_history=True,
        early_stopping=False,
    )
    y_pred_ensemble_scaled = dense_model.predict(meta_validation, verbose=0)
    y_pred_ensemble = y_scaler.inverse_transform(y_pred_ensemble_scaled).ravel()

    # X is also assessed with MSE (and RMSE as a scale-readable companion).
    x_metrics = calc_metrics(X_raw[split_index:], predicted_X_raw)
    bilstm_metrics = calc_metrics(y_true, y_pred_bilstm)
    lgbm_metrics = calc_metrics(y_true, y_pred_lgbm)
    ensemble_metrics = calc_metrics(y_true, y_pred_ensemble)

    metrics = {
        "x_mse": x_metrics["mse"], "x_rmse": x_metrics["rmse"],
        "y_bilstm_mse": bilstm_metrics["mse"], "y_bilstm_rmse": bilstm_metrics["rmse"],
        "y_lightgbm_mse": lgbm_metrics["mse"], "y_lightgbm_rmse": lgbm_metrics["rmse"],
        "y_ensemble_mse": ensemble_metrics["mse"], "y_ensemble_rmse": ensemble_metrics["rmse"],
    }
    histories = {"stage1_bilstm": hist_x, "stage2_bilstm": hist_y, "dense_ensemble": hist_dense}
    return metrics, histories, lgbm_history


def train_one_csv(csv_path, data_root, output_root, start_date, date_column, lookback, epochs, batch_size, validation_fraction):
    print("\n" + "=" * 70)
    print(f"학습 대상: {csv_path}")
    data = load_dataset(csv_path, start_date=start_date, date_column=date_column)
    X_raw, y_raw, dates = data["X"], data["y"], data["dates"]
    x_columns, target_column = data["x_columns"], data["target_column"]
    n_rows, n_features = X_raw.shape
    if n_rows <= lookback + 10:
        raise ValueError(f"데이터가 부족합니다: {n_rows}행, lookback={lookback}")
    split_index = int(n_rows * (1.0 - validation_fraction))
    if split_index <= max(lookback + 1, 370) or split_index >= n_rows:
        raise ValueError("학습/검증 분할이 부적절하거나 365일 LAG를 위한 학습 데이터가 부족합니다.")

    print(f"데이터 행 수: {n_rows} | X 특성: {n_features} | Y: {target_column}")
    print(f"관측 기간: {dates[0].date()} ~ {dates[-1].date()} | 학습: {split_index}행 | 검증: {n_rows - split_index}행")

    metrics, histories, lgbm_history = validate_pipeline(
        X_raw, y_raw, dates, split_index, x_columns, lookback, epochs, batch_size
    )
    print("\n[시간 순서 기반 검증 결과: MSE / RMSE]")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    # Refit deployable base models on the full observed history.
    x_scaler = MinMaxScaler().fit(X_raw)
    y_scaler = MinMaxScaler().fit(y_raw)
    X_scaled = x_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw)
    X_x, y_x = create_x_sequences(X_scaled, lookback)
    X_y, y_y = create_y_sequences(X_scaled, y_scaled, lookback)

    final_x_model = build_x_model(lookback, n_features)
    final_y_model = build_y_model(lookback, n_features)
    final_x_model, final_hist_x = fit_model(final_x_model, X_x, y_x, epochs, batch_size, return_history=True)
    final_y_model, final_hist_y = fit_model(final_y_model, X_y, y_y, epochs, batch_size, return_history=True)

    lgbm_features = create_lightgbm_features(X_raw, x_columns)
    lgbm_mask = valid_feature_mask(lgbm_features)
    if lgbm_mask.sum() < 20:
        raise ValueError("최종 LightGBM 학습에 유효한 Feature가 부족합니다.")
    final_lgbm_model = make_lgbm_model()
    final_lgbm_history = {}
    final_lgbm_model.fit(
        lgbm_features.loc[lgbm_mask], y_raw.ravel()[lgbm_mask],
        eval_set=[(lgbm_features.loc[lgbm_mask], y_raw.ravel()[lgbm_mask])],
        eval_names=["training"], eval_metric="l2",
        callbacks=[record_evaluation(final_lgbm_history), log_evaluation(period=0)],
    )

    # Fit final Dense stacker using aligned base-model predictions on the full data.
    train_bilstm_raw = y_scaler.inverse_transform(final_y_model.predict(X_y, verbose=0)).ravel()
    lstm_indices = np.arange(lookback - 1, len(X_raw))
    train_lgbm_raw = final_lgbm_model.predict(lgbm_features)
    meta_mask = lgbm_mask.copy()
    meta_mask[:lookback - 1] = False
    meta_indices = np.flatnonzero(meta_mask)
    meta_lstm_positions = meta_indices - (lookback - 1)
    meta_train_raw = np.column_stack([train_bilstm_raw[meta_lstm_positions], train_lgbm_raw[meta_indices]])
    meta_y_scaled = y_scaler.transform(y_raw[meta_indices]).astype(np.float32)
    ensemble_input_scaler = StandardScaler().fit(meta_train_raw)
    meta_train = ensemble_input_scaler.transform(meta_train_raw).astype(np.float32)
    final_dense_model = build_dense_ensemble_model(2)
    final_dense_model.fit(
        meta_train, meta_y_scaled, epochs=epochs, batch_size=batch_size,
        validation_split=0.1 if len(meta_train) >= 20 else 0.0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)] if len(meta_train) >= 20 else [],
        shuffle=False, verbose=0,
    )

    model_id = make_model_id(csv_path, data_root)
    model_dir = output_root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    final_x_model.save(model_dir / "x_model.keras")
    final_y_model.save(model_dir / "y_model.keras")
    final_dense_model.save(model_dir / "dense_ensemble_model.keras")
    with open(model_dir / "lightgbm_model.pkl", "wb") as f:
        pickle.dump(final_lgbm_model, f)
    with open(model_dir / "ensemble_input_scaler.pkl", "wb") as f:
        pickle.dump(ensemble_input_scaler, f)

    metadata = {
        "model_id": model_id,
        "relative_csv_path": str(csv_path.relative_to(data_root)),
        "x_columns": x_columns,
        "target_column": target_column,
        "date_column": data["date_column"],
        "start_date": start_date,
        "last_observed_date": str(dates[-1].date()),
        "lookback": lookback,
        "n_features": n_features,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "validation_metrics": metrics,
        "lgbm_feature_columns": list(lgbm_features.columns),
        "ensemble_type": "dense_stacking",
        "ensemble_input_columns": ["bilstm_prediction", "lightgbm_prediction"],
        "ensemble_hidden_units": [16, 8],
    }
    with open(model_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Loss charts use chronological holdout training runs. For LightGBM, show
    # validation-based iteration curves; final full-data fit history is not used
    # to claim generalization performance.
    save_history_artifacts(model_dir, model_id, histories, lgbm_history)
    print(f"\n모델 및 학습 그래프 저장 완료: {model_dir}")
    print(f"손실 CSV: {model_dir / 'training_history' / 'loss_history.csv'}")
    print(f"손실 그래프: {model_dir / 'training_history' / 'loss_overview.png'}")
    return model_id, metrics


def main():
    parser = argparse.ArgumentParser(description="6_dataset의 모든 LSTM_*.csv 학습")
    parser.add_argument("--data-dir", default="6_dataset", help="데이터 폴더")
    parser.add_argument("--output-dir", default="saved_models", help="저장 모델 폴더")
    parser.add_argument("--start-date", default=None, help="날짜 열이 없는 CSV의 첫 행 날짜. 예: 2016-01-01")
    parser.add_argument("--date-column", default=None, help="날짜 열 이름을 명시할 때 사용")
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if args.lookback < 1:
        raise ValueError("--lookback은 1 이상이어야 합니다.")
    if not 0.05 <= args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction은 0.05 이상 0.5 미만이어야 합니다.")

    data_root = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {data_root}")
    csv_files = sorted(path for path in data_root.rglob("LSTM_*.csv") if path.is_file())
    if not csv_files:
        print(f"학습 대상 CSV가 없습니다: {data_root}")
        return

    
    print(f"검색된 CSV 개수: {len(csv_files)}")
    print(f"모델 저장 위치: {output_root}")

    success_count = 0
    skipped_count = 0
    failed_files = []

    for index, csv_path in enumerate(csv_files, start=1):
        model_id = make_model_id(csv_path, data_root)
        model_dir = output_root / model_id

        # 기존 학습 완료 모델은 건너뛰기
        if is_model_complete(model_dir):
            skipped_count += 1
            print(
                f"\n[{index}/{len(csv_files)}] "
                f"[건너뜀] {model_id} — 기존 결과 확인",
                flush=True,
            )
            continue

        print(
            f"\n[{index}/{len(csv_files)}] "
            f"[학습 시작] {model_id}",
            flush=True,
        )

        try:
            train_one_csv(
                csv_path,
                data_root,
                output_root,
                args.start_date,
                args.date_column,
                args.lookback,
                args.epochs,
                args.batch_size,
                args.validation_fraction,
            )
            success_count += 1

        except Exception as exc:
            print(
                f"\n[학습 실패] {csv_path}\n원인: {exc}",
                flush=True,
            )
            traceback.print_exc()
            failed_files.append((csv_path, str(exc)))

        print(
            f"[진행 현황] 학습 성공: {success_count} | "
            f"건너뜀: {skipped_count} | "
            f"실패: {len(failed_files)}",
            flush=True,
        )

    print("\n" + "=" * 70)
    print("전체 학습 작업 종료")
    print(f"이번 실행 학습 성공: {success_count}")
    print(f"기존 완료 모델 건너뜀: {skipped_count}")
    print(f"학습 실패: {len(failed_files)}")

    if failed_files:
        print("\n실패한 파일 목록:")
        for path, reason in failed_files:
            print(f"- {path}\n  원인: {reason}")


if __name__ == "__main__":
    main()
