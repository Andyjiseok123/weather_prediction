
import argparse
import hashlib
import os
from pathlib import Path

import psycopg2
from psycopg2 import Binary
from dotenv import load_dotenv


# =========================================================
# 경로 및 설정
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "saved_models"

# DB에 저장할 모델 파일
# artifact_type은 DB에서 파일 종류를 식별하는 값
ARTIFACT_FILES = {
    "x_model": "x_model.keras",
    "y_model": "y_model.keras",
    "dense_ensemble_model": "dense_ensemble_model.keras",
    "lightgbm_model": "lightgbm_model.pkl",
    "ensemble_input_scaler": "ensemble_input_scaler.pkl",
    "metadata_pickle": "metadata.pkl",
}

# 예측에 필요한 최소 파일
REQUIRED_FILES = {
    "x_model.keras",
    "y_model.keras",
    "metadata.pkl",
}


# =========================================================
# 모델 ID 파싱
# 예: 강릉_강원특별자치도__LSTM_TA_AVG
# =========================================================

def parse_model_id(model_id):
    if "__" not in model_id:
        raise ValueError(
            f"모델 ID 형식이 올바르지 않습니다: {model_id}"
        )

    location_part, model_type_code = model_id.split("__", 1)

    if "_" not in location_part:
        raise ValueError(
            f"지역명과 광역 지역명을 구분할 수 없습니다: {model_id}"
        )

    # 현재 모델 ID 규칙:
    # {세부지역명}_{광역지역명}__{모델유형}
    locality_name, province_name = location_part.rsplit("_", 1)

    if not locality_name or not province_name or not model_type_code:
        raise ValueError(
            f"모델 ID에 지역명 또는 모델 유형이 없습니다: {model_id}"
        )

    return province_name, locality_name, model_type_code


# =========================================================
# 모델 폴더 검사
# =========================================================

def discover_models(model_root):
    if not model_root.exists():
        raise FileNotFoundError(
            f"모델 폴더를 찾을 수 없습니다: {model_root}"
        )

    if not model_root.is_dir():
        raise NotADirectoryError(
            f"모델 경로가 디렉터리가 아닙니다: {model_root}"
        )

    model_dirs = sorted(
        path for path in model_root.iterdir()
        if path.is_dir() and "__" in path.name
    )

    if not model_dirs:
        raise FileNotFoundError(
            f"모델 폴더를 찾을 수 없습니다: {model_root}"
        )

    models = []
    skipped = []

    for model_dir in model_dirs:
        try:
            province_name, locality_name, model_type_code = (
                parse_model_id(model_dir.name)
            )
        except ValueError as exc:
            skipped.append((model_dir.name, str(exc)))
            continue

        missing = [
            filename
            for filename in REQUIRED_FILES
            if not (model_dir / filename).is_file()
        ]

        if missing:
            skipped.append((
                model_dir.name,
                f"필수 파일 누락: {', '.join(sorted(missing))}"
            ))
            continue

        artifacts = {}

        # 존재하는 모델 파일만 등록한다.
        # training_history 폴더의 CSV와 PNG는 저장하지 않는다.
        for artifact_type, filename in ARTIFACT_FILES.items():
            file_path = model_dir / filename

            if file_path.is_file():
                file_bytes = file_path.read_bytes()
                sha256 = hashlib.sha256(file_bytes).hexdigest()

                artifacts[artifact_type] = {
                    "file_name": filename,
                    "data": file_bytes,
                    "sha256": sha256,
                }

        models.append({
            "model_id": model_dir.name,
            "province_name": province_name,
            "locality_name": locality_name,
            "model_type_code": model_type_code,
            "artifacts": artifacts,
        })

    return models, skipped


# =========================================================
# PostgreSQL 연결
# =========================================================

def connect_db():
    load_dotenv(PROJECT_ROOT / ".env")

    required_env = [
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]

    missing_env = [
        name for name in required_env
        if not os.getenv(name)
    ]

    if missing_env:
        raise RuntimeError(
            ".env에 다음 PostgreSQL 환경변수가 없습니다: "
            + ", ".join(missing_env)
        )

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )


# =========================================================
# 지역 정보 등록
# =========================================================

def ensure_location(cursor, province_name, locality_name):
    # 광역 지역 등록
    cursor.execute(
        """
        INSERT INTO provinces (province_name)
        VALUES (%s)
        ON CONFLICT (province_name) DO NOTHING
        """,
        (province_name,),
    )

    # 세부 지역 등록
    cursor.execute(
        """
        INSERT INTO localities (locality_name, province_name)
        VALUES (%s, %s)
        ON CONFLICT (locality_name) DO NOTHING
        """,
        (locality_name, province_name),
    )

    # 기존 세부 지역의 소속 광역 지역 확인
    cursor.execute(
        """
        SELECT province_name
        FROM localities
        WHERE locality_name = %s
        """,
        (locality_name,),
    )

    row = cursor.fetchone()

    if row is None:
        raise RuntimeError(
            f"지역 등록에 실패했습니다: {locality_name}"
        )

    if row[0] != province_name:
        raise ValueError(
            f"지역명 충돌: '{locality_name}'의 기존 광역 지역은 "
            f"'{row[0]}'이지만 모델 ID에서는 '{province_name}'입니다."
        )


# =========================================================
# 모델 유형 등록 및 ID 조회
# =========================================================

def ensure_model_type(cursor, locality_name, model_type_code):
    cursor.execute(
        """
        INSERT INTO model_types (
            locality_name,
            model_type_code
        )
        VALUES (%s, %s)
        ON CONFLICT (locality_name, model_type_code)
        DO NOTHING
        """,
        (locality_name, model_type_code),
    )

    cursor.execute(
        """
        SELECT model_type_id
        FROM model_types
        WHERE locality_name = %s
          AND model_type_code = %s
        """,
        (locality_name, model_type_code),
    )

    row = cursor.fetchone()

    if row is None:
        raise RuntimeError(
            f"모델 유형 등록에 실패했습니다: "
            f"{locality_name} / {model_type_code}"
        )

    return row[0]


# =========================================================
# 모델 바이너리 저장
# =========================================================

def save_artifacts(cursor, model_type_id, artifacts):
    for artifact_type, artifact in artifacts.items():
        cursor.execute(
            """
            INSERT INTO model_artifacts (
                model_type_id,
                artifact_type,
                file_name,
                artifact_data,
                sha256
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (model_type_id, artifact_type)
            DO UPDATE SET
                file_name = EXCLUDED.file_name,
                artifact_data = EXCLUDED.artifact_data,
                sha256 = EXCLUDED.sha256
            """,
            (
                model_type_id,
                artifact_type,
                artifact["file_name"],
                Binary(artifact["data"]),
                artifact["sha256"],
            ),
        )


# =========================================================
# 모델 하나를 트랜잭션으로 저장
# =========================================================

def save_one_model(conn, model):
    # 하나의 모델을 등록하는 과정 전체를 트랜잭션으로 처리한다.
    # 오류가 나면 해당 모델의 변경사항은 롤백된다.
    with conn:
        with conn.cursor() as cursor:
            ensure_location(
                cursor,
                model["province_name"],
                model["locality_name"],
            )

            model_type_id = ensure_model_type(
                cursor,
                model["locality_name"],
                model["model_type_code"],
            )

            save_artifacts(
                cursor,
                model_type_id,
                model["artifacts"],
            )

    return model_type_id


# =========================================================
# 실행
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "saved_models 폴더의 학습 모델을 PostgreSQL에 저장합니다."
        )
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="학습 모델 폴더 (기본값: saved_models)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB에 저장하지 않고 대상 모델만 검사합니다.",
    )

    args = parser.parse_args()

    model_root = Path(args.model_dir)

    if not model_root.is_absolute():
        model_root = PROJECT_ROOT / model_root

    model_root = model_root.resolve()

    print("=" * 65)
    print("기상 예측 모델 PostgreSQL 저장")
    print("=" * 65)
    print(f"모델 폴더: {model_root}")

    models, skipped = discover_models(model_root)

    print(f"\n저장 대상 모델 수: {len(models)}")
    print(f"건너뛸 모델 수: {len(skipped)}")

    for model in models:
        print(
            f"\n[대상] {model['model_id']}"
            f"\n  광역 지역: {model['province_name']}"
            f"\n  세부 지역: {model['locality_name']}"
            f"\n  모델 유형: {model['model_type_code']}"
            f"\n  파일 수: {len(model['artifacts'])}"
        )

    for model_id, reason in skipped:
        print(f"\n[건너뜀] {model_id}: {reason}")

    if args.dry_run:
        print("\n검사 완료. --dry-run 모드이므로 DB에 저장하지 않았습니다.")
        return

    if not models:
        print("\n저장할 수 있는 모델이 없습니다.")
        return

    conn = None
    success_count = 0
    failed_count = 0

    try:
        conn = connect_db()
        print("\nPostgreSQL 연결 성공")

        for model in models:
            try:
                model_type_id = save_one_model(conn, model)

                print(
                    f"\n[저장 완료] {model['model_id']}"
                    f" (model_type_id={model_type_id})"
                    f"\n  저장/갱신 파일 수: {len(model['artifacts'])}"
                )

                success_count += 1

            except Exception as exc:
                failed_count += 1
                print(
                    f"\n[저장 실패] {model['model_id']}: "
                    f"{type(exc).__name__}: {exc}"
                )

    except Exception as exc:
        print(
            f"\nDB 연결 또는 전체 실행 오류: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        if conn is not None:
            conn.close()

    print("\n" + "=" * 65)
    print("저장 작업 결과")
    print("=" * 65)
    print(f"성공 모델: {success_count}")
    print(f"실패 모델: {failed_count}")
    print(f"검사 단계에서 건너뛴 모델: {len(skipped)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
