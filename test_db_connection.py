
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# 프로젝트 루트의 .env 로드
load_dotenv(Path(__file__).resolve().parent / ".env")

# 아래 5개 환경변수 이름을 save_model_to_db.py에서
# 실제로 사용 중인 이름과 동일하게 맞출 것
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

print("DB_HOST 설정 여부:", bool(DB_HOST))
print("DB_PORT:", DB_PORT)
print("DB_NAME 설정 여부:", bool(DB_NAME))
print("DB_USER 설정 여부:", bool(DB_USER))
print("DB_PASSWORD 설정 여부:", bool(DB_PASSWORD))

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10,
    )
    print("PostgreSQL 연결 성공!")
    conn.close()

except psycopg2.OperationalError as e:
    print("PostgreSQL 연결 실패:")
    print(e)
