# db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:비밀번호@localhost:3306/docassistant?charset=utf8mb4"
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,  # 필요하면 True
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# FastAPI/Flask 어디서든 쓰는 공용 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
