# models.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, func
from sqlalchemy import declarative_base
#from sqlalchemy.orm import Mapped, mapped_column, declarative_base
from db import Base

Base = declarative_base()

class OCRRecord(Base):
    __tablename__ = "ocr_results"   # DB에 있는 테이블명
    #__tablename__ = "ocr_records" 

    #id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    id = Column(Integer, primary_key=True, autoincrement=True)
    #filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    #raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text = Column(Text, nullable=True)
    #parsed: Mapped[dict] = mapped_column(JSON, nullable=False)
    parsed   = Column(Text, nullable=True)
    #score: Mapped[int] = mapped_column(Integer, nullable=False)
    score    = Column(Integer, default=0)
    #tier: Mapped[str] = mapped_column(String(8), nullable=False)
    tier     = Column(String(50), default="N/A")

    # ⬇️ 세그멘테이션 추가 (이번에 새로 추가)
    seg_json = Column(Text, nullable=True)      # 레이아웃 원본 JSON (문자열)
    vis_path = Column(String(255), nullable=True)  # overlay 파일명 (예: foo_20231112_overlay.png)

    #created_at: Mapped[datetime] = mapped_column(
    #    DateTime(timezone=True),
    #    server_default=func.now(),
    #)
    #created_at = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, default=datetime.utcnow)

    # 시각화/결과 경로
    overlay_path = Column(Text)
    tables_dir = Column(Text)
    ocr_json_path = Column(Text)
    