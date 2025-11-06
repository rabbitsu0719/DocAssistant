# models.py
from datetime import datetime
from sqlalchemy import Integer, String, Text, JSON, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from db import Base

class OCRRecord(Base):
    __tablename__ = "ocr_results"   # DB에 있는 테이블명

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    parsed: Mapped[dict] = mapped_column(JSON, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    tier: Mapped[str] = mapped_column(String(8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
