# crud.py (업데이트 버전)
from sqlalchemy.orm import Session
from sqlalchemy import select
from models import OCRRecord
from datetime import datetime
import json

# 1) OCR 전용 생성(기존 유지) — 나중에 필요하면 계속 사용
def create_ocr_record(
    db: Session, *,
    filename: str,
    raw_text: str,
    parsed: dict | str,
    score: int,
    tier: str
) -> OCRRecord:
    rec = OCRRecord(
        filename=filename,
        raw_text=raw_text,
        parsed=json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else parsed,
        score=score,
        tier=tier,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec

# 2) 🔥 통합 생성: OCR + 세그멘테이션 + 시각화까지 한 번에 저장
def create_full_record(
    db: Session, *,
    filename: str,
    ocr_text: str,
    parsed: dict | str,
    seg_json: dict | str,
    vis_path: str | None = None,
    score: int = 0,
    tier: str = "default"
) -> OCRRecord:
    rec = OCRRecord(
        filename=filename,
        raw_text=ocr_text,
        parsed=json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else parsed,
        seg_json=json.dumps(seg_json, ensure_ascii=False) if isinstance(seg_json, dict) else seg_json,
        vis_path=vis_path,
        score=score,
        tier=tier,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec

# 3) get — 최신 스타일
def get_record(db: Session, record_id: int) -> OCRRecord | None:
    return db.get(OCRRecord, record_id)

# 4) 리스트
def list_records(db: Session, limit: int = 50):
    stmt = select(OCRRecord).order_by(OCRRecord.id.desc()).limit(limit)
    return db.execute(stmt).scalars().all()
