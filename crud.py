# crud.py
from sqlalchemy.orm import Session
from models import OCRRecord

def create_ocr_record(db: Session, *, filename: str, raw_text: str, parsed: dict, score: int, tier: str) -> OCRRecord:
    rec = OCRRecord(
        filename=filename,
        raw_text=raw_text,
        parsed=parsed,
        score=score,
        tier=tier,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec

def get_record(db: Session, record_id: int) -> OCRRecord | None:
    return db.query(OCRRecord).get(record_id)

def list_records(db: Session, limit: int = 50):
    return db.query(OCRRecord).order_by(OCRRecord.id.desc()).limit(limit).all()
