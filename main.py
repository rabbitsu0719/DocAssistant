from __future__ import annotations
from models import Base, OCRRecord
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from importlib import import_module
from pathlib import Path
from datetime import datetime
import os, json, time
import cv2

# DB
from db import get_db, engine
from crud import create_ocr_record, create_full_record, get_record, list_records

# 세그멘테이션 / 시각화 / OCR 연결
from services.segment import segment_layout
from services.visualize import save_overlay
from services.ocr_service import ocr_text_region, save_upload_to_png


# -----------------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------------
def _as_obj(maybe_json):
    """DB에 문자열/JSON 혼재를 안전하게 dict로 변환"""
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str) and maybe_json.strip():
        try:
            return json.loads(maybe_json)
        except Exception:
            return {"_raw": maybe_json}
    return {}

# -----------------------------------------------------------------------------
# 앱/정적 경로
# -----------------------------------------------------------------------------
app = FastAPI(title="Smart Document Assistant")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 정적 제공: 캡처 이미지와 업로드 파일
os.makedirs(BASE_DIR / "uploads", exist_ok=True)
os.makedirs(BASE_DIR / "captures", exist_ok=True)
os.makedirs(BASE_DIR / "captures" / "tables", exist_ok=True)
app.mount("/captures", StaticFiles(directory=str(BASE_DIR / "captures")), name="captures")
app.mount("/uploads", StaticFiles(directory=str(BASE_DIR / "uploads")), name="uploads")

# -----------------------------------------------------------------------------
# 홈
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "text": None})

# 업로드 페이지 GET → 홈으로 이동
@app.get("/upload_html", response_class=HTMLResponse)
async def upload_html_get(request: Request):
    return RedirectResponse(url="/")

# -----------------------------------------------------------------------------
# (1) 단일 OCR + DB 저장
# -----------------------------------------------------------------------------
@app.post("/upload_html", response_class=HTMLResponse)
async def upload_html(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "빈 파일입니다.")

        OCR = import_module("services.ocr_service")
        result = OCR.run_ocr_on_upload(
            file, raw,
            mode="doc",
            lang="kor+eng",
            use_paddle=True,
            use_easyocr=True,
        )
        text = result.get("text", "(인식 결과 없음)")
        meta = result.get("meta", {})

        # DB 저장
        rec = create_ocr_record(
            db,
            filename=file.filename,
            raw_text=text,
            parsed=meta,
            score=0,
            tier="N/A",
        )
        # ✅ 바로 상세 페이지로 이동
        return RedirectResponse(url=f"/documents/{rec.id}", status_code=303)

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "text": f"❌ {type(e).__name__}: {e}"}
        )

# -----------------------------------------------------------------------------
# (2-A) 세그멘테이션 미리보기(시각화 이미지만 반환)
# -----------------------------------------------------------------------------
@app.post("/segment_preview")
async def segment_preview(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "빈 파일입니다.")

        # 1) PNG 저장 (절대경로 기대)
        png_path = save_upload_to_png(file, raw)

        # 2) 세그멘테이션
        layout = segment_layout(png_path)

        # 3) 오버레이 저장 (절대경로)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(file.filename).stem
        overlay_name = f"{stem}_{ts}_overlay.png"
        overlay_abs = BASE_DIR / "captures" / overlay_name
        save_overlay(png_path, layout, str(overlay_abs))

        # 4) 이미지 파일 응답 (절대경로)
        return FileResponse(str(overlay_abs), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세그멘테이션 미리보기 실패: {e}")

# -----------------------------------------------------------------------------
# (2-B) 세그멘테이션 + 영역별 OCR + 오버레이 + DB 저장(풀 파이프라인)
# -----------------------------------------------------------------------------
@app.post("/upload_and_segment", response_class=HTMLResponse)
async def upload_and_segment(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        t0 = time.time()
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "빈 파일입니다.")

        # 1) PNG 저장 (절대경로 기대)
        png_path = save_upload_to_png(file, raw)

        # 2) 문서 레이아웃 분석
        layout = segment_layout(png_path)
        if not isinstance(layout, dict) or "blocks" not in layout:
            raise RuntimeError("세그멘테이션 결과 형식이 올바르지 않습니다. {'blocks': [...]} 형식 필요")

        # 3) 각 블록 OCR 수행(텍스트만) & 표 썸네일 저장
        bgr_full = cv2.imread(png_path)
        if bgr_full is None:
            raise RuntimeError("이미지 로드 실패")
        H, W = bgr_full.shape[:2]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(file.filename).stem

        for idx, b in enumerate(layout["blocks"], start=1):
            typ = (b.get("type") or b.get("cls") or "").lower()
            bbox = b.get("bbox") or b.get("box") or b.get("poly")
            if not bbox or len(bbox) < 4:
                b["warn"] = "invalid_bbox"
                continue

            # 좌표 클램핑
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            if typ == "text":
                try:
                    b["ocr"] = ocr_text_region(png_path, [x1, y1, x2, y2])
                except Exception as ocr_e:
                    b["ocr_error"] = str(ocr_e)

            elif typ == "table":
                crop = bgr_full[y1:y2, x1:x2]
                if crop.size:
                    tbl_name = f"{stem}_{ts}_t{idx}.png"
                    tbl_abs = BASE_DIR / "captures" / "tables" / tbl_name
                    cv2.imwrite(str(tbl_abs), crop)
                    b.setdefault("table", {})
                    b["table"]["image_url"] = f"/captures/tables/{tbl_name}"
                    if "content" in b and b["content"] is not None:
                        b["table"]["raw"] = b["content"]

        # 4) 오버레이 이미지 생성 (절대경로 저장)
        overlay_name = f"{stem}_{ts}_overlay.png"
        overlay_abs = BASE_DIR / "captures" / overlay_name
        save_overlay(png_path, layout, str(overlay_abs))
        overlay_url = f"/captures/{overlay_name}"

        # 5) DB 저장 — create_full_record 사용 (파라미터명 주의: ocr_text)
        parsed_payload = {
            "layout": layout,
            "overlay_url": overlay_url,
            "source_png": str(Path(png_path).resolve().relative_to(BASE_DIR)), # resolve()를 추가하면 png_path가 상대경로이든 절대경로이든 BASE_DIR 기준의 절대경로로 변환된 뒤 relative_to() 작동
        }
        rec = create_full_record(
            db,
            filename=file.filename,
            ocr_text="(세그멘테이션 결과: 영역별 OCR 포함, 표 썸네일 생성)",
            parsed=parsed_payload,   # UI 친화 메타
            seg_json=layout,         # 모델 친화 원본 구조
            vis_path=overlay_name,   # 파일명만 저장
            score=0,
            tier="layout",
        )

        # ✅ 바로 상세 페이지로 이동
        return RedirectResponse(url=f"/documents/{rec.id}", status_code=303)

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "text": f"❌ {type(e).__name__}: {e}"}
        )

# -----------------------------------------------------------------------------
# 저장된 문서 상세(HTML)
# -----------------------------------------------------------------------------
@app.get("/documents/{record_id}", response_class=HTMLResponse)
async def document_detail(request: Request, record_id: int, db: Session = Depends(get_db)):
    rec = get_record(db, record_id)
    if not rec:
        raise HTTPException(404, "문서를 찾을 수 없습니다.")

    parsed_obj = _as_obj(rec.parsed)

    # vis_path 우선 → parsed.overlay_url → 과거 규칙 추정
    overlay_url = None
    if getattr(rec, "vis_path", None):
        overlay_url = f"/captures/{rec.vis_path}"
    if not overlay_url:
        overlay_url = parsed_obj.get("overlay_url")
    if not overlay_url:
        stem = Path(rec.filename).stem
        overlay_url = f"/captures/{stem}_overlay.png"

    return templates.TemplateResponse(
        "result_detail.html",
        {
            "request": request,
            "record_id": rec.id,
            "filename": rec.filename,
            "overlay_url": overlay_url,
            "doc_json": json.dumps(parsed_obj, ensure_ascii=False, indent=2),
            "parsed": parsed_obj
        }
    )

# -----------------------------------------------------------------------------
# 레이아웃 JSON API
# -----------------------------------------------------------------------------
@app.get("/api/documents/{record_id}/layout")
async def get_layout_json(record_id: int, db: Session = Depends(get_db)):
    rec = get_record(db, record_id)
    if not rec:
        raise HTTPException(404, "문서를 찾을 수 없습니다.")
    # seg_json 우선, 없으면 parsed.layout
    seg_obj = _as_obj(getattr(rec, "seg_json", {}))
    if seg_obj:
        return seg_obj
    parsed_obj = _as_obj(rec.parsed)
    return parsed_obj.get("layout", parsed_obj)

# -----------------------------------------------------------------------------
# 하위호환 라우트
# -----------------------------------------------------------------------------
@app.post("/upload_nutrition", response_class=HTMLResponse)
async def upload_nutrition_compat(
    request: Request, file: UploadFile = File(...), db: Session = Depends(get_db),
):
    return await upload_html(request=request, file=file, db=db)

@app.get("/upload_nutrition", response_class=HTMLResponse)
async def upload_nutrition_get(request: Request):
    return RedirectResponse(url="/")

# 개발 테스트용
@app.get("/_dev/echo", response_class=HTMLResponse)
async def dev_echo(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "text": "🔎 템플릿 출력 테스트 OK"
    })
