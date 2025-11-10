# main.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from importlib import import_module
from pathlib import Path
import os, json, time

# DB
from db import get_db
from crud import create_ocr_record, get_record

# 세그멘테이션 / 시각화 / OCR 연결
from services.segment import segment_layout
from services.visualize import save_overlay
from services.ocr_service import ocr_text_region, save_upload_to_png

app = FastAPI(title="Smart Document Assistant")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
os.makedirs("uploads", exist_ok=True)
os.makedirs("captures", exist_ok=True)
app.mount("/captures", StaticFiles(directory="captures"), name="captures")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 홈
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "text": None})


# 업로드 페이지 GET → 홈으로 이동
@app.get("/upload_html", response_class=HTMLResponse)
async def upload_html_get(request: Request):
    return RedirectResponse(url="/")


# ======================================
# (1) 기존: 단일 OCR + DB 저장
# ======================================
@app.post("/upload_html", response_class=HTMLResponse)
async def upload_html(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
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

        info_lines = [
            f"engine={meta.get('engine')}",
            f"score={meta.get('score')}",
            f"tesseract_score={meta.get('tesseract_score')}",
            f"paddle_score={meta.get('paddle_score')}",
            f"easyocr_score={meta.get('easyocr_score')}",
        ]
        info = "\n\n(meta: " + ", ".join(info_lines) + ")"

        # DB 저장
        rec = create_ocr_record(
            db,
            filename=file.filename,
            raw_text=text,
            parsed=meta,
            score=0,
            tier="N/A",
        )

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "text": text + info + f"\n\n✅ 저장됨: /documents/{rec.id}"}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "text": f"❌ {type(e).__name__}: {e}"}
        )


# ======================================
# (2) 신규: 세그멘테이션 + 영역별 OCR + 오버레이 + DB 저장
# ======================================
@app.post("/upload_and_segment", response_class=HTMLResponse)
async def upload_and_segment(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        start_time = time.time()
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "빈 파일입니다.")

        # 1) 파일 저장 (PNG 변환)
        png_path = save_upload_to_png(file, raw)

        # 2) 문서 레이아웃 분석
        layout = segment_layout(png_path)

        # 3) 각 블록 OCR 수행
        for b in layout["blocks"]:
            if b["type"] == "text":
                b["ocr"] = ocr_text_region(png_path, b["bbox"])
            elif b["type"] == "table":
                b["table"] = {"raw": b.get("content")}

        # 4) 오버레이 이미지 생성
        overlay_name = f"{Path(file.filename).stem}_overlay.png"
        overlay_path = os.path.join("captures", overlay_name)
        save_overlay(png_path, layout, overlay_path)

        # 5) DB 저장
        record = create_ocr_record(
            db,
            filename=file.filename,
            raw_text="(세그멘테이션 결과)",
            parsed=layout,
            score=0,
            tier="layout",
        )

        elapsed = round(time.time() - start_time, 2)
        msg = f"✅ 세그멘테이션 완료 ({elapsed}s) /documents/{record.id}"

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "text": msg}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "text": f"❌ {type(e).__name__}: {e}"}
        )


# 저장된 문서 상세
@app.get("/documents/{record_id}", response_class=HTMLResponse)
async def document_detail(request: Request, record_id: int, db: Session = Depends(get_db)):
    rec = get_record(db, record_id)
    if not rec:
        raise HTTPException(404, "문서를 찾을 수 없습니다.")

    # 파일명 기준으로 오버레이 파일명 추정 (세그멘테이션 라우트에서 저장한 규칙과 일치)
    stem = Path(rec.filename).stem
    overlay_url = f"/captures/{stem}_overlay.png"

    return templates.TemplateResponse(
        "result_detail.html",
        {
            "request": request,
            "record_id": rec.id,
            "filename": rec.filename,
            "overlay_url": overlay_url,
            "doc_json": json.dumps(rec.parsed, ensure_ascii=False, indent=2),
        }
    )


# 하위호환: POST /upload_nutrition → upload_html 로 우회 처리
@app.post("/upload_nutrition", response_class=HTMLResponse)
async def upload_nutrition_compat(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return await upload_html(request=request, file=file, db=db)


# GET /upload_nutrition → 홈으로 이동
@app.get("/upload_nutrition", response_class=HTMLResponse)
async def upload_nutrition_get(request: Request):
    return RedirectResponse(url="/")


# 개발 테스트용 (정상 출력 확인)
@app.get("/_dev/echo", response_class=HTMLResponse)
async def dev_echo(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "text": "🔎 템플릿 출력 테스트 OK"
    })
