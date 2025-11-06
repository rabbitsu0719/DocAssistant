# main.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from importlib import import_module
from pathlib import Path
import os, uuid

app = FastAPI(title="OCR Only")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
os.makedirs("uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload_html", response_class=HTMLResponse)
async def upload_html_get(request: Request):
    return RedirectResponse(url="/")

@app.post("/upload_html", response_class=HTMLResponse)
async def upload_html(request: Request, file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "빈 파일입니다.")

        OCR = import_module("services.ocr_service")
        result = OCR.run_ocr_on_upload(file, raw, mode="doc", lang="kor+eng", use_paddle=True, use_easyocr=True)
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
        return templates.TemplateResponse("index.html", {"request": request, "text": text + info})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "text": f'❌ {type(e).__name__}: {e}'})
