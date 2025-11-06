from __future__ import annotations
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pillow_heif import register_heif_opener
from pdf2image import convert_from_path
import pytesseract, io, os, uuid, tempfile, re
from statistics import median

# ================= 기본 설정 =================
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
register_heif_opener()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= OCR 엔진 초기화 =================
try:
    import cv2, numpy as np
    from paddleocr import PaddleOCR
    _paddle = PaddleOCR(lang='korean', use_angle_cls=True, show_log=False)
    _HAS_PADDLE, _HAS_CV2 = True, True
except Exception:
    _paddle = None
    _HAS_PADDLE, _HAS_CV2 = False, False

try:
    import easyocr
    _easyocr = easyocr.Reader(['ko', 'en'], gpu=False)  # M1은 MPS 자동 인식
    _HAS_EASYOCR = True
except Exception:
    _easyocr = None
    _HAS_EASYOCR = False


# ================= 파일 저장 =================
def save_upload_to_png(file: UploadFile, raw: bytes) -> str:
    """업로드 파일을 PNG로 변환 및 저장"""
    if not raw:
        raise HTTPException(400, "빈 파일입니다.")
    ctype = (file.content_type or "").lower()
    fname = (file.filename or "").lower()
    out_png = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.png")

    try:
        if ctype == "application/pdf" or fname.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                tf.write(raw)
                tmp_pdf = tf.name
            pages = convert_from_path(tmp_pdf, dpi=300)
            if not pages:
                raise HTTPException(400, "PDF 페이지를 읽지 못했습니다.")
            pages[0].save(out_png, "PNG")
            os.remove(tmp_pdf)
        else:
            img = Image.open(io.BytesIO(raw))
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # 해상도 제한 (짧은 변 1600px)
            MAX_SHORT = 1600
            w, h = img.size
            short = min(w, h)
            if short > MAX_SHORT:
                scale = MAX_SHORT / short
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

            img.save(out_png, "PNG")
    except Exception as e:
        raise HTTPException(400, f"파일 처리 실패: {type(e).__name__}")
    return out_png


# ================= 전처리 =================
def preprocess_doc(png_path: str, mode: str = "doc") -> Image.Image:
    """CLAHE + 대비 강화 전처리"""
    if _HAS_CV2:
        img = cv2.imdecode(np.fromfile(png_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        if mode == "table":
            binImg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 35, 10)
            return Image.fromarray(binImg)
        return Image.fromarray(gray)
    else:
        img = Image.open(png_path).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        return img


# ================= OCR 엔진들 =================
def _ocr_with_conf_tesseract(img: Image.Image, lang="kor+eng", psm=4, timeout=30):
    cfg = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    try:
        data = pytesseract.image_to_data(
            img, config=cfg, lang=lang, timeout=timeout,
            output_type=pytesseract.Output.DICT
        )
        text = pytesseract.image_to_string(img, config=cfg, lang=lang, timeout=timeout) or ""
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(500, "Tesseract 미설치")
    confs = []
    for c in data.get("conf", []):
        try:
            confs.append(float(c))
        except Exception:
            continue
    score = median(confs) if confs else -1
    return text.strip(), score


def _ocr_with_paddle(img: Image.Image):
    """PaddleOCR 결과 파싱 안정화 버전"""
    if not _HAS_PADDLE or not _HAS_CV2:
        return None, -1.0
    import numpy as np, cv2
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    try:
        res = _paddle.ocr(bgr)
    except Exception as e:
        print(f"[PaddleOCR] 오류: {e}")
        return None, -1.0

    lines, confs = [], []
    for page in res:
        for item in page:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text_conf = item[1]
            try:
                txt, conf = text_conf[0], float(text_conf[1])
            except Exception:
                continue
            if txt.strip():
                lines.append(txt.strip())
                confs.append(conf * 100 if 0 <= conf <= 1 else conf)

    text = "\n".join(lines).strip()
    score = (sum(confs) / len(confs)) if confs else -1.0
    return text, score


def _ocr_with_easyocr(img: Image.Image):
    if not _HAS_EASYOCR:
        return None, -1.0
    import numpy as np
    arr = np.array(img.convert("RGB"))
    res = _easyocr.readtext(arr)
    lines, confs = [], []
    for _, txt, conf in res:
        lines.append(txt)
        confs.append(float(conf))
    text = "\n".join(lines).strip()
    score = (sum(confs) / len(confs)) if confs else -1.0
    return text, score


# ================= 후처리 =================
def _kor_ratio(t: str) -> float:
    n = len(t)
    return sum('가' <= ch <= '힣' for ch in t) / n if n else 0.0


def _choose_better(txt1, s1, txt2, s2):
    k1, k2 = _kor_ratio(txt1), _kor_ratio(txt2)
    w1, w2 = s1 + k1 * 10, s2 + k2 * 10
    return (txt1, s1) if w1 >= w2 else (txt2, s2)


def _postprocess(text: str) -> str:
    """띄어쓰기, 줄바꿈, 구두점 보정"""
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.replace(" ,", ",").replace(" :", ":").replace(" .", ".")
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 문장 중간 줄바꿈 제거
    text = re.sub(r'(?<=[가-힣0-9]),?\n(?=[가-힣0-9])', '', text)
    # 문장 부호 기준 줄바꿈 유지
    text = re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)
    return text.strip()


# ================= 메인 OCR 로직 =================
def ocr_best(img: Image.Image, lang="kor+eng", psms=(3, 4, 6), timeout=30,
             use_paddle=True, use_easyocr=False):
    best_t = ("", -1, None)
    for p in psms:
        try:
            t, s = _ocr_with_conf_tesseract(img, lang=lang, psm=p, timeout=timeout)
            if s > best_t[1]:
                best_t = (t, s, p)
        except Exception as e:
            print(f"[OCR] psm={p} 실패: {e}")
    t_text, t_score, chosen_psm = best_t

    # PaddleOCR
    p_text, p_score = (None, -1)
    if use_paddle:
        try:
            p_text, p_score = _ocr_with_paddle(img)
        except Exception as e:
            print(f"[PaddleOCR] 실패: {e}")

    # EasyOCR
    e_text, e_score = (None, -1)
    if use_easyocr:
        try:
            e_text, e_score = _ocr_with_easyocr(img)
        except Exception as e:
            print(f"[EasyOCR] 실패: {e}")

    # 세 엔진 중 최적 선택
    candidates = [
        ("tesseract", t_text, t_score),
        ("paddle", p_text, p_score),
        ("easyocr", e_text, e_score)
    ]
    best = max(candidates, key=lambda x: (x[2] if x[1] else -1))
    final_text = _postprocess(best[1])

    return final_text or "(인식 결과 없음)", {
        "engine": best[0],
        "score": round(best[2], 2),
        "tesseract_score": t_score,
        "paddle_score": p_score,
        "easyocr_score": e_score
    }


# ================= End-to-End =================
def run_ocr_on_upload(
    file: UploadFile,
    raw: bytes,
    mode: str = "doc",
    lang: str = "kor+eng",
    timeout: int = 30,
    use_paddle: bool = True,
    use_easyocr: bool = False
) -> dict:
    png = save_upload_to_png(file, raw)
    img = preprocess_doc(png, mode=mode)
    text, meta = ocr_best(img, lang=lang, psms=(3, 4, 6),
                          timeout=timeout, use_paddle=use_paddle, use_easyocr=use_easyocr)
    return {"text": text, "meta": meta}
