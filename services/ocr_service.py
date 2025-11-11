# services/ocr_service.py
from __future__ import annotations

# --- 패키지 경로 보강(services/ 하위에서 utils/ import 가능하게) ---
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pillow_heif import register_heif_opener
from pdf2image import convert_from_path
import pytesseract, io, uuid, tempfile, re
from statistics import median
from typing import Tuple, Dict, Any

# 후처리
from utils.text_cleaner import clean_ocr_text

# ================= 기본 설정 =================
# (Ubuntu 기본 경로. Mac 등 환경에 맞게 조정 가능)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
register_heif_opener()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= OCR 엔진 초기화 =================
_HAS_CV2 = False
_HAS_PADDLE = False
_HAS_EASYOCR = False
_paddle = None
_easyocr = None

try:
    import cv2, numpy as np
    _HAS_CV2 = True
except Exception:
    pass

try:
    from paddleocr import PaddleOCR
    _paddle = PaddleOCR(lang='korean', use_angle_cls=True, show_log=False)
    _HAS_PADDLE = True
except Exception:
    _paddle = None
    _HAS_PADDLE = False

try:
    import easyocr
    _easyocr = easyocr.Reader(['ko', 'en'], gpu=False)  # CPU/M1 안전
    _HAS_EASYOCR = True
except Exception:
    _easyocr = None
    _HAS_EASYOCR = False


# ================= 파일 저장 =================
def save_upload_to_png(file: UploadFile, raw: bytes, pdf_dpi: int = 200) -> str:
    """
    업로드 파일을 PNG로 변환 및 저장.
    - PDF는 dpi=200으로 렌더(속도 개선)
    - 이미지 짧은 변 1600px로 리사이즈(인식률/속도 밸런스)
    """
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
            pages = convert_from_path(tmp_pdf, dpi=pdf_dpi)
            if not pages:
                raise HTTPException(400, "PDF 페이지를 읽지 못했습니다.")
            pages[0].save(out_png, "PNG")
            os.remove(tmp_pdf)
        else:
            img = Image.open(io.BytesIO(raw))
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            MAX_SHORT = 1600
            w, h = img.size
            short = min(w, h)
            if short > MAX_SHORT:
                scale = MAX_SHORT / short
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

            img.save(out_png, "PNG")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"파일 처리 실패: {type(e).__name__}")
    return out_png


# ================= 전처리 =================
def preprocess_doc(png_path: str, mode: str = "doc") -> Image.Image:
    """CLAHE + 대비 강화 전처리 (OpenCV 우선, 없으면 Pillow fallback)"""
    if _HAS_CV2:
        img = cv2.imdecode(np.fromfile(png_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "이미지 로드 실패")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        if mode == "table":
            bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 35, 10)
            return Image.fromarray(bin_img)
        return Image.fromarray(gray)
    else:
        img = Image.open(png_path).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        return img


# ================= OCR 엔진들 =================
def _ocr_with_conf_tesseract(img: Image.Image, lang="kor+eng", psm=6, timeout=60) -> Tuple[str, float]:
    """
    Tesseract 호출 + word-level confidence로 중앙값 스코어 계산.
    - 큰 이미지는 max side 2000px로 축소(속도/안정성)
    - timeout 기본 60초
    """
    W, H = img.size
    max_side = max(W, H)
    if max_side > 2000:
        scale = 2000 / max_side
        img = img.resize((int(W*scale), int(H*scale)), Image.LANCZOS)

    cfg = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    try:
        data = pytesseract.image_to_data(
            img, config=cfg, lang=lang, timeout=timeout,
            output_type=pytesseract.Output.DICT
        )
        text = pytesseract.image_to_string(img, config=cfg, lang=lang, timeout=timeout) or ""
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(500, "Tesseract 미설치")
    except RuntimeError as e:  # pytesseract timeout은 RuntimeError로 올라오는 경우가 많음
        # 상위에서 timeout 스킵 로직 처리하도록 메시지 유지
        raise RuntimeError("Tesseract process timeout") from e

    confs = []
    for c in data.get("conf", []):
        try:
            confs.append(float(c))
        except Exception:
            continue
    score = median(confs) if confs else -1.0
    return text.strip(), score


def _ocr_with_paddle(img: Image.Image) -> Tuple[str | None, float]:
    """PaddleOCR 호출(텍스트 결합 + 평균 스코어)"""
    if not (_HAS_PADDLE and _HAS_CV2):
        return None, -1.0
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
            if txt and txt.strip():
                lines.append(txt.strip())
                confs.append(conf * 100 if 0 <= conf <= 1 else conf)

    text = "\n".join(lines).strip()
    score = (sum(confs) / len(confs)) if confs else -1.0
    return text, score


def _ocr_with_easyocr(img: Image.Image) -> Tuple[str | None, float]:
    """EasyOCR 호출(텍스트 결합 + 평균 스코어)"""
    if not _HAS_EASYOCR:
        return None, -1.0
    arr = np.array(img.convert("RGB"))
    res = _easyocr.readtext(arr)
    lines, confs = [], []
    for _, txt, conf in res:
        if txt and str(txt).strip():
            lines.append(str(txt).strip())
            confs.append(float(conf))
    text = "\n".join(lines).strip()
    score = (sum(confs) / len(confs)) if confs else -1.0
    return text, score


# ================= 후처리 =================
def _postprocess(text: str) -> str:
    """띄어쓰기/줄바꿈/구두점 보정(한국어 친화)"""
    if not text:
        return ""
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
def ocr_best(
    img: Image.Image,
    lang: str = "kor+eng",
    psms: Tuple[int, ...] = (6,),     # 기본 psm 6만 시도(문단/문장)
    timeout: int = 60,                # 기본 60초
    use_paddle: bool = True,
    use_easyocr: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    다중 엔진 호출 → 신뢰도(score)로 최고 결과 선택.
    - 테서랙트 timeout 발생 시 이후 PSM은 즉시 스킵하고 다른 엔진으로 전환.
    """
    best_t = ("", -1.0, None)
    tesseract_failed = False

    # Tesseract (여러 PSM)
    for p in psms:
        if tesseract_failed:
            break
        try:
            t, s = _ocr_with_conf_tesseract(img, lang=lang, psm=p, timeout=timeout)
            if s > best_t[1]:
                best_t = (t, s, p)
        except Exception as e:
            print(f"[OCR] psm={p} 실패: {e}")
            if "timeout" in str(e).lower():
                tesseract_failed = True

    t_text, t_score, chosen_psm = best_t

    # Paddle
    p_text, p_score = (None, -1.0)
    if use_paddle:
        try:
            p_text, p_score = _ocr_with_paddle(img)
        except Exception as e:
            print(f"[PaddleOCR] 실패: {e}")

    # EasyOCR
    e_text, e_score = (None, -1.0)
    if use_easyocr:
        try:
            e_text, e_score = _ocr_with_easyocr(img)
        except Exception as e:
            print(f"[EasyOCR] 실패: {e}")

    # 후보 정리 및 최종 선택
    candidates = [
        ("tesseract", t_text, t_score),
        ("paddle",    p_text, p_score),
        ("easyocr",   e_text, e_score),
    ]
    # 내용 없는 후보는 score -1로 취급
    def eff_score(x): return x[2] if (x[1] and isinstance(x[2], (int, float))) else -1.0
    best = max(candidates, key=eff_score)

    final_text = _postprocess(best[1] or "")
    final_text = clean_ocr_text(final_text)

    return (final_text or "(인식 결과 없음)"), {
        "engine": best[0],
        "score": round(best[2], 2) if isinstance(best[2], (int, float)) else -1.0,
        "tesseract_score": round(t_score, 2) if isinstance(t_score, (int, float)) else -1.0,
        "paddle_score":    round(p_score, 2) if isinstance(p_score, (int, float)) else -1.0,
        "easyocr_score":   round(e_score, 2) if isinstance(e_score, (int, float)) else -1.0,
        "psm": chosen_psm
    }


# ================= 세그먼트 영역 OCR (세그멘테이션 연동) =================
def ocr_text_region(img_path: str, bbox: list[int]) -> Dict[str, Any]:
    """
    세그멘테이션된 텍스트 영역(bbox)에 대해 OCR 수행.
    return: {"text": ..., "meta": {...}}
    """
    if not _HAS_CV2:
        raise HTTPException(500, "cv2 미설치로 영역 OCR 불가")
    import cv2
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise HTTPException(400, "이미지 로드 실패")
    x1, y1, x2, y2 = map(int, bbox[:4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1+1, x2), max(y1+1, y2)
    roi = bgr[y1:y2, x1:x2]
    pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    text, meta = ocr_best(pil_roi, lang="kor+eng", psms=(6,), timeout=60)
    return {"text": text, "meta": meta}


# ================= End-to-End(단일 업로드 OCR) =================
def run_ocr_on_upload(
    file: UploadFile,
    raw: bytes,
    mode: str = "doc",
    lang: str = "kor+eng",
    timeout: int = 60,
    use_paddle: bool = True,
    use_easyocr: bool = False
) -> dict:
    """
    업로드 저장 → 전처리 → 다중엔진 OCR → 후처리 → 결과 반환
    (세그멘트 없는 단일 페이지 OCR용)
    """
    # 업로드 저장 / 전처리
    png = save_upload_to_png(file, raw, pdf_dpi=200)
    img = preprocess_doc(png, mode=mode)

    # OCR 수행
    text, meta = ocr_best(
        img, lang=lang, psms=(6,),
        timeout=timeout, use_paddle=use_paddle, use_easyocr=use_easyocr
    )

    # 결과 리턴
    return {"text": text, "meta": meta}
