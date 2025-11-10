# services/segment.py — OpenCV only (테이블 감지 + 전체 텍스트 블록)
import os, cv2, numpy as np

def _opencv_layout_tables(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 35, 10)
    h, w = gray.shape[:2]
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//50), 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//50)))
    hor = cv2.morphologyEx(thr, cv2.MORPH_OPEN, hor_kernel, iterations=1)
    ver = cv2.morphologyEx(thr, cv2.MORPH_OPEN, ver_kernel, iterations=1)
    table_mask = cv2.add(hor, ver)
    table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), 2)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw*bh < (w*h)*0.01:   # 너무 작은 건 제외
            continue
        blocks.append({"id": f"t{x}_{y}", "type": "table", "bbox": [x, y, x+bw, y+bh], "content": None})
    return blocks

def segment_layout(img_path: str) -> dict:
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {img_path}")
    h, w = img.shape[:2]

    # 테이블 후보 추정
    table_blocks = _opencv_layout_tables(img)

    # 항상 텍스트 전역 블록 하나 추가(중첩 허용)
    blocks = [{"id": "b1", "type": "text", "bbox": [0, 0, w, h], "content": None}]
    blocks.extend(table_blocks)

    return {"engine": "opencv-only", "width": w, "height": h, "blocks": blocks}
