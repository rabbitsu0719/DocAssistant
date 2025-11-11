# services/visualize.py
from __future__ import annotations
import os, cv2, random
from typing import Dict, List, Any, Tuple

# BGR 컬러맵
COLORS: Dict[str, Tuple[int, int, int]] = {
    "text":   (50, 220, 50),
    "table":  (60, 160, 255),
    "figure": (255, 160, 60),
    # 필요 시 확장: "title", "header", "footer", "cell", "unknown" 등
}
DEFAULT_COLOR = (0, 255, 0)

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _to_int_bbox(bbox: List[float | int]) -> Tuple[int, int, int, int]:
    if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
        raise ValueError(f"bbox 형식 오류: {bbox}")
    x1, y1, x2, y2 = bbox[:4]
    return int(x1), int(y1), int(x2), int(y2)

def _choose_color(typ: str) -> Tuple[int, int, int]:
    if not isinstance(typ, str):
        return DEFAULT_COLOR
    return COLORS.get(typ.lower(), DEFAULT_COLOR)

def _draw_label_with_bg(img, x: int, y: int, text: str, bg_color: Tuple[int, int, int],
                        font_scale: float = 0.6, thickness: int = 2) -> None:
    # 텍스트 박스 크기 계산
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx1, ty1 = x, max(0, y - th - 6)
    tx2, ty2 = x + tw + 6, y
    # 배경(채움)
    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), bg_color, -1)
    # 흰 글씨로 라벨
    cv2.putText(img, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

def save_overlay(
    img_path: str,
    layout_json: Dict[str, Any],
    out_path: str,
    thickness: int = 2,
    font_scale: float = 0.6,
    min_area: int = 0,  # 너무 작은 박스는 스킵 (픽셀^2)
) -> str:
    """
    문서 레이아웃 결과(JSON)를 이미지에 오버레이하여 저장.
    - img_path: 원본 이미지 경로
    - layout_json: {"blocks":[{"type":str,"bbox":[x1,y1,x2,y2], "score":float?}, ...]} 형태 권장
                   (리스트 그대로 넘겨도 되고, 키 이름이 다르면 'bbox'/'box' fallback)
    - out_path: 저장 경로
    - thickness/font_scale: 시각화 파라미터
    - min_area: 최소 면적(너무 작은 박스 suppression)
    """
    _ensure_dir(os.path.dirname(out_path))

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {img_path}")
    H, W = img.shape[:2]

    # blocks 확보: dict 또는 list 모두 대응
    blocks = layout_json.get("blocks") if isinstance(layout_json, dict) else layout_json
    if not isinstance(blocks, list):
        raise ValueError("layout_json은 리스트이거나 {'blocks': [...]} 형태여야 합니다.")

    for i, b in enumerate(blocks, start=1):
        # 키 fallback: bbox/box/poly(사각형만 사용)
        bbox = b.get("bbox") or b.get("box") or b.get("poly")
        if bbox is None:
            # 키가 다른 원시 결과가 들어올 수 있으므로 스킵
            continue

        x1, y1, x2, y2 = _to_int_bbox(bbox)
        # 경계 클램핑
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        # 면적 필터
        if min_area > 0:
            if (x2 - x1) * (y2 - y1) < min_area:
                continue

        typ = (b.get("type") or b.get("cls") or "unknown")
        color = _choose_color(typ)

        # 사각형 박스
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 라벨 문자열: "index: type (score)"
        score = b.get("score")
        label = f"{i}: {typ}" if isinstance(typ, str) else f"{i}"
        if isinstance(score, (int, float)):
            label += f" ({score:.2f})"

        _draw_label_with_bg(img, x1, y1, label, color, font_scale, thickness)

    cv2.imwrite(out_path, img)
    return out_path
