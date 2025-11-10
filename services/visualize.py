# services/visualize.py
import cv2, os

COLORS = {
    "text":   (50, 220, 50),
    "table":  (60, 160, 255),
    "figure": (255, 160, 60),
}

def save_overlay(img_path: str, layout_json: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = cv2.imread(img_path)
    if img is None: 
        raise ValueError(f"이미지를 읽을 수 없습니다: {img_path}")

    for b in layout_json.get("blocks", []):
        x1, y1, x2, y2 = b["bbox"]
        color = COLORS.get(b["type"], (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{b["type"]}'
        cv2.putText(img, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(out_path, img)
    return out_path
