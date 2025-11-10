# utils/diagnose.py
import cv2, numpy as np

def diagnose(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 블러 지표(라플라시안 분산)
    blur = cv2.Laplacian(g, cv2.CV_64F).var()
    # 대비 지표(표준편차)
    contrast = g.std()
    # 기울기 대략 추정
    th = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)[1]
    coords = np.column_stack(np.where(th > 0))
    angle = 0.0
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
    return {"blur": blur, "contrast": float(contrast), "skew_deg": float(angle)}
