# 📄 DocAssistant  
> **OCR · 문서 구조 분석 · 요약 · 질의응답을 통합한 개인 AI 문서 비서**

---

## ✨ 프로젝트 소개

**DocAssistant**는 PDF, 이미지 등 비정형 문서를 업로드하면  
OCR 기반 텍스트 추출부터 문서 구조 분석, 요약, 질의응답까지 수행하는  
**개인 AI 문서 비서 서비스**입니다.

단순 텍스트 추출이 아닌,  
문서의 **레이아웃과 문단 구조를 이해**하여  
사용자가 문서를 더 빠르게 파악하고 탐색할 수 있도록 설계했습니다.

---

## 🎯 프로젝트 목적

- 스캔 문서·보고서·논문 등 비정형 문서 처리 자동화
- OCR → 구조 분석 → NLP로 이어지는 **문서 처리 파이프라인 구축**
- AI 기능을 실제 **웹 서비스 형태로 구현**
- 백엔드 + AI 연계 시스템 설계 경험 확보

---

## 🧠 주요 기능

### 1️⃣ 다중 OCR 기반 텍스트 추출
- 이미지 / PDF 문서 업로드 지원
- 여러 OCR 엔진을 조합하여 인식 정확도 개선
  - Tesseract
  - PaddleOCR
  - EasyOCR

---

### 2️⃣ 문서 레이아웃 & 문단 분석
- 문서를 단순 텍스트가 아닌 **구조화된 데이터**로 처리
- 제목 / 본문 / 표 / 이미지 영역 분리
- 문단 단위 정제 및 재구성

---

### 3️⃣ 문서 요약 (Summarization)
- 긴 문서의 핵심 내용 자동 요약
- 문단별 요약 + 전체 요약 제공
- 학습·업무 문서에 적합한 요약 방식 적용

---

### 4️⃣ 문서 기반 질의응답 (Q&A)
- 업로드한 문서 내용만을 기반으로 질문 응답
- 문서 외 정보에 대한 응답 제한
- 실제 “문서를 읽어주는 비서” 역할 수행

---

## 🧩 전체 처리 흐름

```
Document Upload
      ↓
OCR (Image / PDF)
      ↓
Layout & Paragraph Segmentation
      ↓
Text Cleaning & Chunking
      ↓
Summarization / Q&A
```

---

## 🛠️ 기술 스택

### Backend
- Python
- FastAPI
- RESTful API 설계
- 비동기 기반 문서 업로드 처리

### Document AI
- OCR: Tesseract, PaddleOCR, EasyOCR
- Layout Analysis: LayoutParser, OpenCV
- NLP: 문서 요약, 질의응답 파이프라인

### Frontend
- Jinja2 Template
- 문서 업로드 및 결과 확인 UI

---

## 📂 프로젝트 구조

```
DocAssistant/
 ├─ app/
 │   ├─ routers/
 │   │   └─ document.py
 │   ├─ services/
 │   │   ├─ ocr_service.py
 │   │   ├─ layout_service.py
 │   │   └─ summary_service.py
 │   └─ templates/
 │       └─ index.html
 ├─ utils/
 │   ├─ image_utils.py
 │   └─ text_utils.py
 └─ main.py
```

---

## 👤 역할

**개인 프로젝트 (Solo Project)**

- 
