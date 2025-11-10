# utils/text_cleaner.py
import re, unicodedata

# ==============================
# 정규식 패턴 정의
# ==============================
_KO_SENT_END = re.compile(r'[.?!…]+["”\']?$|[가-힣][다요]\s*$')  # 한국어 문장 끝 추정
_LIST_BULLET = re.compile(r'^\s*(?:[-*•]|[\d]{1,3}[.)])\s+')    # 불릿/번호
_HYPHEN_BR   = re.compile(r'(\w)-\n(\w)')                       # 하이픈 줄바꿈
_SPACE_BEFORE= re.compile(r'\s+([,.;:!?%)\]”])')                # 구두점 앞 공백
_SPACE_AFTER = re.compile(r'([(\[“])\s+')                       # 괄호 뒤 공백
_MULTI_SPACE = re.compile(r'[ \t]{2,}')                         # 다중 공백
_MULTI_NL    = re.compile(r'\n{3,}')                            # 다중 빈줄

# ==============================
# OCR 후처리 핵심 함수
# ==============================
def clean_ocr_text(text: str) -> str:
    """
    OCR 결과 텍스트 후처리:
      - 하이픈+줄바꿈 정리
      - 문장 단위로 줄바꿈 재구성
      - 구두점/괄호 주변 공백 보정
      - 다중 공백/빈 줄 정리
    """
    if not text:
        return ""

    # 1. 유니코드 정규화 (NFKC: 한글 조합형/분리형 통합)
    t = unicodedata.normalize("NFKC", text)

    # 2. 하이픈 줄바꿈 보정 (예: "데이-\n터" → "데이터")
    t = _HYPHEN_BR.sub(r'\1\2', t)

    # 3. 줄 단위 분리
    lines, out, buf = t.splitlines(), [], ""

    def flush():
        """현재 문장 버퍼를 출력 목록에 추가"""
        nonlocal buf
        s = buf.strip()
        if s:
            out.append(s)
        buf = ""

    # 4. 문장 단위 재조립
    for ln in lines:
        ln = ln.rstrip()

        # 리스트 불릿이면 단독 라인으로 출력
        if _LIST_BULLET.match(ln):
            flush()
            out.append(ln.strip())
            continue

        # 버퍼가 비어 있으면 새 문장 시작
        if not buf:
            buf = ln
            continue

        # 버퍼가 문장 끝이 아니면 다음 줄 이어붙임
        if not _KO_SENT_END.search(buf):
            if ln and (ln[0].islower() or ln[0].isdigit() or ln[0] in '([{"\'' or re.match(r'^[가-힣]', ln)):
                buf = buf + " " + ln.strip()
                continue

        # 문장 끝났으면 flush 후 새 줄 시작
        flush()
        buf = ln
    flush()

    # 5. 구두점 주변 공백 보정
    t = "\n".join(out)
    t = _SPACE_BEFORE.sub(r'\1', t)
    t = _SPACE_AFTER.sub(r'\1', t)

    # 6. 다중 공백/빈 줄 정리
    t = _MULTI_SPACE.sub(' ', t)
    t = _MULTI_NL.sub('\n\n', t)

    return t.strip()
