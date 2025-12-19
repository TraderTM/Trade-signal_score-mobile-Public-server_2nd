# Trade Signal (Personal) — iPhone에서 바로 쓰는 깔끔한 시그널 앱

이 프로젝트는 **너 혼자 쓰는 해외주식 시그널 화면**을 빠르게 만들기 위한 **Streamlit 기반 웹앱**이야.
아이폰에서 Safari로 접속한 다음 **홈 화면에 추가(Add to Home Screen)** 하면 앱처럼 바로 실행 가능해.

> 참고: 이 앱은 투자 조언이 아니고, 개인 참고용이야.

---

## 1) 설치 (PC/Mac 1번만)
Python 3.10+ 권장

```bash
pip install -r requirements.txt
```

## 2) 실행
```bash
streamlit run app.py
```

실행되면 터미널에 뜨는 URL로 접속해.
- PC에서: `http://localhost:8501`
- 아이폰에서: PC와 **같은 Wi‑Fi**에 연결된 상태에서
  - PC의 로컬 IP 확인 (예: 192.168.0.12)
  - 아이폰 Safari에서 `http://192.168.0.12:8501` 접속

## 3) 아이폰에서 앱처럼 쓰기
Safari에서 열고 → 공유 버튼 → **홈 화면에 추가**

---

## 화면 구성
- VIX 경고/주의/안정 바
- 티커 + 현재가
- 미니 차트(종가/MA10/거래량)
- AI 점수 `현재 → 다음` (드리프트 표시)
- 추세, 1W 성과, 파동/에너지/패턴, RSI/MFI
- 목표가(TARGET) / 손절가(STOP) 자동 제시

---

## 데이터
- 가격/거래량/지표: yfinance (일봉 기준)
- VIX: `^VIX`

---

## 커스터마이징 포인트
- `compute_score()` : 점수 산출 로직(가중치 조절)
- `calc_target_stop()` : 단타/스윙 목표/손절 ATR 배수 조절
- `label_wave/energy/pattern()` : 텍스트 라벨 규칙 변경

---

문제 생기면 에러 메시지 그대로 붙여주면 수정해줄게.
