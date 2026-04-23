# Stock Bot — 전체 시스템 분석 문서

> 작성일: 2026-04-23  
> 프로젝트 경로: `/Users/songchang-ug/stock_bot`  
> 레포지토리: `git@github.com:scw8/chart_bot.git`

---

## 1. 프로젝트 개요

**목적**: ETF 가격 데이터를 주기적으로 수집하고, 텔레그램으로 정기 리포트 및 실시간 변동 알림을 발송하는 자동화 봇.  
**운영 방식**: GitHub Actions의 cron 스케줄로 자동 실행 (로컬 서버 불필요).  
**추가 개발 중**: XGBoost 기반 AI 방향 예측 기능 — 각 ETF의 다음 거래일 상승/하락 확률 예측.

---

## 2. 전체 파일 구조

```
stock_bot/
├── main.py              # 진입점 — 실행 모드에 따라 함수 분기
├── config.py            # ETF 목록, 알림 기준값, 시장 시간 설정
├── data.py              # yfinance로 ETF 데이터 수집
├── alerts.py            # 변동 감지 로직 (급락/급등/매수타이밍)
├── telegram.py          # 텔레그램 메시지 포맷팅 및 발송
├── scheduler.py         # 리포트/알림 실행 함수 묶음
├── .env                 # BOT_TOKEN, CHAT_ID (로컬 전용, git 제외)
├── .github/
│   └── workflows/
│       └── etf_bot.yml  # GitHub Actions 자동화 설정
└── lstm_test/           # AI 예측 모델 개발/테스트 폴더
    ├── test_tomorrow.py          # (구) LSTM 종가 회귀 예측
    ├── test_classification.py    # LSTM 이진 분류 (상승/하락) — 54.4% 정확도
    ├── test_ensemble.py          # 시드 앙상블 (3모델 평균) — 실패
    ├── test_ensemble_v2.py       # TIME_STEP 앙상블 (10/20/40일) — 55.6%
    ├── test_xgboost.py           # XGBoost 분류 — 57.7% (현재 최고)
    └── weights/                  # 학습된 모델 가중치 저장
        ├── clf_weights.weights.h5
        ├── ensemble_*.weights.h5
        ├── ensemble_v2_ts*.weights.h5
        └── xgb_model.pkl
```

---

## 3. 실행 흐름 (GitHub Actions → main.py)

```
GitHub Actions cron 트리거
        ↓
etf_bot.yml: github.event.schedule 값 확인
        ↓
python main.py <mode>
        ↓
  [정기 리포트]               [변동 알림]
  run_report(title, mode)     run_alert_check()
        ↓                           ↓
  get_all_etf_data()     get_etf_data_by_market("korea"/"us")
        ↓                           ↓
  send_report(...)           check_alerts(etf_data)
                                     ↓
                             send_alerts(alerts, market)
```

---

## 4. config.py — 중앙 설정

### ETF 목록 (총 14종)

| 이름 | 티커 | 통화 | 분류 |
|------|------|------|------|
| TIGER 미국나스닥100 | 133690.KS | KRW | 한국 ETF |
| TIGER 미국S&P500 | 360750.KS | KRW | 한국 ETF |
| KODEX S&P500 | 379800.KS | KRW | 한국 ETF |
| SPY | SPY | USD | 미국 S&P500 |
| SPLG | SPLG | USD | 미국 S&P500 |
| SPYG | SPYG | USD | 미국 S&P500 성장 |
| SSO | SSO | USD | S&P500 2배 레버리지 |
| QQQ | QQQ | USD | 나스닥 |
| QQQM | QQQM | USD | 나스닥 (소액용) |
| QLD | QLD | USD | 나스닥 2배 레버리지 |
| TQQQ | TQQQ | USD | 나스닥 3배 레버리지 |
| SCHD | SCHD | USD | 배당 |
| DGRW | DGRW | USD | 배당 성장 |
| VOO | VOO | USD | S&P500 |

### 알림 조건 (ALERT_CONFIG)

| 조건 | 기준값 | 설명 |
|------|--------|------|
| `drop_from_high` | -5.0% | 30일 고점 대비 -5% 이하 |
| `sudden_change` | ±1.0% | 직전 종가 대비 ±1% 이상 |
| `ma20_cross` | True | 20일 이동평균 하향 돌파 |

### 시장 시간

- 한국장: 09:00 ~ 15:30 KST
- 미국장: 22:30~05:00 KST (써머타임) / 23:30~06:00 KST (표준시) — `get_us_market_hours()`로 자동 감지

---

## 5. data.py — 데이터 수집

### `get_exchange_rate()`
- `yf.Ticker("USDKRW=X").history(period="1d")`로 달러 환율 1건 조회.

### `get_etf_data(ticker)`
- `yf.Ticker(ticker).history(period="60d")`로 60일치 OHLCV 데이터 조회.
- NaN 행 제거 후 반환.

### `get_all_etf_data()` / `get_etf_data_by_market(market)`
두 함수는 동일한 처리 로직을 공유:

1. 환율 조회
2. ETF_LIST 순회
3. 각 ETF별 계산:
   - `latest_price` = 가장 최근 종가
   - `high_30d` = 최근 30일 최고가
   - `ma20` = 최근 20일 단순이동평균
   - `prev_price` = 직전 종가
   - `drop_pct` = `(latest - high30d) / high30d * 100` (고점 대비 하락률)
   - `change_pct` = `(latest - prev) / prev * 100` (전일 대비 변동률)
4. KRW: 원화 포맷 / USD: 달러 + 원화 환산 병기
5. 데이터가 20일 미만이면 스킵

`get_etf_data_by_market`은 `market="korea"`면 KRW만, `"us"`면 USD만 필터링.

---

## 6. alerts.py — 변동 감지

### 신호 우선순위

| 우선순위 | 카테고리 | 조건 |
|---------|---------|------|
| 1 | 강력 (강한 매수 타이밍) | 20일선 하향 **AND** 고점대비 -5% 이하 |
| 2 | 급락 | 전일대비 -1% 이하 |
| 3 | 매수고려 | 20일선 하향 (단독) |
| 4 | 급등 | 전일대비 +1% 이상 |

### `check_alerts(etf_data)`
- 전체 ETF에 대해 `get_signal()` 호출
- 신호 있는 ETF만 우선순위별로 정렬
- 섹션별 메시지 문자열 리스트 반환:
  ```
  🚨 강한 매수 타이밍 (고점대비 -5% 이상)
  🔻 급락 감지
  📉 매수 고려
  🔺 급등 감지
  ```

### `get_signal(name, data)`
- 각 ETF 하나의 신호 분류 → 없으면 `None` 반환
- 있으면 `{"priority": int, "message": str}` 반환

---

## 7. telegram.py — 메시지 발송

### 환경변수
- `BOT_TOKEN`: 텔레그램 봇 토큰
- `CHAT_ID`: 수신 채팅방 ID
- `.env` 파일 또는 GitHub Secrets에서 주입

### `send_message(message)`
- Telegram Bot API `sendMessage` 엔드포인트 호출
- `parse_mode: "HTML"` 사용 (`<b>`, `<i>` 태그 지원)

### `_etf_line(name, data, signal_desc=None)`
각 ETF 한 줄 포맷 생성:
```
🟥 KODEX S&P500
   현재가: 17,340원 | 20일선: 16,800원
   고점대비: -3.2% | 전일대비: +0.8%
   → 20일선 하향 (고점대비 -3.2%)  ← signal_desc 있을 때만
```
- 상승이면 🟥, 하락이면 🟦, 보합이면 ➖

### `_classify(data)`
ETF를 카테고리로 분류:
- `"강한매수"`: 20일선 하향 + 고점대비 -5%
- `"급락"`: 전일대비 -1%
- `"매수고려"`: 20일선 하향 단독
- `"급등"`: 전일대비 +1%
- `"기타"`: 해당 없음

### `_build_market_section(etf_data)`
시장 섹션 구성 — 순서: **기타 → 강한매수 → 매수고려 → 급락 → 급등**
- 신호 없는 ETF가 먼저 나열되고, 이후 신호별 섹션 분리

### `send_report(title, etf_data, exchange_rate, mode)`
정기 리포트 발송:
- `mode="us_open"`: 미국장 먼저, 한국장 나중
- 나머지 모드: 한국장 먼저, 미국장 나중
- 포맷:
  ```
  📊 한국장 마감 요약
  🕐 2026년 04월 23일 15:30 기준
  💵 환율: 1,380원

  ━━━ 🇰🇷 한국장 ━━━
  [ETF 리스트]

  ━━━ 🇺🇸 미국장 ━━━
  [ETF 리스트]
  ```

### `send_alerts(alerts, market=None)`
변동 알림 발송:
- `market="korea"` → `🇰🇷 한국장` 헤더
- `market="us"` → `🇺🇸 미국장` 헤더
- 섹션 사이 구분선 (`─────────────────`) 삽입

---

## 8. scheduler.py — 실행 관리

### `is_market_open(market)`
현재 시간이 해당 장의 운영 시간 내인지 확인.

### `run_report(title, mode)`
정기 리포트 실행:
1. `get_all_etf_data()` — 전체 14개 ETF 데이터 수집
2. `send_report(title, etf_data, exchange_rate, mode)` — 텔레그램 발송

### `run_alert_check()`
변동 감지 실행:
1. 현재 열려있는 장(한국/미국) 확인
2. 해당 장의 ETF만 데이터 수집
3. 신호 감지 → 있으면 발송, 없으면 로그만 출력

---

## 9. main.py — 진입점

```
python main.py <mode>
```

| mode | 동작 | 발송 시각 |
|------|------|----------|
| `korea_open` | 한국장 시작 리포트 | 09:00 KST |
| `korea_noon` | 한국장 오후 현황 리포트 | 12:00 KST |
| `korea_close` | 한국장 마감 리포트 | 15:30 KST |
| `us_open` | 미국장 시작 리포트 | 22:30 / 23:30 KST |
| `alert` (default) | 변동 감지 체크 | 30분마다 |
| `test` | 즉시 테스트 실행 | 수동 |

---

## 10. GitHub Actions (etf_bot.yml)

### 트리거 스케줄

| cron 표현식 | UTC 시각 | KST 시각 | 동작 |
|------------|---------|---------|------|
| `0 0 * * 1-5` | 00:00 | 09:00 | `korea_open` |
| `0 3 * * 1-5` | 03:00 | 12:00 | `korea_noon` |
| `30 6 * * 1-5` | 06:30 | 15:30 | `korea_close` |
| `30 13 * * 1-5` | 13:30 | 22:30 | `us_open` (써머타임) |
| `30 14 * * 1-5` | 14:30 | 23:30 | `us_open` (표준시) |
| `0,30 0-21 * * 1-5` | 매 30분 | 매 30분 | `alert` |

### 모드 판별 로직
`github.event.schedule` 값으로 어떤 cron이 트리거됐는지 직접 비교:
```bash
SCHEDULE="${{ github.event.schedule }}"
if [ "$SCHEDULE" = "30 6 * * 1-5" ]; then
  python main.py korea_close
fi
```
> **왜 이 방식인가**: `date -u`로 현재 시간을 비교하면 GitHub Actions의 실행 지연(최대 수 분) 때문에 시간이 어긋나서 모드 판별 실패. `github.event.schedule`은 트리거된 cron 표현식 자체를 값으로 가지므로 정확.

### 실행 환경
- `ubuntu-latest` 러너
- Python 3.12
- 설치 패키지: `yfinance`, `finance-datareader`, `python-telegram-bot`, `requests`, `python-dotenv`, `schedule`, `pytz`
- Secrets: `BOT_TOKEN`, `CHAT_ID` → `.env` 파일로 주입

---

## 11. AI 예측 시스템 (lstm_test/)

### 개발 이력 및 결과 비교

| 단계 | 파일 | 방식 | 전체 정확도 | 60% 이상 신호 |
|------|------|------|------------|--------------|
| Step 0 | test_tomorrow.py | LSTM 종가 회귀 | - | - |
| Step 1 | test_classification.py | LSTM 이진 분류 | 54.4% | 68.8% / 16일 |
| Step 2a | test_ensemble.py | 시드 앙상블 (42/123/7) | 54.5% | 없음 |
| Step 2b | test_ensemble_v2.py | TIME_STEP 앙상블 (10/20/40) | 55.6% | 없음 |
| Step 3 | test_xgboost.py | XGBoost | **57.7%** | 없음 |

### 공통 피처 (12개)

| 피처 | 설명 | 누수 방지 |
|------|------|----------|
| `ret_1d` | 1일 수익률 | - |
| `ret_5d` | 5일 수익률 | - |
| `ret_10d` | 10일 수익률 | - |
| `ret_20d` | 20일 수익률 | - |
| `rsi` | RSI(14) | - |
| `macd_hist` | MACD 히스토그램 | - |
| `bb_pos` | 볼린저 밴드 내 위치 (0~1) | - |
| `vol_ratio` | 거래량 / 20일 평균 | - |
| `hl_range` | (고가-저가) / 종가 | - |
| `spy_ret` | SPY 전일 수익률 | `shift(1)` 적용 |
| `usdkrw_ret` | USD/KRW 전일 변동 | `shift(1)` 적용 |
| `vix_chg` | VIX 전일 변동률 | `shift(1)` 적용 |

> **외부 피처에 shift(1) 적용하는 이유**: 한국장 개장 시점에는 전날 미국 지수/환율 데이터만 알 수 있음. 당일 데이터 사용시 미래 누수(look-ahead bias) 발생.

### XGBoost 전용 추가 피처 (12개, 총 24개)

래그 피처 — XGBoost는 시퀀스 없이 이전 시점 값을 직접 피처로 사용:
- `ret_1d_lag1/2/3`, `rsi_lag1/2/3`, `spy_ret_lag1/2/3`, `vix_chg_lag1/2/3`

### 타겟 변수

```python
target = (close.shift(-1) > close).astype(int)
# 내일 종가 > 오늘 종가 → 1 (상승), 아니면 → 0 (하락)
```

### 학습/테스트 분리

- 학습: 2014-01-01 ~ 2023-12-31 (약 2,500일)
- 테스트: 2024-01-01 ~ 현재 (약 557일)
- **시간 순서를 지킨 분리** — 미래 데이터로 과거를 예측하는 오류 방지

### XGBoost 모델 설정

```python
xgb.XGBClassifier(
    n_estimators=1000,      # 트리 개수 (early stopping으로 실제로는 더 적게 사용)
    max_depth=4,            # 트리 깊이 제한 (과적합 방지)
    learning_rate=0.01,     # 학습률 낮게 설정
    subsample=0.8,          # 데이터 80%만 사용
    colsample_bytree=0.8,   # 피처 80%만 사용
    min_child_weight=5,     # 리프 노드 최소 샘플 수
    gamma=0.1,              # 분기 최소 이득
    early_stopping_rounds=50
)
```

### 신뢰도 임계값 분석 (KODEX S&P500 테스트 결과)

| 임계값 | XGBoost | LSTM(단일) | 신호 수 | 커버리지 |
|--------|---------|-----------|--------|---------|
| 50%+ | 57.7% | 54.4% | 전체 | 100% |
| 55%+ | - | 59.4% | 276일 | 50% |
| 60%+ | - | 68.8% | 16일 | 3% |

> **현재 한계**: XGBoost는 전체 정확도는 높지만 확률이 50% 근방에 몰려 고신뢰 신호를 거의 내지 않음. LSTM 단일 모델은 전체 정확도는 낮지만 60% 이상 신호에서는 정확도가 높음.

---

## 12. 계획 중인 기능: AI 예측 통합

### 개요
- 한국장 마감 (`korea_close`): 한국 ETF 3종 다음날 예측 표시
- 미국장 마감 (`us_close`, 신규): 미국 ETF 11종 다음날 예측 표시

### 텔레그램 메시지 변경 예시
```
🟥 KODEX S&P500
   현재가: 17,340원 | 20일선: 16,800원
   고점대비: -3.2% | 전일대비: +0.8%
   🤖 내일 예측: 📈 상승 62.3%
```

### 모델 저장 전략
- 기존 Supabase 프로젝트 내 `ml-models` Storage 버킷 신규 생성
- 파일명 규칙: `xgb_{ticker}.pkl` (예: `xgb_379800_KS.pkl`, `xgb_SPY.pkl`)
- 총 14개 모델 파일

### 재학습 주기
- **매일 재학습** (`korea_close` / `us_close` 실행 시마다)
- 최신 데이터 반영으로 정확도 유지
- XGBoost는 학습 속도가 빠르므로 (ETF당 약 5~10초) 전체 약 2~3분 소요

### 신규 추가 필요 사항
1. `lstm_test/predictor.py` — 모델 학습/예측/Supabase 연동 모듈
2. `etf_bot.yml` — `us_close` 스케줄 추가 (06:00 KST 써머타임 / 07:00 KST 표준시)
3. `main.py` — `us_close` 모드 추가
4. `scheduler.py` — `run_report`에 AI 예측 결과 주입
5. `telegram.py` — `_etf_line()`에 예측 결과 한 줄 추가
6. GitHub Secrets — `SUPABASE_URL`, `SUPABASE_KEY` 추가 필요

---

## 13. 환경 변수 목록

| 변수명 | 위치 | 용도 |
|--------|------|------|
| `BOT_TOKEN` | `.env` / GitHub Secrets | 텔레그램 봇 인증 |
| `CHAT_ID` | `.env` / GitHub Secrets | 텔레그램 발송 대상 |
| `SUPABASE_URL` | GitHub Secrets (예정) | Supabase 프로젝트 주소 |
| `SUPABASE_KEY` | GitHub Secrets (예정) | Supabase 인증 키 |

---

## 14. 알려진 이슈 및 해결 이력

| 이슈 | 원인 | 해결 방법 |
|------|------|----------|
| 정기 알림 미발송 | `date -u` 시간 비교 → GitHub Actions 실행 지연 | `github.event.schedule` 값으로 직접 비교 |
| SSH push 실패 | `~/.ssh/monkey` 키가 자동 로드 안 됨 | `~/.ssh/config`에 `IdentityFile ~/.ssh/monkey` 등록 |
| 한글 깨짐 | matplotlib 기본 폰트에 한글 없음 | `plt.rcParams['font.family'] = 'AppleGothic'` |
| 86억 원 예측 버그 | 역정규화 이중 적용 | 수동 역정규화만 사용 (`pred * scale + mean`) |
| 차트 shape mismatch | `test_dates = test.index[TIME_STEP:]` 날짜 20개 누락 | `test_dates = test.index` 로 수정 |
