"""
XGBoost 방향 예측 테스트

LSTM과의 핵심 차이:
  - 시퀀스 불필요 → 시점 t의 피처만으로 예측
  - 테이블형 피처(RSI, MACD 등)에서 LSTM보다 강함
  - 학습 속도 압도적으로 빠름
  - 피처 중요도 시각화 가능

추가 피처:
  - 래그 피처(lag 1~3): 전전일/전전전일 수익률/RSI 등
  - XGBoost는 시퀀스 없이 래그 피처로 시간 정보 처리
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import joblib

SYMBOL      = "379800.KS"
SYMBOL_NAME = "KODEX S&P500"
TRAIN_END   = "2023-12-31"
MODEL_PATH  = "lstm_test/weights/xgb_model.pkl"

os.makedirs("lstm_test/weights", exist_ok=True)

KST   = pytz.timezone("Asia/Seoul")
today = datetime.now(KST).date()

# ── 기술적 지표 ───────────────────────────────────────────────
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return 100 - (100 / (1 + g / l))

def calc_macd_hist(s):
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()

def calc_bb_pos(s, w=20):
    ma, std = s.rolling(w).mean(), s.rolling(w).std()
    return ((s - (ma - 2*std)) / (4*std + 1e-10)).clip(0, 1)

def align_external(series, target_index):
    return series.reindex(target_index, method='ffill').shift(1)

# ── 데이터 다운로드 ──────────────────────────────────────────
print("데이터 다운로드 중...")
start = "2014-01-01"
end   = str(today + timedelta(days=1))

kodex = yf.download(SYMBOL,     start=start, end=end, auto_adjust=True, progress=False)
spy   = yf.download("SPY",      start=start, end=end, auto_adjust=True, progress=False)
fx    = yf.download("USDKRW=X", start=start, end=end, auto_adjust=True, progress=False)
vix   = yf.download("^VIX",     start=start, end=end, auto_adjust=True, progress=False)

for df in [kodex, spy, fx, vix]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

close  = kodex['Close'].squeeze()
volume = kodex['Volume'].squeeze()
high   = kodex['High'].squeeze()
low    = kodex['Low'].squeeze()

# ── 피처 구성 ─────────────────────────────────────────────────
print("피처 생성 중...")
feat = pd.DataFrame(index=kodex.index)

# 기본 피처
feat['ret_1d']     = close.pct_change(1)
feat['ret_5d']     = close.pct_change(5)
feat['ret_10d']    = close.pct_change(10)
feat['ret_20d']    = close.pct_change(20)
feat['rsi']        = calc_rsi(close)
feat['macd_hist']  = calc_macd_hist(close)
feat['bb_pos']     = calc_bb_pos(close)
feat['vol_ratio']  = volume / volume.rolling(20).mean()
feat['hl_range']   = (high - low) / close
feat['spy_ret']    = align_external(spy['Close'].squeeze().pct_change(),  kodex.index)
feat['usdkrw_ret'] = align_external(fx['Close'].squeeze().pct_change(),   kodex.index)
feat['vix_chg']    = align_external(vix['Close'].squeeze().pct_change(),  kodex.index)

# 래그 피처 (XGBoost 전용 - 전일/전전일 정보)
for lag in [1, 2, 3]:
    feat[f'ret_1d_lag{lag}']   = feat['ret_1d'].shift(lag)
    feat[f'rsi_lag{lag}']      = feat['rsi'].shift(lag)
    feat[f'spy_ret_lag{lag}']  = feat['spy_ret'].shift(lag)
    feat[f'vix_chg_lag{lag}']  = feat['vix_chg'].shift(lag)

# 타겟
feat['target'] = (close.shift(-1) > close).astype(int)
feat = feat.dropna()

FEATURE_COLS = [c for c in feat.columns if c != 'target']
print(f"피처 수:   {len(FEATURE_COLS)}개")
print(f"데이터:    {feat.index[0].date()} ~ {feat.index[-1].date()}  ({len(feat)}일)")
print(f"상승 비율: {feat['target'].mean():.1%}\n")

# ── 학습 / 테스트 분리 ────────────────────────────────────────
train = feat[feat.index <= TRAIN_END]
test  = feat[feat.index >  TRAIN_END]

X_train, y_train = train[FEATURE_COLS].values, train['target'].values
X_test,  y_test  = test[FEATURE_COLS].values,  test['target'].values
test_dates        = test.index

# 검증셋 분리 (학습 데이터 마지막 10%)
val_size  = int(len(X_train) * 0.1)
X_val     = X_train[-val_size:]
y_val     = y_train[-val_size:]
X_tr      = X_train[:-val_size]
y_tr      = y_train[:-val_size]

print(f"학습: {len(X_tr)}일  |  검증: {len(X_val)}일  |  테스트: {len(X_test)}일\n")

# ── 모델 학습 / 로드 ──────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ 저장된 모델 로드: {MODEL_PATH}\n")
else:
    print("XGBoost 학습 중...")
    model = xgb.XGBClassifier(
        n_estimators     = 1000,
        max_depth        = 4,
        learning_rate    = 0.01,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        gamma            = 0.1,
        eval_metric      = 'logloss',
        early_stopping_rounds = 50,
        random_state     = 42,
        verbosity        = 1
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=100)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ 모델 저장: {MODEL_PATH}\n")

# ── 백테스트 ─────────────────────────────────────────────────
probs    = model.predict_proba(X_test)[:, 1]
pred_dir = (probs >= 0.5).astype(int)
actual   = y_test.astype(int)
overall  = (pred_dir == actual).mean()

print(f"\n{'='*60}")
print(f"  백테스트: {test_dates[0].date()} ~ {test_dates[-1].date()}  ({len(actual)}일)")
print(f"  전체 방향 정확도: {overall:.1%}")
print(f"\n  ─── 신뢰도 임계값별 분석 (LSTM 단일 모델 비교) ───")
print(f"  {'임계값':<8} {'XGBoost':<14} {'LSTM(clf)':<14} {'신호 수':<10} {'커버리지'}")
print(f"  {'-'*58}")

# LSTM 참고값 (이전 결과)
lstm_ref = {0.50: 0.544, 0.55: 0.594, 0.60: 0.688}

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = (probs >= thresh) | (probs <= 1-thresh)
    if mask.sum() == 0:
        print(f"  {thresh:.0%}↑      -              -              0일        0%")
        continue
    acc      = (pred_dir[mask] == actual[mask]).mean()
    n        = mask.sum()
    coverage = n / len(actual)
    lstm_val = f"{lstm_ref[thresh]:.1%}" if thresh in lstm_ref else "-"
    print(f"  {thresh:.0%}↑      {acc:.1%}          {lstm_val:<14} {n}일       {coverage:.0%}")

print(f"{'='*60}\n")

# ── 내일 예측 ─────────────────────────────────────────────────
tomorrow_prob = float(model.predict_proba(feat[FEATURE_COLS].values[-1:])[0, 1])
today_close   = float(close[feat.index[-1]])

next_day = today + timedelta(days=1)
while next_day.weekday() >= 5:
    next_day += timedelta(days=1)

if tomorrow_prob >= 0.65:
    direction, confidence = "📈 상승", "🔥 고신뢰"
elif tomorrow_prob >= 0.55:
    direction, confidence = "📈 상승", "✅ 보통"
elif tomorrow_prob <= 0.35:
    direction, confidence = "📉 하락", "🔥 고신뢰"
elif tomorrow_prob <= 0.45:
    direction, confidence = "📉 하락", "✅ 보통"
else:
    direction, confidence = "➖ 불확실", "⚠️ 관망 권장"

print(f"  오늘 종가:   {today_close:,.0f}원  ({feat.index[-1].date()})")
print(f"  내일 예측:   {direction}  확률 {tomorrow_prob:.1%}  {confidence}")
print(f"  예측 날짜:   {next_day}\n")

# ── 피처 중요도 (XGBoost 강점) ───────────────────────────────
importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
importance = importance.sort_values(ascending=True).tail(20)

# ── 차트 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

ax1 = axes[0]
recent_close = close[close.index >= test_dates[0]]
ax1.plot(recent_close.index, recent_close.values, color='steelblue', linewidth=1.2)
ax1.set_title(f"{SYMBOL_NAME} — 백테스트 기간 실제 종가")
ax1.set_ylabel("종가 (원)")

ax2 = axes[1]
ax2.plot(test_dates, probs, color='darkorange', linewidth=0.8, alpha=0.8, label="XGBoost 상승 확률")
ax2.axhline(0.65, color='green', linestyle='--', linewidth=1,   label='상승 고신뢰 (65%)')
ax2.axhline(0.35, color='red',   linestyle='--', linewidth=1,   label='하락 고신뢰 (35%)')
ax2.axhline(0.50, color='gray',  linestyle='-',  linewidth=0.5)
ax2.fill_between(test_dates, 0.65, 1.0, alpha=0.08, color='green')
ax2.fill_between(test_dates, 0.0, 0.35, alpha=0.08, color='red')
ax2.set_title("XGBoost 예측 상승 확률")
ax2.set_ylabel("확률")
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right')

ax3 = axes[2]
ax3.barh(importance.index, importance.values, color='steelblue', alpha=0.8)
ax3.set_title("XGBoost 피처 중요도 (상위 20개)")
ax3.set_xlabel("중요도 (Feature Importance)")

plt.tight_layout()
plt.savefig("lstm_test/result_xgboost.png", dpi=150)
print("📊 차트 저장 완료: lstm_test/result_xgboost.png")
