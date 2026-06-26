"""
Step 1: 분류 기반 방향 예측 테스트

핵심 변경:
  - 타겟: 내일 상승(1) / 하락(0) 이진 분류
  - 외부 피처: SPY 전일 수익률, USD/KRW 전일 수익률, VIX 전일 변동
  - 내부 피처: 다기간 수익률, RSI, MACD, BB, 거래량, 일중 변동폭
  - 신뢰도 필터링: 임계값별 정확도 분석
  - 백테스트: 2024~2026 (학습에 사용 안 한 데이터로 검증)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta
import pytz

SYMBOL       = "379800.KS"
SYMBOL_NAME  = "KODEX S&P500"
TIME_STEP    = 20       # 20 거래일 = 약 1달
EPOCHS       = 100
BATCH_SIZE   = 32
TRAIN_END    = "2023-12-31"
WEIGHTS_PATH = "lstm_test/weights/clf_weights.weights.h5"

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
    """외부 데이터를 KODEX 날짜에 맞추고 1일 시프트 (미래 누수 방지)"""
    return series.reindex(target_index, method='ffill').shift(1)

# ── 데이터 다운로드 ──────────────────────────────────────────
print("데이터 다운로드 중...")
start = "2014-01-01"
end   = str(today + timedelta(days=1))

kodex   = yf.download(SYMBOL,     start=start, end=end, auto_adjust=True, progress=False)
spy     = yf.download("SPY",      start=start, end=end, auto_adjust=True, progress=False)
fx      = yf.download("USDKRW=X", start=start, end=end, auto_adjust=True, progress=False)
vix     = yf.download("^VIX",     start=start, end=end, auto_adjust=True, progress=False)

for df in [kodex, spy, fx, vix]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

close  = kodex['Close'].squeeze()
volume = kodex['Volume'].squeeze()
high   = kodex['High'].squeeze()
low    = kodex['Low'].squeeze()

spy_ret    = spy['Close'].squeeze().pct_change()
usdkrw_ret = fx['Close'].squeeze().pct_change()
vix_chg    = vix['Close'].squeeze().pct_change()

# ── 피처 구성 ─────────────────────────────────────────────────
print("피처 생성 중...")
feat = pd.DataFrame(index=kodex.index)

# 내부: 다기간 수익률 (모멘텀)
feat['ret_1d']    = close.pct_change(1)
feat['ret_5d']    = close.pct_change(5)
feat['ret_10d']   = close.pct_change(10)
feat['ret_20d']   = close.pct_change(20)

# 내부: 기술적 지표
feat['rsi']       = calc_rsi(close)
feat['macd_hist'] = calc_macd_hist(close)
feat['bb_pos']    = calc_bb_pos(close)
feat['vol_ratio'] = volume / volume.rolling(20).mean()
feat['hl_range']  = (high - low) / close      # 일중 변동폭

# 외부: 전일 데이터 (한국장 시작 전 이미 확인 가능)
feat['spy_ret']    = align_external(spy_ret,    kodex.index)
feat['usdkrw_ret'] = align_external(usdkrw_ret, kodex.index)
feat['vix_chg']    = align_external(vix_chg,    kodex.index)

FEATURE_COLS = list(feat.columns)

# 타겟: 내일 종가 > 오늘 종가 → 1 (상승), 0 (하락)
feat['target'] = (close.shift(-1) > close).astype(float)
feat = feat.dropna()

print(f"피처:      {len(FEATURE_COLS)}개 {FEATURE_COLS}")
print(f"데이터:    {feat.index[0].date()} ~ {feat.index[-1].date()}  ({len(feat)}일)")
print(f"상승 비율: {feat['target'].mean():.1%}\n")

# ── 학습 / 테스트 분리 ────────────────────────────────────────
train = feat[feat.index <= TRAIN_END]
test  = feat[feat.index >  TRAIN_END]
print(f"학습: {len(train)}일 ({train.index[0].date()} ~ {train.index[-1].date()})")
print(f"테스트: {len(test)}일 ({test.index[0].date()} ~ {test.index[-1].date()})\n")

# ── 스케일링 (학습 데이터로만 fit) ────────────────────────────
scaler     = StandardScaler()
scaled_all = np.zeros((len(feat), len(FEATURE_COLS)))
scaled_all[:len(train)] = scaler.fit_transform(train[FEATURE_COLS].values)
scaled_all[len(train):] = scaler.transform(test[FEATURE_COLS].values)

# ── 데이터셋 ─────────────────────────────────────────────────
def make_dataset(X, y, ts):
    Xs = np.array([X[i:i+ts] for i in range(len(X)-ts)])
    Ys = np.array([y[i+ts]   for i in range(len(X)-ts)])
    return Xs, Ys

X_train, y_train = make_dataset(scaled_all[:len(train)], train['target'].values, TIME_STEP)

test_X_arr = scaled_all[len(train)-TIME_STEP:]
test_y_arr = test['target'].values
X_test = np.array([test_X_arr[i:i+TIME_STEP] for i in range(len(test_y_arr))])
y_test = test_y_arr

num_feat = X_train.shape[2]
print(f"학습 샘플: {X_train.shape[0]}  |  테스트 샘플: {X_test.shape[0]}  |  피처: {num_feat}\n")

# ── 모델 ─────────────────────────────────────────────────────
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIME_STEP, num_feat)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1,  activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ── 가중치 로드 or 학습 ───────────────────────────────────────
if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print(f"\n✅ 저장된 가중치 로드: {WEIGHTS_PATH}")
    print("   (재학습하려면 weights 파일 삭제 후 실행)\n")
else:
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]
    print("\n모델 학습 중...")
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_split=0.1, callbacks=callbacks, verbose=1)
    model.save_weights(WEIGHTS_PATH)
    print(f"\n✅ 가중치 저장: {WEIGHTS_PATH}\n")

    fig_loss, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'],     label="학습 손실", color="steelblue")
    ax.plot(history.history['val_loss'], label="검증 손실", color="orange")
    ax.set_title("학습 손실 곡선")
    ax.legend()
    plt.tight_layout()
    plt.savefig("lstm_test/result_clf_loss.png", dpi=150)
    plt.close()

# ── 백테스트 결과 ─────────────────────────────────────────────
probs    = model.predict(X_test).flatten()
pred_dir = (probs >= 0.5).astype(int)
actual   = y_test.astype(int)
test_dates = test.index

overall_acc = (pred_dir == actual).mean()

print(f"\n{'='*55}")
print(f"  백테스트: {test.index[0].date()} ~ {test.index[-1].date()}  ({len(actual)}일)")
print(f"  전체 방향 정확도: {overall_acc:.1%}")
print(f"\n  ─── 신뢰도 임계값별 분석 ───────────────────────")
print(f"  {'임계값':<10} {'정확도':<10} {'신호 수':<10} {'커버리지'}")
print(f"  {'-'*45}")
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = (probs >= thresh) | (probs <= 1-thresh)
    if mask.sum() == 0:
        continue
    acc      = (pred_dir[mask] == actual[mask]).mean()
    n        = mask.sum()
    coverage = n / len(actual)
    print(f"  {thresh:.0%}↑      {acc:.1%}      {n}일       {coverage:.0%}")
print(f"{'='*55}\n")

# ── 내일 방향 예측 ────────────────────────────────────────────
last_window   = scaler.transform(feat[FEATURE_COLS].values[-TIME_STEP:])
tomorrow_prob = float(model.predict(last_window.reshape(1, TIME_STEP, num_feat))[0][0])
today_close   = float(close[feat.index[-1]])

next_day = today + timedelta(days=1)
while next_day.weekday() >= 5:
    next_day += timedelta(days=1)

if tomorrow_prob >= 0.65:
    direction, confidence = "📈 상승", "🔥 고신뢰 (강한 신호)"
elif tomorrow_prob >= 0.55:
    direction, confidence = "📈 상승", "✅ 보통 신호"
elif tomorrow_prob <= 0.35:
    direction, confidence = "📉 하락", "🔥 고신뢰 (강한 신호)"
elif tomorrow_prob <= 0.45:
    direction, confidence = "📉 하락", "✅ 보통 신호"
else:
    direction, confidence = "➖ 불확실", "⚠️ 낮음 (관망 권장)"

print(f"  오늘 종가: {today_close:,.0f}원  ({feat.index[-1].date()})")
print(f"  내일({next_day}) 예측: {direction}")
print(f"  상승 확률: {tomorrow_prob:.1%}  |  {confidence}\n")

# ── 차트 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax1 = axes[0]
recent_close = close[close.index >= test.index[0]]
ax1.plot(recent_close.index, recent_close.values, color='steelblue', linewidth=1.2)
ax1.set_title(f"{SYMBOL_NAME} — 백테스트 기간 실제 종가")
ax1.set_ylabel("종가 (원)")

ax2 = axes[1]
ax2.plot(test_dates, probs, color='purple', linewidth=0.8, alpha=0.8, label="상승 확률")
ax2.axhline(0.65, color='green', linestyle='--', linewidth=1, label='상승 고신뢰 (65%)')
ax2.axhline(0.35, color='red',   linestyle='--', linewidth=1, label='하락 고신뢰 (35%)')
ax2.axhline(0.50, color='gray',  linestyle='-',  linewidth=0.5)
ax2.fill_between(test_dates, 0.65, 1.0, alpha=0.08, color='green')
ax2.fill_between(test_dates, 0.0, 0.35, alpha=0.08, color='red')
ax2.set_title("예측 상승 확률")
ax2.set_ylabel("확률")
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right')

ax3 = axes[2]
correct       = (pred_dir == actual).astype(float)
rolling_acc   = pd.Series(correct, index=test_dates).rolling(30).mean()
ax3.plot(test_dates, rolling_acc, color='darkorange', linewidth=1.2, label="롤링 정확도 (30일)")
ax3.axhline(0.5,        color='gray', linestyle='--', linewidth=0.8, label='기준선 (50%)')
ax3.axhline(overall_acc, color='blue', linestyle='--', linewidth=0.8, label=f'전체 평균 ({overall_acc:.1%})')
ax3.set_title("방향 정확도 (롤링 30일)")
ax3.set_ylabel("정확도")
ax3.set_ylim(0.2, 0.9)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig("lstm_test/result_classification.png", dpi=150)
print("📊 차트 저장 완료: lstm_test/result_classification.png")
