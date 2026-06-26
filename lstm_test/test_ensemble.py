"""
Step 2: 앙상블 (3개 모델 평균)

단일 모델 대비 개선 목표:
  - 60%↑ 신호 수 증가 (현재 16일 → 목표 30일+)
  - 방향 정확도 유지 또는 향상
  - 3개 모델이 동의하는 구간만 고신뢰 신호로 처리
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta
import pytz

SYMBOL       = "379800.KS"
SYMBOL_NAME  = "KODEX S&P500"
TIME_STEP    = 20
EPOCHS       = 100
BATCH_SIZE   = 32
TRAIN_END    = "2023-12-31"
SEEDS        = [42, 123, 7]
WEIGHTS_DIR  = "lstm_test/weights"

os.makedirs(WEIGHTS_DIR, exist_ok=True)

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

def build_model(num_feat, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEP, num_feat)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1,  activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

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
feat = pd.DataFrame(index=kodex.index)
feat['ret_1d']     = close.pct_change(1)
feat['ret_5d']     = close.pct_change(5)
feat['ret_10d']    = close.pct_change(10)
feat['ret_20d']    = close.pct_change(20)
feat['rsi']        = calc_rsi(close)
feat['macd_hist']  = calc_macd_hist(close)
feat['bb_pos']     = calc_bb_pos(close)
feat['vol_ratio']  = volume / volume.rolling(20).mean()
feat['hl_range']   = (high - low) / close
feat['spy_ret']    = align_external(spy['Close'].squeeze().pct_change(),    kodex.index)
feat['usdkrw_ret'] = align_external(fx['Close'].squeeze().pct_change(),     kodex.index)
feat['vix_chg']    = align_external(vix['Close'].squeeze().pct_change(),    kodex.index)

FEATURE_COLS = list(feat.columns)
feat['target'] = (close.shift(-1) > close).astype(float)
feat = feat.dropna()

print(f"데이터: {feat.index[0].date()} ~ {feat.index[-1].date()}  ({len(feat)}일)\n")

# ── 학습 / 테스트 분리 ────────────────────────────────────────
train = feat[feat.index <= TRAIN_END]
test  = feat[feat.index >  TRAIN_END]

scaler     = StandardScaler()
scaled_all = np.zeros((len(feat), len(FEATURE_COLS)))
scaled_all[:len(train)] = scaler.fit_transform(train[FEATURE_COLS].values)
scaled_all[len(train):] = scaler.transform(test[FEATURE_COLS].values)

def make_dataset(X, y, ts):
    Xs = np.array([X[i:i+ts] for i in range(len(X)-ts)])
    Ys = np.array([y[i+ts]   for i in range(len(X)-ts)])
    return Xs, Ys

X_train, y_train = make_dataset(scaled_all[:len(train)], train['target'].values, TIME_STEP)

test_X_arr = scaled_all[len(train)-TIME_STEP:]
y_test     = test['target'].values
X_test     = np.array([test_X_arr[i:i+TIME_STEP] for i in range(len(y_test))])
num_feat   = X_train.shape[2]

# ── 3개 모델 학습 / 로드 ──────────────────────────────────────
all_probs = []

for i, seed in enumerate(SEEDS):
    weights_path = f"{WEIGHTS_DIR}/ensemble_{seed}.weights.h5"
    model = build_model(num_feat, seed)

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"[모델 {i+1}/3] 가중치 로드: seed={seed}")
    else:
        print(f"\n[모델 {i+1}/3] 학습 시작: seed={seed}")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        ]
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_split=0.1, callbacks=callbacks, verbose=1)
        model.save_weights(weights_path)
        print(f"[모델 {i+1}/3] 가중치 저장 완료\n")

    probs = model.predict(X_test, verbose=0).flatten()
    all_probs.append(probs)
    single_acc = ((probs >= 0.5).astype(int) == y_test.astype(int)).mean()
    print(f"  → 단일 모델 정확도: {single_acc:.1%}")

# ── 앙상블 예측 ───────────────────────────────────────────────
ensemble_probs = np.mean(all_probs, axis=0)
single_ref     = np.mean([((p >= 0.5).astype(int) == y_test.astype(int)).mean() for p in all_probs])
ensemble_dir   = (ensemble_probs >= 0.5).astype(int)
actual         = y_test.astype(int)
ensemble_acc   = (ensemble_dir == actual).mean()
test_dates     = test.index

print(f"\n{'='*55}")
print(f"  백테스트: {test.index[0].date()} ~ {test.index[-1].date()}  ({len(actual)}일)")
print(f"  단일 모델 평균 정확도: {single_ref:.1%}")
print(f"  앙상블 전체 정확도:   {ensemble_acc:.1%}  (개선: {(ensemble_acc-single_ref)*100:+.1f}%p)")
print(f"\n  ─── 신뢰도 임계값별 비교 (단일 모델 vs 앙상블) ───")
print(f"  {'임계값':<8} {'단일 정확도':<14} {'앙상블 정확도':<16} {'신호 수':<10} {'커버리지'}")
print(f"  {'-'*60}")

ref_probs = np.mean(all_probs, axis=0)  # same as ensemble
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    # 단일 모델 (첫 번째)
    s_mask = (all_probs[0] >= thresh) | (all_probs[0] <= 1-thresh)
    s_acc  = ((all_probs[0][s_mask] >= 0.5).astype(int) == actual[s_mask]).mean() if s_mask.sum() > 0 else 0

    # 앙상블
    e_mask = (ensemble_probs >= thresh) | (ensemble_probs <= 1-thresh)
    if e_mask.sum() == 0:
        continue
    e_acc     = (ensemble_dir[e_mask] == actual[e_mask]).mean()
    n         = e_mask.sum()
    coverage  = n / len(actual)
    print(f"  {thresh:.0%}↑      {s_acc:.1%}          {e_acc:.1%}            {n}일       {coverage:.0%}")

print(f"{'='*55}\n")

# ── 내일 예측 (앙상블) ────────────────────────────────────────
last_window = scaler.transform(feat[FEATURE_COLS].values[-TIME_STEP:]).reshape(1, TIME_STEP, num_feat)

tomorrow_probs = []
for i, seed in enumerate(SEEDS):
    weights_path = f"{WEIGHTS_DIR}/ensemble_{seed}.weights.h5"
    m = build_model(num_feat, seed)
    m.load_weights(weights_path)
    tomorrow_probs.append(float(m.predict(last_window, verbose=0)[0][0]))

ensemble_tomorrow = np.mean(tomorrow_probs)
today_close = float(close[feat.index[-1]])

next_day = today + timedelta(days=1)
while next_day.weekday() >= 5:
    next_day += timedelta(days=1)

if ensemble_tomorrow >= 0.65:
    direction, confidence = "📈 상승", "🔥 고신뢰"
elif ensemble_tomorrow >= 0.55:
    direction, confidence = "📈 상승", "✅ 보통"
elif ensemble_tomorrow <= 0.35:
    direction, confidence = "📉 하락", "🔥 고신뢰"
elif ensemble_tomorrow <= 0.45:
    direction, confidence = "📉 하락", "✅ 보통"
else:
    direction, confidence = "➖ 불확실", "⚠️ 관망 권장"

print(f"  오늘 종가:  {today_close:,.0f}원  ({feat.index[-1].date()})")
print(f"  개별 모델:  {' / '.join(f'{p:.1%}' for p in tomorrow_probs)}")
print(f"  앙상블 확률: {ensemble_tomorrow:.1%}  →  {direction}  {confidence}")
print(f"  예측 날짜:  {next_day}\n")

# ── 차트 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax1 = axes[0]
recent_close = close[close.index >= test.index[0]]
ax1.plot(recent_close.index, recent_close.values, color='steelblue', linewidth=1.2)
ax1.set_title(f"{SYMBOL_NAME} — 백테스트 기간 실제 종가")
ax1.set_ylabel("종가 (원)")

ax2 = axes[1]
for i, (p, seed) in enumerate(zip(all_probs, SEEDS)):
    ax2.plot(test_dates, p, linewidth=0.5, alpha=0.4, label=f"모델{i+1}(seed={seed})")
ax2.plot(test_dates, ensemble_probs, color='purple', linewidth=1.2, label="앙상블 평균")
ax2.axhline(0.65, color='green', linestyle='--', linewidth=1,   label='상승 고신뢰 (65%)')
ax2.axhline(0.35, color='red',   linestyle='--', linewidth=1,   label='하락 고신뢰 (35%)')
ax2.axhline(0.50, color='gray',  linestyle='-',  linewidth=0.5)
ax2.fill_between(test_dates, 0.65, 1.0, alpha=0.07, color='green')
ax2.fill_between(test_dates, 0.0, 0.35, alpha=0.07, color='red')
ax2.set_title("모델별 상승 확률 + 앙상블 평균")
ax2.set_ylabel("확률")
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right', fontsize=7)

ax3 = axes[2]
correct_single   = ((all_probs[0] >= 0.5).astype(int) == actual).astype(float)
correct_ensemble = (ensemble_dir == actual).astype(float)
roll_single   = pd.Series(correct_single,   index=test_dates).rolling(30).mean()
roll_ensemble = pd.Series(correct_ensemble, index=test_dates).rolling(30).mean()
ax3.plot(test_dates, roll_single,   color='gray',   linewidth=1,   linestyle='--', label=f"단일 모델 ({single_ref:.1%})")
ax3.plot(test_dates, roll_ensemble, color='purple', linewidth=1.5, label=f"앙상블 ({ensemble_acc:.1%})")
ax3.axhline(0.5, color='lightgray', linestyle='--', linewidth=0.8, label='기준선 (50%)')
ax3.set_title("방향 정확도 비교 — 단일 모델 vs 앙상블 (롤링 30일)")
ax3.set_ylabel("정확도")
ax3.set_ylim(0.2, 0.9)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig("lstm_test/result_ensemble.png", dpi=150)
print("📊 차트 저장 완료: lstm_test/result_ensemble.png")
