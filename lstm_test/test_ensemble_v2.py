"""
Step 2 (개선): TIME_STEP 앙상블

단순 시드 앙상블의 실패 원인:
  → 같은 데이터를 다른 시드로 학습 = 서로 다른 패턴 학습 = 상쇄

구조적 앙상블:
  모델A (TIME_STEP=10): 단기 모멘텀 전문 (2주)
  모델B (TIME_STEP=20): 중단기 추세 (1달)
  모델C (TIME_STEP=40): 중기 추세 (2달)
  → 서로 다른 시간대를 보고 동의할 때 = 진짜 강한 신호
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

SYMBOL      = "379800.KS"
SYMBOL_NAME = "KODEX S&P500"
TIME_STEPS  = [10, 20, 40]   # 단기 / 중단기 / 중기
EPOCHS      = 100
BATCH_SIZE  = 32
TRAIN_END   = "2023-12-31"
WEIGHTS_DIR = "lstm_test/weights"
SEED        = 42

os.makedirs(WEIGHTS_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

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

def build_model(ts, num_feat):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(ts, num_feat)),
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
start = "2013-01-01"   # TIME_STEP=40 여유 확보
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
feat['spy_ret']    = align_external(spy['Close'].squeeze().pct_change(),  kodex.index)
feat['usdkrw_ret'] = align_external(fx['Close'].squeeze().pct_change(),   kodex.index)
feat['vix_chg']    = align_external(vix['Close'].squeeze().pct_change(),  kodex.index)

FEATURE_COLS     = list(feat.columns)
feat['target']   = (close.shift(-1) > close).astype(float)
feat             = feat.dropna()

train      = feat[feat.index <= TRAIN_END]
test       = feat[feat.index >  TRAIN_END]
y_test     = test['target'].values.astype(int)
test_dates = test.index

print(f"학습: {len(train)}일  |  테스트: {len(test)}일\n")

# ── 스케일링 (학습 데이터로만 fit) ────────────────────────────
scaler     = StandardScaler()
scaled_all = np.zeros((len(feat), len(FEATURE_COLS)))
scaled_all[:len(train)] = scaler.fit_transform(train[FEATURE_COLS].values)
scaled_all[len(train):] = scaler.transform(test[FEATURE_COLS].values)

num_feat = len(FEATURE_COLS)

# ── TIME_STEP별 학습 / 예측 ───────────────────────────────────
all_probs      = []
all_labels     = ['단기(10일)', '중단기(20일)', '중기(40일)']

for ts, label in zip(TIME_STEPS, all_labels):
    weights_path = f"{WEIGHTS_DIR}/ensemble_v2_ts{ts}.weights.h5"

    # 학습 데이터셋
    X_tr = np.array([scaled_all[i:i+ts] for i in range(len(train)-ts)])
    y_tr = np.array([train['target'].values[i+ts] for i in range(len(train)-ts)])

    # 테스트 데이터셋 (모든 테스트 날짜 커버)
    test_X_arr = scaled_all[len(train)-ts:]
    X_te = np.array([test_X_arr[i:i+ts] for i in range(len(y_test))])

    model = build_model(ts, num_feat)

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"[{label}] 가중치 로드 완료")
    else:
        print(f"\n[{label}] 학습 시작...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        ]
        model.fit(X_tr, y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_split=0.1, callbacks=callbacks, verbose=1)
        model.save_weights(weights_path)
        print(f"[{label}] 저장 완료")

    probs      = model.predict(X_te, verbose=0).flatten()
    single_acc = ((probs >= 0.5).astype(int) == y_test).mean()
    print(f"  → 단독 정확도: {single_acc:.1%}")
    all_probs.append(probs)

# ── 앙상블 결과 ───────────────────────────────────────────────
ensemble_probs = np.mean(all_probs, axis=0)
ensemble_dir   = (ensemble_probs >= 0.5).astype(int)
ensemble_acc   = (ensemble_dir == y_test).mean()
single_avg     = np.mean([((p >= 0.5).astype(int) == y_test).mean() for p in all_probs])

print(f"\n{'='*60}")
print(f"  백테스트: {test_dates[0].date()} ~ {test_dates[-1].date()}  ({len(y_test)}일)")
print(f"  단일 모델 평균 정확도:  {single_avg:.1%}")
print(f"  앙상블 전체 정확도:    {ensemble_acc:.1%}  ({(ensemble_acc-single_avg)*100:+.1f}%p)")
print(f"\n  ─── 신뢰도 임계값별 분석 ───────────────────────────")
print(f"  {'임계값':<8} {'단독(20일)':<14} {'앙상블':<14} {'신호 수':<10} {'커버리지'}")
print(f"  {'-'*58}")

ref_probs = all_probs[1]  # TIME_STEP=20 단독 모델 (기준)
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    s_mask = (ref_probs >= thresh) | (ref_probs <= 1-thresh)
    s_acc  = ((ref_probs[s_mask] >= 0.5).astype(int) == y_test[s_mask]).mean() if s_mask.sum() > 0 else 0

    e_mask = (ensemble_probs >= thresh) | (ensemble_probs <= 1-thresh)
    if e_mask.sum() == 0:
        print(f"  {thresh:.0%}↑      {s_acc:.1%}          -              0일        0%")
        continue
    e_acc    = (ensemble_dir[e_mask] == y_test[e_mask]).mean()
    n        = e_mask.sum()
    coverage = n / len(y_test)
    print(f"  {thresh:.0%}↑      {s_acc:.1%}          {e_acc:.1%}          {n}일       {coverage:.0%}")

print(f"{'='*60}\n")

# ── 내일 예측 (앙상블) ────────────────────────────────────────
tomorrow_probs = []
for ts, label in zip(TIME_STEPS, all_labels):
    weights_path = f"{WEIGHTS_DIR}/ensemble_v2_ts{ts}.weights.h5"
    m = build_model(ts, num_feat)
    m.load_weights(weights_path)
    last_window = scaler.transform(feat[FEATURE_COLS].values[-ts:]).reshape(1, ts, num_feat)
    p = float(m.predict(last_window, verbose=0)[0][0])
    tomorrow_probs.append(p)
    print(f"  {label}: {p:.1%}")

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

print(f"\n  오늘 종가:   {today_close:,.0f}원  ({feat.index[-1].date()})")
print(f"  앙상블 확률: {ensemble_tomorrow:.1%}  →  {direction}  {confidence}")
print(f"  예측 날짜:   {next_day}\n")

# ── 차트 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax1 = axes[0]
recent_close = close[close.index >= test_dates[0]]
ax1.plot(recent_close.index, recent_close.values, color='steelblue', linewidth=1.2)
ax1.set_title(f"{SYMBOL_NAME} — 백테스트 기간 실제 종가")
ax1.set_ylabel("종가 (원)")

ax2 = axes[1]
colors = ['#ff7f7f', '#7fbfff', '#7fff7f']
for probs, label, color in zip(all_probs, all_labels, colors):
    ax2.plot(test_dates, probs, linewidth=0.6, alpha=0.5, color=color, label=label)
ax2.plot(test_dates, ensemble_probs, color='purple', linewidth=1.4, label="앙상블 평균")
ax2.axhline(0.65, color='green', linestyle='--', linewidth=1,   label='65% 고신뢰선')
ax2.axhline(0.35, color='red',   linestyle='--', linewidth=1)
ax2.axhline(0.50, color='gray',  linestyle='-',  linewidth=0.5)
ax2.fill_between(test_dates, 0.65, 1.0, alpha=0.07, color='green')
ax2.fill_between(test_dates, 0.0, 0.35, alpha=0.07, color='red')
ax2.set_title("TIME_STEP별 상승 확률 + 앙상블 평균")
ax2.set_ylabel("확률")
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right', fontsize=8)

ax3 = axes[2]
correct_ref      = ((ref_probs >= 0.5).astype(int) == y_test).astype(float)
correct_ensemble = (ensemble_dir == y_test).astype(float)
roll_ref      = pd.Series(correct_ref,      index=test_dates).rolling(30).mean()
roll_ensemble = pd.Series(correct_ensemble, index=test_dates).rolling(30).mean()
ax3.plot(test_dates, roll_ref,      color='gray',   linewidth=1,   linestyle='--', label=f"단독 20일 ({single_avg:.1%})")
ax3.plot(test_dates, roll_ensemble, color='purple', linewidth=1.5, label=f"앙상블 ({ensemble_acc:.1%})")
ax3.axhline(0.5, color='lightgray', linestyle='--', linewidth=0.8, label='기준선 (50%)')
ax3.set_title("방향 정확도 비교 — 단독 vs TIME_STEP 앙상블 (롤링 30일)")
ax3.set_ylabel("정확도")
ax3.set_ylim(0.2, 0.9)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig("lstm_test/result_ensemble_v2.png", dpi=150)
print("📊 차트 저장 완료: lstm_test/result_ensemble_v2.png")
