"""
테스트 1 (개선버전 v2): 어제 종가 예측 vs 실제 비교

개선사항:
  - StandardScaler: 이상치에 강한 정규화
  - 피처 12개: Open/High/Low/Close/Volume + MA비율/RSI/MACD/볼린저/수익률
  - Bidirectional LSTM 3레이어
  - Early Stopping + ReduceLROnPlateau
"""
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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta
import pytz

SYMBOL      = "379800.KS"
SYMBOL_NAME = "KODEX S&P500"
TIME_STEP   = 90
EPOCHS      = 100
BATCH_SIZE  = 32
CLOSE_IDX   = 0   # features 배열에서 종가 컬럼 인덱스

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

def build_features(df):
    c = df['Close'].squeeze()
    feat = pd.DataFrame(index=df.index)
    feat['close']        = c
    feat['open']         = df['Open'].squeeze()
    feat['high']         = df['High'].squeeze()
    feat['low']          = df['Low'].squeeze()
    feat['volume']       = df['Volume'].squeeze()
    feat['ma5_ratio']    = c / c.rolling(5).mean()
    feat['ma20_ratio']   = c / c.rolling(20).mean()
    feat['ma60_ratio']   = c / c.rolling(60).mean()
    feat['rsi']          = calc_rsi(c)
    feat['macd_hist']    = calc_macd_hist(c)
    feat['bb_pos']       = calc_bb_pos(c)
    feat['daily_return'] = c.pct_change()
    return feat.dropna()

# ── 데이터 다운로드 ──────────────────────────────────────────
KST   = pytz.timezone("Asia/Seoul")
today = datetime.now(KST).date()

print(f"[{SYMBOL_NAME}] 데이터 다운로드 중...")
raw = yf.download(SYMBOL, start="2015-01-01",
                  end=str(today + timedelta(days=1)),
                  auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.droplevel(1)

features     = build_features(raw)
target_date  = features.index[-1].date()
actual_price = float(features['close'].iloc[-1])
prev_price   = float(features['close'].iloc[-2])
train_feat   = features.iloc[:-1]

print(f"피처 수:    {features.shape[1]}개")
print(f"학습 기간: {train_feat.index[0].date()} ~ {train_feat.index[-1].date()}")
print(f"예측 날짜: {target_date}  |  실제 종가: {actual_price:,.0f}원\n")

# ── StandardScaler (학습 데이터로만 fit → 미래 데이터 누수 방지) ──
scaler      = StandardScaler()
scaled_train = scaler.fit_transform(train_feat.values)
scaled_all   = scaler.transform(features.values)   # 역변환용

# ── 데이터셋 ─────────────────────────────────────────────────
def create_dataset(data, ts):
    X = [data[i:i+ts] for i in range(len(data)-ts)]
    Y = [data[i+ts, CLOSE_IDX] for i in range(len(data)-ts)]
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(scaled_train, TIME_STEP)
num_feat = X_train.shape[2]
print(f"학습 샘플: {X_train.shape[0]}  |  피처: {num_feat}  |  윈도우: {TIME_STEP}일\n")

# ── 모델 ─────────────────────────────────────────────────────
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(TIME_STEP, num_feat)),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='huber')
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
]
print("\n모델 학습 중...")
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=0.1, callbacks=callbacks, verbose=1)

# ── 예측 & 역변환 (StandardScaler 역변환 기법) ─────────────────
window     = scaled_all[-(TIME_STEP+1):-1].reshape(1, TIME_STEP, num_feat)
pred_scaled = float(model.predict(window)[0][0])

pred_price = float(pred_scaled * scaler.scale_[CLOSE_IDX] + scaler.mean_[CLOSE_IDX])

error_pct        = abs(pred_price - actual_price) / actual_price * 100
direction_actual = "상승" if actual_price > prev_price else "하락"
direction_pred   = "상승" if pred_price  > prev_price else "하락"
direction_ok     = direction_actual == direction_pred

print(f"\n{'='*45}")
print(f"  예측 종가:  {pred_price:,.0f}원")
print(f"  실제 종가:  {actual_price:,.0f}원")
print(f"  오차:       {abs(pred_price - actual_price):,.0f}원")
print(f"  오차율:     {error_pct:.2f}%")
print(f"  방향 예측:  {'✅ 맞음' if direction_ok else '❌ 틀림'} (예측: {direction_pred} / 실제: {direction_actual})")
print(f"{'='*45}\n")

# ── 차트 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

ax1 = axes[0]
recent = features['close'].tail(90)
ax1.plot(recent.index, recent.values, label="실제 종가", color="steelblue", linewidth=1.5)
ax1.scatter([features.index[-1]], [actual_price], color="green", s=80, zorder=5,
            label=f"실제 ({actual_price:,.0f}원)")
ax1.scatter([features.index[-1]], [pred_price], color="red", marker="x", s=120,
            zorder=5, linewidths=2, label=f"예측 ({pred_price:,.0f}원) | 오차 {error_pct:.2f}%")
ax1.set_title(f"{SYMBOL_NAME} — 예측 정확도 테스트 ({target_date})")
ax1.set_ylabel("종가 (원)")
ax1.legend()

ax2 = axes[1]
ax2.plot(history.history['loss'],     label="학습 손실", color="steelblue")
ax2.plot(history.history['val_loss'], label="검증 손실", color="orange")
ax2.set_title("학습 손실 곡선")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()
plt.savefig("lstm_test/result_accuracy.png", dpi=150)
print("📊 차트 저장 완료: lstm_test/result_accuracy.png")
