"""
테스트 2 (개선버전 v2): 내일 종가 예측

개선사항:
  - StandardScaler: 이상치에 강한 정규화
  - 피처 12개: Open/High/Low/Close/Volume + MA비율/RSI/MACD/볼린저/수익률
  - 모델 가중치 저장/로드 (재실행 시 학습 생략)
  - Bidirectional LSTM 3레이어
  - Early Stopping + ReduceLROnPlateau
"""
import numpy as np
import pandas as pd
import os
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

SYMBOL        = "379800.KS"
SYMBOL_NAME   = "KODEX S&P500"
TIME_STEP     = 90
EPOCHS        = 100
BATCH_SIZE    = 32
CLOSE_IDX     = 0
WEIGHTS_PATH  = "lstm_test/weights/model_weights.weights.h5"

os.makedirs("lstm_test/weights", exist_ok=True)

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

# ── 날짜 계산 ─────────────────────────────────────────────────
KST      = pytz.timezone("Asia/Seoul")
today    = datetime.now(KST).date()
next_day = today + timedelta(days=1)
while next_day.weekday() >= 5:
    next_day += timedelta(days=1)

# ── 데이터 다운로드 ──────────────────────────────────────────
print(f"[{SYMBOL_NAME}] 데이터 다운로드 중...")
raw = yf.download(SYMBOL, start="2015-01-01",
                  end=str(today + timedelta(days=1)),
                  auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.droplevel(1)

features    = build_features(raw)
today_price = float(features['close'].iloc[-1])
last_date   = features.index[-1].date()

print(f"피처 수:    {features.shape[1]}개")
print(f"학습 기간: {features.index[0].date()} ~ {last_date}")
print(f"오늘 종가: {today_price:,.0f}원  |  예측 날짜: {next_day}\n")

# ── StandardScaler ────────────────────────────────────────────
scaler = StandardScaler()
scaled = scaler.fit_transform(features.values)

# ── 데이터셋 ─────────────────────────────────────────────────
def create_dataset(data, ts):
    X = [data[i:i+ts] for i in range(len(data)-ts)]
    Y = [data[i+ts, CLOSE_IDX] for i in range(len(data)-ts)]
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(scaled, TIME_STEP)
num_feat = X_train.shape[2]

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

# ── 가중치 로드 or 학습 ───────────────────────────────────────
if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print(f"✅ 저장된 가중치 로드 완료: {WEIGHTS_PATH}")
    print("   (재학습하려면 weights 파일을 삭제 후 실행)\n")
else:
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]
    print("모델 학습 중...")
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_split=0.1, callbacks=callbacks, verbose=1)
    model.save_weights(WEIGHTS_PATH)
    print(f"✅ 가중치 저장 완료: {WEIGHTS_PATH}\n")

    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
    ax_loss.plot(history.history['loss'],     label="학습 손실", color="steelblue")
    ax_loss.plot(history.history['val_loss'], label="검증 손실", color="orange")
    ax_loss.set_title("학습 손실 곡선")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    plt.tight_layout()
    plt.savefig("lstm_test/result_loss.png", dpi=150)
    plt.close(fig_loss)

# ── 예측 & 역변환 ─────────────────────────────────────────────
window      = scaled[-TIME_STEP:].reshape(1, TIME_STEP, num_feat)
pred_scaled = float(model.predict(window)[0][0])

pred_price = float(pred_scaled * scaler.scale_[CLOSE_IDX] + scaler.mean_[CLOSE_IDX])
change_pct = (pred_price - today_price) / today_price * 100

print(f"\n{'='*45}")
print(f"  오늘 종가:  {today_price:,.0f}원  ({last_date})")
print(f"  내일 예측:  {pred_price:,.0f}원  ({next_day})")
print(f"  예측 변동:  {change_pct:+.2f}%")
print(f"  예측 방향:  {'📈 상승' if change_pct > 0 else '📉 하락'}")
print(f"{'='*45}\n")

# ── 차트 ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
recent = features['close'].tail(90)
ax.plot(recent.index, recent.values, label="실제 종가", color="steelblue", linewidth=1.5)
ax.scatter([pd.Timestamp(next_day)], [pred_price], color="red", s=100, zorder=5,
           label=f"내일 예측 ({pred_price:,.0f}원 / {change_pct:+.2f}%)")
ax.axvline(x=features.index[-1], color="gray", linestyle="--", linewidth=0.8, label="오늘")
ax.set_title(f"{SYMBOL_NAME} — 내일({next_day}) 종가 예측")
ax.set_ylabel("종가 (원)")
ax.legend()
plt.tight_layout()
plt.savefig("lstm_test/result_tomorrow.png", dpi=150)
print("📊 차트 저장 완료: lstm_test/result_tomorrow.png")
