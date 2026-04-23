"""
XGBoost 방향 예측 모듈 — stock_bot 연동용

흐름:
  1. yfinance로 해당 ETF + 외부 데이터(SPY/FX/VIX) 수집
  2. XGBoost 매일 재학습 (최신 데이터 반영)
  3. 학습된 모델을 Supabase Storage에 저장
  4. 다음 거래일 상승/하락 확률 예측
  5. 예측 로그를 Supabase predictions 테이블에 저장

외부 데이터는 1번만 다운로드해서 전 ETF에 재사용 (속도 최적화)
"""
import os
import io
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pytz

from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET       = "ml-models"
TRAIN_END    = "2023-12-31"
KST          = pytz.timezone("Asia/Seoul")


def _get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ── 기술적 지표 ───────────────────────────────────────────────
def _calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return 100 - (100 / (1 + g / l))

def _calc_macd_hist(s):
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()

def _calc_bb_pos(s, w=20):
    ma, std = s.rolling(w).mean(), s.rolling(w).std()
    return ((s - (ma - 2*std)) / (4*std + 1e-10)).clip(0, 1)


# ── 외부 데이터 다운로드 (1번만) ──────────────────────────────
def download_external_data():
    """SPY, USD/KRW, VIX 공통 데이터 1번만 다운로드"""
    start = "2013-01-01"
    end   = str(datetime.now(KST).date() + timedelta(days=1))

    spy = yf.download("SPY",      start=start, end=end, auto_adjust=True, progress=False)
    fx  = yf.download("USDKRW=X", start=start, end=end, auto_adjust=True, progress=False)
    vix = yf.download("^VIX",     start=start, end=end, auto_adjust=True, progress=False)

    for df in [spy, fx, vix]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

    return spy, fx, vix


# ── 피처 생성 ─────────────────────────────────────────────────
def _build_features(ticker, spy_df, fx_df, vix_df):
    """개별 ETF 피처 생성 (외부 데이터 재사용)"""
    start = "2013-01-01"
    end   = str(datetime.now(KST).date() + timedelta(days=1))

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    if len(raw) < 30:
        return None, None

    close  = raw['Close'].squeeze()
    volume = raw['Volume'].squeeze()
    high   = raw['High'].squeeze()
    low    = raw['Low'].squeeze()

    feat = pd.DataFrame(index=raw.index)
    feat['ret_1d']    = close.pct_change(1)
    feat['ret_5d']    = close.pct_change(5)
    feat['ret_10d']   = close.pct_change(10)
    feat['ret_20d']   = close.pct_change(20)
    feat['rsi']       = _calc_rsi(close)
    feat['macd_hist'] = _calc_macd_hist(close)
    feat['bb_pos']    = _calc_bb_pos(close)
    feat['vol_ratio'] = volume / volume.rolling(20).mean()
    feat['hl_range']  = (high - low) / close

    def align(series):
        return series.reindex(raw.index, method='ffill').shift(1)

    feat['spy_ret']    = align(spy_df['Close'].squeeze().pct_change())
    feat['usdkrw_ret'] = align(fx_df['Close'].squeeze().pct_change())
    feat['vix_chg']    = align(vix_df['Close'].squeeze().pct_change())

    for lag in [1, 2, 3]:
        feat[f'ret_1d_lag{lag}']  = feat['ret_1d'].shift(lag)
        feat[f'rsi_lag{lag}']     = feat['rsi'].shift(lag)
        feat[f'spy_ret_lag{lag}'] = feat['spy_ret'].shift(lag)
        feat[f'vix_chg_lag{lag}'] = feat['vix_chg'].shift(lag)

    feat['target'] = (close.shift(-1) > close).astype(int)
    feat = feat.dropna()

    return feat, close


# ── 모델 학습 ─────────────────────────────────────────────────
def _train(feat):
    feature_cols = [c for c in feat.columns if c != 'target']
    train = feat[feat.index <= TRAIN_END]

    if len(train) < 100:
        return None, None

    val_size = int(len(train) * 0.1)
    X_tr = train[feature_cols].values[:-val_size]
    y_tr = train['target'].values[:-val_size]
    X_val = train[feature_cols].values[-val_size:]
    y_val = train['target'].values[-val_size:]

    model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=4, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.1,
        eval_metric='logloss', early_stopping_rounds=50,
        random_state=42, verbosity=0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model, feature_cols


# ── Supabase 모델 저장/로드 ───────────────────────────────────
def _model_filename(ticker):
    return f"xgb_{ticker.replace('.', '_')}.pkl"

def _save_model(model, ticker):
    filename = _model_filename(ticker)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    try:
        sb = _get_supabase()
        try:
            sb.storage.from_(BUCKET).remove([filename])
        except Exception:
            pass
        sb.storage.from_(BUCKET).upload(
            filename, buf.read(),
            {"content-type": "application/octet-stream", "upsert": "true"}
        )
    except Exception as e:
        print(f"    모델 저장 실패 ({ticker}): {e}")


# ── 예측 결과 로그 저장 ───────────────────────────────────────
def _save_log(ticker, etf_name, predicted_date, direction, prob, market):
    try:
        sb = _get_supabase()
        sb.table("predictions").insert({
            "ticker":         ticker,
            "etf_name":       etf_name,
            "market":         market,
            "predicted_date": str(predicted_date),
            "direction":      direction,
            "probability":    round(float(prob), 4),
        }).execute()
    except Exception as e:
        print(f"    로그 저장 실패 ({ticker}): {e}")


# ── 방향/이모지 결정 ──────────────────────────────────────────
def _to_direction(prob):
    if prob >= 0.60:
        return "상승", "📈", "고신뢰"
    elif prob >= 0.55:
        return "상승", "📈", "보통"
    elif prob <= 0.40:
        return "하락", "📉", "고신뢰"
    elif prob <= 0.45:
        return "하락", "📉", "보통"
    else:
        return "불확실", "➖", "관망"


# ── 다음 거래일 계산 ──────────────────────────────────────────
def _next_trading_day():
    d = datetime.now(KST).date() + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ── 메인 예측 함수 ────────────────────────────────────────────
def predict_single(ticker, etf_name, spy_df, fx_df, vix_df, market="korea"):
    """ETF 1종 예측. 반환: {"direction", "emoji", "confidence", "prob"} 또는 None"""
    feat, close = _build_features(ticker, spy_df, fx_df, vix_df)
    if feat is None or len(feat) < 50:
        print(f"    [{etf_name}] 데이터 부족 — 스킵")
        return None

    model, feature_cols = _train(feat)
    if model is None:
        print(f"    [{etf_name}] 학습 데이터 부족 — 스킵")
        return None

    _save_model(model, ticker)

    prob = float(model.predict_proba(feat[feature_cols].values[-1:])[0, 1])
    direction, emoji, confidence = _to_direction(prob)

    predicted_date = _next_trading_day()
    _save_log(ticker, etf_name, predicted_date, direction, prob, market)

    print(f"    [{etf_name}] {emoji} {direction} {prob:.1%} ({confidence})")
    return {"direction": direction, "emoji": emoji, "confidence": confidence, "prob": prob}


def predict_market(etf_list, market="korea"):
    """
    여러 ETF 일괄 예측 (외부 데이터 1번만 다운로드)

    etf_list: {etf_name: (ticker, currency), ...}  — config.ETF_LIST 형식
    반환: {etf_name: {"direction", "emoji", "confidence", "prob"}}
    """
    print(f"  외부 데이터 다운로드 중...")
    spy_df, fx_df, vix_df = download_external_data()

    results = {}
    for etf_name, (ticker, _) in etf_list.items():
        try:
            result = predict_single(ticker, etf_name, spy_df, fx_df, vix_df, market)
            if result:
                results[etf_name] = result
        except Exception as e:
            print(f"    [{etf_name}] 예측 오류: {e}")

    return results
