import yfinance as yf
from config import ETF_LIST

def get_exchange_rate():
    """달러 환율 가져오기"""
    ticker = yf.Ticker("USDKRW=X")
    df = ticker.history(period="1d")
    return df["Close"].iloc[-1]

def get_etf_data(ticker):
    """ETF 60일치 데이터 가져오기"""
    df = yf.Ticker(ticker).history(period="60d")

     # NaN 값 제거
    df = df.dropna()

    return df

def get_all_etf_data():

    """전체 ETF 데이터 딕셔너리로 반환"""
    result = {}
    exchange_rate = get_exchange_rate()

    for name, (ticker, currency) in ETF_LIST.items():
        df = get_etf_data(ticker)

        # 데이터 유효성 체크
        if df.empty or len(df) < 20:
            print(f"{name} ({ticker}) 데이터 부족 — 스킵")
            continue

        latest_price = df["Close"].iloc[-1]   # 현재가
        high_30d = df["Close"].tail(30).max() # 30일 고점
        ma20 = df["Close"].tail(20).mean()    # 20일 이동평균
        prev_price = df["Close"].iloc[-2]     # 직전 종가 (급등락 비교용)

        # 고점 대비 하락률
        drop_pct = (latest_price - high_30d) / high_30d * 100

        # 직전 대비 변동률 (급등락 감지용)
        change_pct = (latest_price - prev_price) / prev_price * 100

        # 원화 환산
        if currency == "KRW":
            price_str = f"{latest_price:,.0f}원"
            ma20_str = f"{ma20:,.0f}원"
        else:
            krw_price = latest_price * exchange_rate
            price_str = f"${latest_price:,.2f} (약 {krw_price:,.0f}원)"
            ma20_str = f"${ma20:,.2f}"

        result[name] = {
            "ticker": ticker,
            "currency": currency,
            "latest_price": latest_price,
            "ma20": ma20,
            "drop_pct": drop_pct,
            "change_pct": change_pct,
            "price_str": price_str,
            "ma20_str": ma20_str,
        }

    return result, exchange_rate