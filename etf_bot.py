import yfinance as yf
import requests
from dotenv import load_dotenv
import os

# .env 불러오기
load_dotenv()

# 환경변수에서 가져오기
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=data)

ETF_LIST = {
    "TIGER 미국나스닥100": ("133690.KS", "KRW"),
    "TIGER 미국S&P500": ("360750.KS", "KRW"),
    "KODEX S&P500": ("379800.KS", "KRW"),
    "SPY": ("SPY", "USD"),
    "QQQ": ("QQQ", "USD"),
    "VOO": ("VOO", "USD"),
}

# 환율 가져오기
usd_krw = yf.Ticker("USDKRW=X")
exchange_rate = usd_krw.history(period="1d")["Close"].iloc[-1]
print(f"💱 현재 환율: {exchange_rate:,.0f}원\n")

alerts = []  # 알림 메시지 모아두는 리스트

for name, (ticker, currency) in ETF_LIST.items():
    df = yf.Ticker(ticker).history(period="60d")  # 60일로 늘림
    
    latest_price = df["Close"].iloc[-1]      # 현재가
    high_30d = df["Close"].tail(30).max()    # 최근 30일 고점
    ma20 = df["Close"].tail(20).mean()       # 20일 이동평균

    # 고점 대비 하락률 계산
    drop_pct = (latest_price - high_30d) / high_30d * 100

    # 가격 문자열 만들기
    if currency == "KRW":
        price_str = f"{latest_price:,.0f}원"
        ma20_str = f"{ma20:,.0f}원"
    else:
        krw_price = latest_price * exchange_rate
        price_str = f"${latest_price:,.2f} (약 {krw_price:,.0f}원)"
        ma20_str = f"${ma20:,.2f}"

    # 조건 체크
    is_below_ma20 = latest_price < ma20
    is_drop_5pct = drop_pct <= -5.0

    # 신호 판단
    if is_below_ma20 and is_drop_5pct:
        signal = f"🚨 강력 매수 신호 (20일선 하향 + 고점대비 {drop_pct:.1f}% 하락)"
    elif is_below_ma20:
        signal = f"📉 매수 고려 (20일선 하향 돌파, 고점대비 {drop_pct:.1f}%)"
    elif is_drop_5pct:
        signal = f"🚨 공격적 매수 신호 (고점대비 {drop_pct:.1f}% 하락)"
    else:
        signal = None

    if signal:
        alerts.append(
            f"⚠️ {name}\n"
            f"   현재가: {price_str}\n"
            f"   20일선: {ma20_str}\n"
            f"   {signal}"
        )
    else:
        print(f"✅ {name}: {price_str} | 20일선: {ma20_str} | 고점대비: {drop_pct:.1f}%")

# 알림 출력
if alerts:
    alert_message = "🔔 알림 발생!\n\n" + "\n\n".join(alerts)
    send_telegram(alert_message)
    print(alert_message)  # 터미널에도 출력
else:
    normal_message = f"💱 환율: {exchange_rate:,.0f}원\n\n👍 모든 ETF 정상 범위입니다."
    send_telegram(normal_message)
    print(normal_message)