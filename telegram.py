import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_message(message):
    """텔레그램 메시지 전송"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"  # 볼드, 이탤릭 등 서식 사용 가능
    }
    response = requests.post(url, data=data)
    return response.json()

def send_report(title, etf_data, exchange_rate):
    """전체 현황 리포트 전송"""
    from datetime import datetime
    import pytz

    KST = pytz.timezone("Asia/Seoul")
    now = datetime.now(KST).strftime("%Y년 %m월 %d일 %H:%M")

    message = f"<b>{title}</b>\n"
    message += f"🕐 {now} 기준\n"
    message += f"💵환율: {exchange_rate:,.0f}원\n\n"

    for name, data in etf_data.items():
        change_pct = data["change_pct"]

        # 등락 이모지
        if change_pct > 0:
            emoji = "🟥"
        elif change_pct < 0:
            emoji = "🟦"
        else:
            emoji = "➖"

        message += (
            f"{emoji} <b>{name}</b>\n"
            f"   현재가: {data['price_str']}\n"
            f"   20일선: {data['ma20_str']}\n"
            f"   고점대비: {data['drop_pct']:.1f}% | 전일대비: {change_pct:+.1f}%\n\n"
        )

    send_message(message)

def send_alerts(alerts):
    """변동 알림 전송"""
    if not alerts:
        return

    message = "🔔 <b>변동 알림 발생!</b>\n\n"
    message += "\n\n─────────────────\n\n".join(alerts)
    send_message(message)