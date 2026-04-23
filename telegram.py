import requests
import os
from dotenv import load_dotenv
from config import ALERT_CONFIG

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    response = requests.post(url, data=data)
    return response.json()

def _etf_line(name, data, signal_desc=None, ai_pred=None):
    change_pct = data["change_pct"]
    emoji = "🟥" if change_pct > 0 else ("🟦" if change_pct < 0 else "➖")
    line = (
        f"{emoji} <b>{name}</b>\n"
        f"   현재가: {data['price_str']} | 20일선: {data['ma20_str']}\n"
        f"   고점대비: {data['drop_pct']:.1f}% | 전일대비: {change_pct:+.1f}%"
    )
    if signal_desc:
        line += f"\n   → {signal_desc}"
    if ai_pred:
        line += f"\n   🤖 내일: {ai_pred['emoji']} {ai_pred['direction']} {ai_pred['prob']:.1%}"
    return line

def _classify(data):
    """ETF를 신호 카테고리로 분류"""
    latest_price = data["latest_price"]
    ma20 = data["ma20"]
    drop_pct = data["drop_pct"]
    change_pct = data["change_pct"]

    is_below_ma20 = latest_price < ma20
    is_drop_5pct = drop_pct <= ALERT_CONFIG["drop_from_high"]
    is_sudden_up = change_pct >= ALERT_CONFIG["sudden_change"]
    is_sudden_down = change_pct <= -ALERT_CONFIG["sudden_change"]

    if is_below_ma20 and is_drop_5pct:
        return "강한매수", f"20일선 하향 + 고점대비 {drop_pct:.1f}% 하락"
    elif is_sudden_down:
        return "급락", None
    elif is_below_ma20:
        return "매수고려", f"20일선 하향 (고점대비 {drop_pct:.1f}%)"
    elif is_sudden_up:
        return "급등", None
    else:
        return "기타", None

def _build_market_section(etf_data, ai_predictions=None):
    """장 섹션 메시지 구성: 기타 → 강한매수 → 매수고려 → 급락 → 급등"""
    groups = {"기타": [], "강한매수": [], "매수고려": [], "급락": [], "급등": []}
    preds = ai_predictions or {}

    for name, data in etf_data.items():
        category, desc = _classify(data)
        groups[category].append(_etf_line(name, data, desc, ai_pred=preds.get(name)))

    parts = []
    if groups["기타"]:
        parts.append("\n\n".join(groups["기타"]))
    if groups["강한매수"]:
        parts.append("🚨 <b>강한 매수 타이밍 (고점대비 -5% 이상)</b>\n" + "\n\n".join(groups["강한매수"]))
    if groups["매수고려"]:
        parts.append("📉 <b>매수 고려</b>\n" + "\n\n".join(groups["매수고려"]))
    if groups["급락"]:
        parts.append("🔻 <b>급락 감지</b>\n" + "\n\n".join(groups["급락"]))
    if groups["급등"]:
        parts.append("🔺 <b>급등 감지</b>\n" + "\n\n".join(groups["급등"]))

    return "\n\n".join(parts)

def send_report(title, etf_data, exchange_rate, mode="korea_open", ai_predictions=None):
    from datetime import datetime
    import pytz

    KST = pytz.timezone("Asia/Seoul")
    now = datetime.now(KST).strftime("%Y년 %m월 %d일 %H:%M")

    korea_data = {n: d for n, d in etf_data.items() if d["currency"] == "KRW"}
    us_data    = {n: d for n, d in etf_data.items() if d["currency"] == "USD"}
    preds      = ai_predictions or {}

    if mode in ("us_open", "us_close"):
        market_order = [("🇺🇸 미국장", us_data, preds), ("🇰🇷 한국장", korea_data, {})]
    else:
        market_order = [("🇰🇷 한국장", korea_data, preds), ("🇺🇸 미국장", us_data, {})]

    message = f"<b>{title}</b>\n"
    message += f"🕐 {now} 기준\n"
    message += f"💵 환율: {exchange_rate:,.0f}원\n"

    for label, data, market_preds in market_order:
        if not data:
            continue
        message += f"\n━━━ {label} ━━━\n\n"
        message += _build_market_section(data, market_preds)

    send_message(message)

def send_alerts(alerts, market=None):
    if not alerts:
        return

    if market == "korea":
        header = "🔔 <b>변동 알림</b>\n\n━━━ 🇰🇷 한국장 ━━━\n\n"
    elif market == "us":
        header = "🔔 <b>변동 알림</b>\n\n━━━ 🇺🇸 미국장 ━━━\n\n"
    else:
        header = "🔔 <b>변동 알림</b>\n\n"

    message = header + "\n\n─────────────────\n\n".join(alerts)
    send_message(message)
