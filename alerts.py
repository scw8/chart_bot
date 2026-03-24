from config import ALERT_CONFIG

# 우선순위 상수
PRIORITY = {
    "강력": 1,
    "급락": 2,
    "공격적": 3,
    "매수고려": 4,
    "급등": 5,
}

def check_alerts(etf_data):
    """전체 ETF 변동 감지 후 알림 목록 반환"""
    alerts = []

    for name, data in etf_data.items():
        result = get_signal(name, data)
        if result:
            alerts.append(result)

    # 우선순위 기준 정렬
    alerts.sort(key=lambda x: x["priority"])

    # 카테고리별 분류해서 메시지 만들기
    message_parts = []

    강력 = [a for a in alerts if a["priority"] == 1]
    급락 = [a for a in alerts if a["priority"] == 2]
    공격적 = [a for a in alerts if a["priority"] == 3]
    매수고려 = [a for a in alerts if a["priority"] == 4]
    급등 = [a for a in alerts if a["priority"] == 5]

    if 강력:
        message_parts.append("🚨 <b>강력 매수 신호</b>\n" + "\n\n".join(a["message"] for a in 강력))
    if 급락:
        message_parts.append("🔻 <b>급락 감지</b>\n" + "\n\n".join(a["message"] for a in 급락))
    if 공격적:
        message_parts.append("⚡️ <b>공격적 매수 신호</b>\n" + "\n\n".join(a["message"] for a in 공격적))
    if 매수고려:
        message_parts.append("📉 <b>매수 고려</b>\n" + "\n\n".join(a["message"] for a in 매수고려))
    if 급등:
        message_parts.append("🔺 <b>급등 감지</b>\n" + "\n\n".join(a["message"] for a in 급등))

    return message_parts

def get_signal(name, data):
    """ETF 하나의 신호 감지"""
    latest_price = data["latest_price"]
    ma20 = data["ma20"]
    drop_pct = data["drop_pct"]
    change_pct = data["change_pct"]
    price_str = data["price_str"]
    ma20_str = data["ma20_str"]

    is_below_ma20 = latest_price < ma20
    is_drop_5pct = drop_pct <= ALERT_CONFIG["drop_from_high"]
    is_sudden_up = change_pct >= ALERT_CONFIG["sudden_change"]
    is_sudden_down = change_pct <= -ALERT_CONFIG["sudden_change"]

    signals = []
    priority = 99

    if is_below_ma20 and is_drop_5pct:
        signals.append(f"🚨 강력 매수 신호 (20일선 하향 + 고점대비 {drop_pct:.1f}% 하락)")
        priority = min(priority, PRIORITY["강력"])
    elif is_below_ma20:
        signals.append(f"📉 매수 고려 (20일선 하향 돌파, 고점대비 {drop_pct:.1f}%)")
        priority = min(priority, PRIORITY["매수고려"])
    elif is_drop_5pct:
        signals.append(f"⚡️ 공격적 매수 신호 (고점대비 {drop_pct:.1f}% 하락)")
        priority = min(priority, PRIORITY["공격적"])

    if is_sudden_up:
        signals.append(f"🔺 급등 감지 (+{change_pct:.1f}%)")
        priority = min(priority, PRIORITY["급등"])
    elif is_sudden_down:
        signals.append(f"🔻 급락 감지 ({change_pct:.1f}%)")
        priority = min(priority, PRIORITY["급락"])

    if not signals:
        return None

    message = (
        f"⚠️ <b>{name}</b>\n"
        f"   현재가: {price_str}\n"
        f"   20일선: {ma20_str}\n"
        f"   " + "\n   ".join(signals)
    )

    return {"priority": priority, "message": message}