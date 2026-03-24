from datetime import datetime
import pytz

from data import get_all_etf_data
from alerts import check_alerts
from telegram import send_report, send_alerts
from config import MARKET_HOURS

KST = pytz.timezone("Asia/Seoul")

def is_market_open(market):
    """현재 장이 열려있는지 확인"""
    now = datetime.now(KST).strftime("%H:%M")
    start, end = MARKET_HOURS[market]

    if market == "korea":
        return start <= now <= end
    else:
        return now >= start or now <= end

def run_report(title):
    """리포트 실행"""
    print(f"[{datetime.now(KST).strftime('%H:%M')}] {title} 실행 중...")
    etf_data, exchange_rate = get_all_etf_data()
    send_report(title, etf_data, exchange_rate)

def run_alert_check():
    """변동 감지 실행 (장 중에만)"""
    now = datetime.now(KST).strftime("%H:%M")

    korea_open = is_market_open("korea")
    us_open = is_market_open("us")

    if not korea_open and not us_open:
        print(f"[{now}] 장 외 시간 — 변동 체크 스킵")
        return

    market = "한국장" if korea_open else "미국장"
    print(f"[{now}] {market} 변동 체크 중...")

    etf_data, _ = get_all_etf_data()
    alerts = check_alerts(etf_data)

    if alerts:
        send_alerts(alerts)
    else:
        print(f"[{now}] 변동 없음")

# import schedule
# import time
# from datetime import datetime
# import pytz

# from data import get_all_etf_data
# from alerts import check_alerts
# from telegram import send_report, send_alerts
# from config import MARKET_HOURS

# KST = pytz.timezone("Asia/Seoul")

# def is_market_open(market):
#     """현재 장이 열려있는지 확인"""
#     now = datetime.now(KST).strftime("%H:%M")
#     start, end = MARKET_HOURS[market]

#     if market == "korea":
#         return start <= now <= end
#     else:
#         # 미국장은 자정을 넘기기 때문에 별도 처리
#         return now >= start or now <= end

# def run_report(title):
#     """리포트 실행"""
#     print(f"[{datetime.now(KST).strftime('%H:%M')}] {title} 실행 중...")
#     etf_data, exchange_rate = get_all_etf_data()
#     send_report(title, etf_data, exchange_rate)

# def run_alert_check():
#     """변동 감지 실행 (장 중에만)"""
#     now = datetime.now(KST).strftime("%H:%M")

#     korea_open = is_market_open("korea")
#     us_open = is_market_open("us")

#     if not korea_open and not us_open:
#         print(f"[{now}] 장 외 시간 — 변동 체크 스킵")
#         return

#     market = "한국장" if korea_open else "미국장"
#     print(f"[{now}] {market} 변동 체크 중...")

#     etf_data, _ = get_all_etf_data()
#     alerts = check_alerts(etf_data)

#     if alerts:
#         send_alerts(alerts)
#     else:
#         print(f"[{now}] 변동 없음")

# def start():
#     """스케줄 등록 및 실행"""
#     print("ETF 알림 봇 시작!")

#     # 정기 리포트
#     schedule.every().day.at("09:00").do(
#         run_report, title="📊 한국장 시작 + 미국 마감 요약"
#     )
#     schedule.every().day.at("15:30").do(
#         run_report, title="📊 한국장 마감 요약"
#     )
#     schedule.every().day.at("23:30").do(
#         run_report, title="📊 미국장 시작"
#     )

#     # 30분마다 변동 체크
#     schedule.every(30).minutes.do(run_alert_check)

#     # 스케줄 계속 실행
#     while True:
#         schedule.run_pending()
#         time.sleep(60)  # 1분마다 스케줄 확인