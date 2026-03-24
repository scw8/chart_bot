from scheduler import run_report, run_alert_check, is_market_open
from data import get_all_etf_data
from alerts import check_alerts
from telegram import send_report, send_alerts
from datetime import datetime
import pytz
import sys

KST = pytz.timezone("Asia/Seoul")

def test_run():
    """테스트용 즉시 실행"""
    print("테스트 실행 중...")
    etf_data, exchange_rate = get_all_etf_data()
    send_report("📊 ETF 현황 테스트", etf_data, exchange_rate)
    alerts = check_alerts(etf_data)
    if alerts:
        send_alerts(alerts)
    else:
        print("변동 없음")
    print("테스트 완료!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_run()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        # 정기 리포트 실행
        now = datetime.now(KST).strftime("%H:%M")
        if "09:00" <= now < "09:30":
            run_report("📊 한국장 시작 + 미국 마감 요약")
        elif "15:30" <= now < "16:00":
            run_report("📊 한국장 마감 요약")
        elif now >= "23:30" or now < "00:00":
            run_report("📊 미국장 시작")
    else:
        # 변동 체크 실행
        run_alert_check()

# from scheduler import start
# from data import get_all_etf_data
# from alerts import check_alerts
# from telegram import send_report, send_alerts

# def test_run():
#     """스케줄 없이 즉시 한 번 실행 (테스트용)"""
#     print("테스트 실행 중...")
#     etf_data, exchange_rate = get_all_etf_data()

#     # 리포트 전송
#     send_report("📊 ETF 현황 테스트", etf_data, exchange_rate)

#     # 변동 체크
#     alerts = check_alerts(etf_data)
#     if alerts:
#         send_alerts(alerts)
#     else:
#         print("변동 없음")

#     print("테스트 완료!")

# if __name__ == "__main__":
#     import sys

#     # 인자 없이 실행하면 테스트
#     # python main.py test → 테스트 실행
#     # python main.py → 스케줄 실행
#     if len(sys.argv) > 1 and sys.argv[1] == "test":
#         test_run()
#     else:
#         start()