from scheduler import run_report, run_alert_check
from data import get_all_etf_data
from alerts import check_alerts
from telegram import send_report, send_alerts
import sys

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
    mode = sys.argv[1] if len(sys.argv) > 1 else "alert"

    if mode == "test":
        test_run()
    elif mode == "korea_open":
        run_report("📊 한국장 시작 + 미국 마감 요약")
    elif mode == "korea_noon":
        run_report("📊 한국장 오후 장 현황")
    elif mode == "korea_close":
        run_report("📊 한국장 마감 요약")
    elif mode == "us_open":
        run_report("📊 미국장 시작")
    else:
        run_alert_check()