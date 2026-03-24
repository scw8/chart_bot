# ETF 목록 (이름: (티커, 통화))
ETF_LIST = {
    # 국내 ETF
    "TIGER 미국나스닥100": ("133690.KS", "KRW"),
    "TIGER 미국S&P500": ("360750.KS", "KRW"),
    "KODEX S&P500": ("379800.KS", "KRW"),

    # 해외 ETF - S&P500 계열
    "SPY": ("SPY", "USD"),
    "SPLG": ("SPLG", "USD"),
    "SPYG": ("SPYG", "USD"),
    "SSO": ("SSO", "USD"),      # S&P500 2배 레버리지

    # 해외 ETF - 나스닥 계열
    "QQQ": ("QQQ", "USD"),
    "QQQM": ("QQQM", "USD"),
    "QLD": ("QLD", "USD"),      # 나스닥 2배 레버리지
    "TQQQ": ("TQQQ", "USD"),    # 나스닥 3배 레버리지

    # 해외 ETF - 배당 계열
    "SCHD": ("SCHD", "USD"),
    "DGRW": ("DGRW", "USD"),

    # 기존 VOO
    "VOO": ("VOO", "USD"),
}

# 알림 조건
ALERT_CONFIG = {
    "ma20_cross": True,        # 20일선 하향 돌파
    "drop_from_high": -5.0,    # 고점 대비 5% 하락
    "sudden_change": 1.0,      # 30분 내 1% 급등락
}

# 스케줄 시간 (한국 시간 기준)
SCHEDULE_CONFIG = {
    "korea_open": "09:00",     # 한국장 시작 리포트
    "korea_close": "15:30",    # 한국장 마감 리포트
    "us_open": "23:30",        # 미국장 시작 리포트
}

# 장 운영 시간 (변동 체크할 시간대)
MARKET_HOURS = {
    "korea": ("09:00", "15:30"),
    "us": ("23:30", "06:00"),
}