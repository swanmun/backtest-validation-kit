"""Real public prices for the examples: H4 candles from Binance's public market-data API (no key, no account).

The window is fixed, so every run sees the same bars. The first run downloads them (about 15 requests per symbol)
and caches them in examples/data/; later runs read the cache and need no network. Times are UTC.

[한국어] 예제용 실제 공개 시세: 바이낸스 공개 시세 API의 4시간 봉(키도 계정도 필요 없음).
구간을 고정해 두어서 언제 돌려도 같은 봉을 본다. 처음 한 번은 내려받아(종목마다 요청 약 15번) examples/data/에
저장하고, 그 뒤로는 저장본을 읽어 네트워크가 필요 없다. 시각은 UTC다.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://api.binance.com/api/v3/klines"
CACHE = Path(__file__).resolve().parent / "data"
START, END = "2020-01-01", "2026-07-01"  # fixed window, END excluded / 고정 구간, END는 포함하지 않음


def h4(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    """OHLC DataFrame of H4 bars (index = bar open time, UTC).

    [한국어] 4시간 봉 OHLC DataFrame(인덱스 = 봉 시작 시각, UTC).
    """
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{symbol}_4h_{start}_{end}.csv"
    if not f.exists():
        _download(symbol, start, end).to_csv(f)
    return pd.read_csv(f, index_col="time", parse_dates=["time"])


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    t = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    stop = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows = []
    while t < stop:
        q = f"{URL}?symbol={symbol}&interval=4h&startTime={t}&endTime={stop - 1}&limit=1000"
        req = urllib.request.Request(q, headers={"User-Agent": "backtest-validation-kit"})
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows += [(k[0], *map(float, k[1:5])) for k in batch]  # open time, open, high, low, close / 시작 시각, 시가, 고가, 저가, 종가
        t = batch[-1][0] + 1
        time.sleep(0.2)  # be polite to a free public API / 무료 공개 API이니 천천히
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.set_index("time")
