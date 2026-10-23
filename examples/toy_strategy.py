"""A deliberately simple strategy to feed the tools. Not a recommendation.

Moving-average breakout, one position at a time:
  long  when the close is above its moving average and breaks the highest high of the last `lookback` bars
  short when the close is below its moving average and breaks the lowest low (if both_sides)
  exit  when the close crosses back over the moving average
Entries and exits happen at the bar close. `stop` (the opposite end of the breakout range) is
recorded for the reconciliation example; this toy exits only on the average.

[한국어] 도구에 넣어 볼 일부러 단순한 전략. 추천하는 전략이 아니다.
이동평균 돌파, 포지션은 한 번에 하나:
  매수  종가가 이동평균 위이고, 최근 lookback개 봉의 최고가를 넘을 때
  매도  종가가 이동평균 아래이고, 최저가를 깰 때(both_sides일 때)
  청산  종가가 이동평균을 다시 건널 때
진입 · 청산은 봉 종가에서 한다. stop(돌파 구간의 반대쪽 끝)은 거래 대조 예제용으로만 기록하고,
이 전략은 이동평균으로만 청산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ma_breakout(ohlc: pd.DataFrame, ma: int = 50, lookback: int = 20, cost: float = 0.0005,
                both_sides: bool = True, symbol: str | None = None) -> pd.DataFrame:
    c = ohlc["close"].to_numpy(float)
    avg = ohlc["close"].rolling(ma).mean().to_numpy()
    hh = ohlc["high"].rolling(lookback).max().shift(1).to_numpy()  # highest high of the previous bars / 직전 봉들의 최고가
    ll = ohlc["low"].rolling(lookback).min().shift(1).to_numpy()   # lowest low of the previous bars / 직전 봉들의 최저가
    idx = ohlc.index

    trades, pos, start, stop = [], 0, 0, np.nan
    for i in range(max(ma, lookback + 1), len(c)):
        if pos == 0:
            if c[i] > avg[i] and c[i] > hh[i]:
                pos, start, stop = 1, i, ll[i]
            elif both_sides and c[i] < avg[i] and c[i] < ll[i]:
                pos, start, stop = -1, i, hh[i]
        elif (pos == 1 and c[i] < avg[i]) or (pos == -1 and c[i] > avg[i]) or i == len(c) - 1:
            trades.append((pos, idx[start], idx[i], c[start], c[i], stop))  # the last bar closes any open trade / 마지막 봉에서는 열린 거래를 닫는다
            pos = 0

    t = pd.DataFrame(trades, columns=["side", "entry_time", "exit_time", "entry_price", "exit_price", "stop"])
    t["size"] = 1.0
    t["cost"] = cost
    t["pnl"] = t["side"] * (t["exit_price"] / t["entry_price"] - 1) - t["cost"]
    if symbol is not None:
        t.insert(0, "symbol", symbol)
    return t
