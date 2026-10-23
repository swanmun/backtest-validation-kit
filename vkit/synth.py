"""Synthetic prices for the examples and tests. No real market data is shipped.

random_walk   OHLC bars from a random walk, with optional drift regimes
mix_in_daily  replace the older part of a file with daily bars, the way some exports fill
              history they do not have at the requested timeframe (for the data_check demo)
add_spike     one bad tick

[한국어] 예제와 테스트용 합성 가격. 실제 시장 데이터는 넣지 않는다.
random_walk   무작위 행보로 만든 OHLC 봉(추세 구간을 넣을 수 있음)
mix_in_daily  파일의 오래된 구간을 일봉으로 바꾼다. 요청한 시간 단위의 과거가 없을 때 일부 내보내기가
              그렇게 채우는 것을 흉내 낸다(data_check 시연용)
add_spike     잘못된 틱 하나
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk(n_bars: int = 12000, freq: str = "4h", start: str = "2016-01-04", vol: float = 0.003,
                drift: float = 0.0, regimes: list[tuple[float, float]] | None = None, skip_weekends: bool = True,
                start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """regimes: [(share of bars, drift per bar), ...], e.g. [(0.5, 0), (0.5, 0.0004)] for a trend in the second half.

    [한국어] regimes: [(봉 비율, 봉당 추세), ...]. 예: [(0.5, 0), (0.5, 0.0004)]면 뒤 절반에 상승 추세.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(n_bars * 1.5) + 10, freq=freq)
    if skip_weekends:
        idx = idx[idx.dayofweek < 5]  # no Saturday or Sunday bars / 토 · 일 봉 없음
    idx = idx[:n_bars]
    n = len(idx)

    mu = np.full(n, drift)
    if regimes:
        edges = np.cumsum([0] + [round(s * n) for s, _ in regimes])
        for (lo, hi), (_, d) in zip(zip(edges[:-1], edges[1:]), regimes):
            mu[lo:hi] = d
    close = start_price * np.exp(np.cumsum(mu + vol * rng.standard_normal(n)))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.exp(np.abs(rng.standard_normal((2, n))) * vol * 0.5)  # random wicks above and below / 위아래 꼬리
    high = np.maximum(open_, close) * wick[0]
    low = np.minimum(open_, close) / wick[1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def mix_in_daily(ohlc: pd.DataFrame, until) -> pd.DataFrame:
    """Bars before `until` become daily bars stamped at midnight; the rest stays as it was.

    [한국어] until 전의 봉은 자정 시각의 일봉으로 바뀌고, 나머지는 그대로다.
    """
    old = ohlc[ohlc.index < pd.Timestamp(until)]
    daily = old.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return pd.concat([daily, ohlc[ohlc.index >= pd.Timestamp(until)]])


def add_spike(ohlc: pd.DataFrame, at: int, factor: float = 1.2) -> pd.DataFrame:
    """Multiply one bar's close (and stretch its range) by factor; the next bar opens where it would have.

    [한국어] 봉 하나의 종가에 factor를 곱한다(고가 · 저가도 늘린다). 다음 봉은 원래 자리에서 시작한다.
    """
    out = ohlc.copy()
    t = out.index[at]
    out.loc[t, "close"] *= factor
    out.loc[t, "high"] = max(out.loc[t, "high"], out.loc[t, "close"])
    out.loc[t, "low"] = min(out.loc[t, "low"], out.loc[t, "close"])
    return out
