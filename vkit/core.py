"""Shared pieces: trade lists, mark-to-market returns, and the Sharpe ratio.

A trade list is a DataFrame with one row per trade:

    symbol      market name (optional when there is only one market)
    side        +1 long, -1 short
    entry_time  bar time of the entry; the trade enters at that bar's close
    exit_time   bar time of the exit; the trade exits at that bar's close
    size        exposure as a fraction of equity (optional, default 1)
    cost        total cost of the trade as a fraction of equity (optional, default 0)

Other columns (pnl, entry_price, stop, ...) are kept but not used here.

Every trade, real or random, is marked to the bar closes in the same way,
so a comparison between them differs only in timing.

[한국어] 공통 부품: 거래 목록, 종가 기준 평가 수익, 샤프 비율.
거래 목록은 거래 하나에 한 줄인 DataFrame이다.
    symbol      종목 이름(종목이 하나면 없어도 됨)
    side        +1 매수, -1 매도
    entry_time  진입 봉 시각. 그 봉의 종가에 들어간다
    exit_time   청산 봉 시각. 그 봉의 종가에 나온다
    size        자본 대비 노출(없으면 1)
    cost        거래 하나의 총비용, 자본 대비(없으면 0)
다른 열(pnl, entry_price, stop 등)은 남겨 두지만 여기서는 쓰지 않는다.
실제 거래든 무작위 거래든 똑같이 봉 종가로 평가하므로, 둘의 차이는 타이밍뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ONE = "_"  # market name used when there is only one market and no symbol column / 종목이 하나이고 symbol 열이 없을 때 쓰는 이름


def sharpe(returns, periods_per_year: float | None = None) -> float:
    """Mean over standard deviation of per-period returns. Annualized if periods_per_year is given.

    [한국어] 기간별 수익의 평균 ÷ 표준편차. periods_per_year를 주면 연 환산한다.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    s = r.mean() / sd
    return float(s * np.sqrt(periods_per_year)) if periods_per_year else float(s)


def as_closes(prices) -> dict[str, pd.Series]:
    """Accept a close Series, an OHLC DataFrame, or a dict of either (symbol -> prices).

    [한국어] 종가 Series, OHLC DataFrame, 또는 그 둘의 dict(종목 → 가격)를 받아 종목별 종가로 바꾼다.
    """
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        prices = {ONE: prices}
    out = {}
    for name, p in prices.items():
        s = p["close"] if isinstance(p, pd.DataFrame) else p
        out[ONE if name is None else name] = s.astype(float).sort_index()
    return out


def bar_position(index: pd.DatetimeIndex, times) -> np.ndarray:
    """Position of the bar at or before each time.

    [한국어] 각 시각과 같거나 바로 앞인 봉의 위치(번호).
    """
    pos = index.searchsorted(pd.DatetimeIndex(times), side="right") - 1
    if (pos < 0).any():
        raise ValueError("a trade time is before the first bar")
    return pos


@dataclass
class Market:
    name: str
    ret: np.ndarray      # bar returns, close to close (first bar 0) / 봉 수익률, 종가 대 종가(첫 봉은 0)
    axis_pos: np.ndarray  # where each bar sits on the shared time axis / 공통 시간 축에서 각 봉의 자리
    entry: np.ndarray    # entry bar positions / 진입 봉 위치
    exit: np.ndarray     # exit bar positions / 청산 봉 위치
    weight: np.ndarray   # side * size / 방향 × 크기
    cost: np.ndarray     # cost per trade / 거래별 비용

    @property
    def n(self) -> int:
        return len(self.ret)

    def pnl(self, entry: np.ndarray | None = None) -> np.ndarray:
        """Per-bar P&L. With a new entry array, each trade keeps its holding length.

        [한국어] 봉별 손익. 새 진입 위치를 주면 거래마다 보유 기간은 그대로 두고 옮긴다.
        """
        e = self.entry if entry is None else entry
        x = e + (self.exit - self.entry)
        diff = np.zeros(self.n + 1)
        np.add.at(diff, e + 1, self.weight)  # in the market from the bar after entry ... / 진입 다음 봉부터 ...
        np.add.at(diff, x + 1, -self.weight)  # ... through the exit bar / ... 청산 봉까지 보유
        out = np.cumsum(diff)[: self.n] * self.ret
        np.subtract.at(out, x, self.cost)  # cost is paid at the exit bar / 비용은 청산 봉에서 낸다
        return out


class Book:
    """Trades turned into bar positions per market, on one shared time axis.

    Built once and reused: the random tests only move entry positions.
    freq groups bar P&L into periods (default one calendar day) before any Sharpe ratio.

    [한국어] 거래를 종목별 봉 위치로 바꿔 하나의 공통 시간 축 위에 둔다.
    한 번 만들어 계속 쓴다. 무작위 시험은 진입 위치만 옮긴다.
    freq는 샤프를 재기 전에 봉 손익을 묶는 기간이다(기본 하루).
    """

    def __init__(self, trades: pd.DataFrame, prices, freq: str | None = "1D"):
        closes = as_closes(prices)
        t = trades.copy()
        if "symbol" not in t:
            if len(closes) != 1:
                raise ValueError("trades need a 'symbol' column when there are several markets")
            t["symbol"] = next(iter(closes))
        missing = set(t["symbol"]) - set(closes)
        if missing:
            raise ValueError(f"no prices for {sorted(missing)}")

        indexes = [c.index for c in closes.values()]
        self.axis = indexes[0]
        for ix in indexes[1:]:
            self.axis = self.axis.union(ix)
        self.markets: list[Market] = []
        for name, close in closes.items():
            g = t[t["symbol"] == name]
            entry = bar_position(close.index, g["entry_time"])
            exit_ = bar_position(close.index, g["exit_time"])
            if (exit_ < entry).any():
                raise ValueError(f"{name}: a trade exits before it enters")
            side = g["side"].to_numpy(float)
            size = g["size"].to_numpy(float) if "size" in g else np.ones(len(g))
            cost = g["cost"].to_numpy(float) if "cost" in g else np.zeros(len(g))
            ret = close.pct_change().fillna(0.0).to_numpy()
            self.markets.append(Market(name, ret, self.axis.get_indexer(close.index),
                                       entry, exit_, side * size, cost))

        if freq:
            periods = self.axis.floor(freq)
            self.codes, self.period_index = pd.factorize(periods, sort=True)
            self.period_index = pd.DatetimeIndex(self.period_index)
        else:
            self.codes, self.period_index = np.arange(len(self.axis)), self.axis
        years = (self.period_index[-1] - self.period_index[0]).days / 365.25
        self.periods_per_year = len(self.period_index) / years if years > 0 else None

    def returns(self, entries: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Portfolio return per period. entries maps market name -> new entry positions.

        [한국어] 기간별 포트폴리오 수익. entries는 종목 이름 → 새 진입 위치.
        """
        total = np.zeros(len(self.axis))
        for m in self.markets:
            total[m.axis_pos] += m.pnl(None if entries is None else entries.get(m.name))
        return np.bincount(self.codes, weights=total, minlength=len(self.period_index))

    def sharpe(self, entries: dict[str, np.ndarray] | None = None) -> float:
        return sharpe(self.returns(entries), self.periods_per_year)


def mtm_returns(trades: pd.DataFrame, prices, freq: str | None = "1D") -> pd.Series:
    """Portfolio return per period from a trade list, marking open trades to the close.

    [한국어] 거래 목록에서 기간별 포트폴리오 수익을 낸다. 열린 거래는 종가로 평가한다.
    """
    book = Book(trades, prices, freq)
    return pd.Series(book.returns(), index=book.period_index, name="return")
