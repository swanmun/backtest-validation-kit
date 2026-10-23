"""Does the entry timing add anything?

Keep everything about the trades except when they start: each market keeps its
number of trades, each trade keeps its side, size, cost, and holding length, and only
the entry bar is drawn at random. Repeat n times. If random timing does as well as
the strategy often enough, the strategy's timing is indistinguishable from chance.

    res = random_entry_test(trades, prices, n=1000)
    res["p"]        # share of random runs with a Sharpe ratio at least the strategy's

Random runs keep the long/short mix, so a strategy that is long in a rising market
shares that drift with its random copies; what is left to beat is the timing.
Random trades may overlap each other.

allowed (optional): bool Series per market (same bars as its prices). Random entries are
drawn only where it is True, e.g. "close above the 200-bar average", to ask whether the
timing adds anything beyond a simple trend filter.

[한국어] 진입 타이밍이 뭔가를 더하는가?
거래에서 시작 시점만 빼고 모두 그대로 둔다. 종목마다 거래 수, 거래마다 방향 · 크기 · 비용 · 보유 기간은 같고,
진입 봉만 무작위로 뽑는다. 이를 n번 되풀이한다. 무작위 타이밍이 전략만큼 하는 경우가 충분히 많으면,
전략의 타이밍은 우연과 구분되지 않는다.
    res["p"]  샤프가 전략 이상인 무작위 실행의 비율
무작위 실행은 매수 · 매도 비율을 그대로 두므로, 오르는 시장에서 매수한 전략은 그 상승을 무작위 복사본과 나눈다.
남는 것은 타이밍뿐이다. 무작위 거래끼리는 겹칠 수 있다.
allowed(선택): 종목별 bool Series(가격과 같은 봉). True인 봉에서만 무작위 진입을 뽑는다.
예를 들어 '종가가 200봉 평균 위'로 두면, 단순 추세 필터를 넘어서 타이밍이 더하는 것이 있는지 묻는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import ONE, Book


def random_entry_test(trades: pd.DataFrame, prices, n: int = 1000, seed: int = 0,
                      freq: str | None = "1D", allowed=None) -> dict:
    book = Book(trades, prices, freq)
    actual = book.sharpe()
    rng = np.random.default_rng(seed)
    pools = _pools(book, prices, allowed)

    sims = np.empty(n)
    for i in range(n):
        entries = {}
        for m in book.markets:
            if len(m.entry) == 0:
                continue
            last = m.n - 1 - (m.exit - m.entry)  # latest entry that keeps the exit inside the data / 청산이 데이터 안에 남는 가장 늦은 진입
            pool = pools.get(m.name)
            if pool is None:
                entries[m.name] = rng.integers(0, last + 1)
            else:
                k = np.searchsorted(pool, last, side="right")
                if (k == 0).any():
                    raise ValueError(f"{m.name}: no allowed bar early enough for a trade")
                entries[m.name] = pool[(rng.random(len(k)) * k).astype(int)]
        sims[i] = book.sharpe(entries)

    return {"actual": actual, "random": sims, "p": float((sims >= actual).mean()), "n": n,
            "periods_per_year": book.periods_per_year}


def _pools(book: Book, prices, allowed) -> dict[str, np.ndarray]:
    """Bar positions where a random entry may be drawn, per market. / 종목별로 무작위 진입을 뽑을 수 있는 봉 위치."""
    if allowed is None:
        return {}
    if isinstance(allowed, pd.Series):
        allowed = {ONE: allowed}
    out = {}
    for m in book.markets:
        mask = allowed.get(m.name)
        if mask is not None:
            out[m.name] = np.flatnonzero(mask.to_numpy(bool))
    return out
