"""Does the strategy use price patterns, or just the drift and volatility?

Build fake prices by shuffling the order of the bars. Each bar is kept as a whole
(open, high, low, close relative to the previous close), so the distribution of moves,
the overall drift, and the shape of each bar stay the same. What disappears is order:
trends that persist, pullbacks followed by rebounds, anything that depends on sequence.
Run the same strategy on many fake histories and see where the real result falls.

    res = shuffle_test(strategy, ohlc, n=200)
    res["p"]        # share of fake histories where the strategy did at least as well

strategy: callable(ohlc DataFrame) -> trade list (see core.py) for that one market.
block > 1 shuffles blocks of consecutive bars instead, keeping structure shorter than a block.

[한국어] 전략이 가격 패턴을 쓰는가, 그냥 상승과 변동성 덕인가?
봉의 순서를 섞어 가짜 가격을 만든다. 봉은 통째로 옮긴다(직전 종가 대비 시가 · 고가 · 저가 · 종가).
그래서 움직임의 분포, 전체 상승, 봉 하나의 모양은 그대로이고, 순서만 사라진다.
이어지는 추세, 눌림 뒤 반등처럼 순서에 기대는 것은 모두 없어진다.
같은 전략을 가짜 가격 여러 개에 돌려, 실제 결과가 그 사이 어디에 있는지 본다.
    res["p"]  전략이 실제 이상으로 한 가짜 가격의 비율
strategy: 가격(OHLC DataFrame)을 받아 그 종목의 거래 목록을 돌려주는 함수.
block > 1이면 이어진 봉 묶음 단위로 섞어, 묶음보다 짧은 구조는 남긴다.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .core import Book

FIELDS = ["open", "high", "low", "close"]


def shuffle_bars(ohlc: pd.DataFrame, rng: np.random.Generator, block: int = 1) -> pd.DataFrame:
    """Same bars in a random order, rebuilt into a continuous price path on the same index.

    [한국어] 같은 봉들을 무작위 순서로 놓고, 같은 시각 축 위에 이어진 가격 경로로 다시 만든다.
    """
    x = ohlc[FIELDS].to_numpy(float)
    prev = x[:-1, 3]
    rel = x[1:] / prev[:, None]  # each bar relative to the close before it / 봉마다 직전 종가 대비 비율
    starts = np.arange(0, len(rel), block)
    order = np.concatenate([np.arange(s, min(s + block, len(rel))) for s in rng.permutation(starts)])
    rel = rel[order]
    closes = x[0, 3] * np.cumprod(rel[:, 3])
    base = np.concatenate([[x[0, 3]], closes[:-1]])
    out = np.vstack([x[:1], rel * base[:, None]])
    return pd.DataFrame(out, index=ohlc.index, columns=FIELDS)


def shuffle_test(strategy: Callable[[pd.DataFrame], pd.DataFrame], ohlc: pd.DataFrame, n: int = 200,
                 seed: int = 0, freq: str | None = "1D", block: int = 1) -> dict:
    actual = _score(strategy, ohlc, freq)
    rng = np.random.default_rng(seed)
    fake = np.array([_score(strategy, shuffle_bars(ohlc, rng, block), freq) for _ in range(n)])
    return {"actual": actual, "fake": fake, "p": float((fake >= actual).mean()), "n": n}


def _score(strategy, ohlc: pd.DataFrame, freq) -> float:
    """Sharpe ratio of the strategy on these prices (0 if it makes no trades). / 이 가격에서 전략의 샤프(거래가 없으면 0)."""
    trades = strategy(ohlc)
    if len(trades) == 0:
        return 0.0
    return Book(trades, ohlc, freq).sharpe()
