"""Is a filter picking better trades, or just fewer of them?

In a strategy whose average trade loses money after costs, trading less loses less,
whatever you remove. So a filter that skips trades (after a losing streak, in low
volatility, on certain days) can look like an improvement without selecting anything.
Compare it with removing the same number of trades at random.

    res = removal_test(trades, keep=mask, n=1000)
    res["p"]                # share of random removals that did at least as well as the filter
    res["from_fewer"]       # improvement you get from trading less alone (random removal, on average)
    res["from_selection"]   # what the filter adds beyond that

keep: boolean mask over the trade list, True for trades the filter kept.
metric: "sum" (total P&L), "mean" (average trade), or callable(kept trades DataFrame) -> float,
        for metrics that depend on order such as drawdown. Trades are passed in their original order.

[한국어] 필터가 좋은 거래를 고르는가, 거래를 줄일 뿐인가?
비용까지 치르면 평균 거래가 손실인 전략에서는, 무엇을 빼든 적게 거래할수록 덜 잃는다.
그래서 거래를 건너뛰는 필터(연패 뒤, 변동성이 낮을 때, 특정 요일 등)는 아무것도 고르지 않고도 개선처럼 보일 수 있다.
같은 수의 거래를 무작위로 뺀 결과와 비교한다.
    res["p"]               무작위 제거가 필터 이상으로 한 비율
    res["from_fewer"]      덜 거래한 것만으로 얻는 개선(무작위 제거의 평균)
    res["from_selection"]  필터가 그 위에 더한 몫
keep: 거래 목록 위의 bool 마스크, 필터가 남긴 거래가 True.
metric: "sum"(총손익), "mean"(평균 거래), 또는 남긴 거래 DataFrame을 받아 값을 내는 함수
        (낙폭처럼 순서에 기대는 지표용. 거래는 원래 순서대로 넘긴다).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def removal_test(trades: pd.DataFrame, keep, n: int = 1000, seed: int = 0,
                 metric: str | Callable[[pd.DataFrame], float] = "sum", pnl: str = "pnl") -> dict:
    keep = np.asarray(keep, dtype=bool)
    if len(keep) != len(trades):
        raise ValueError("keep must have one value per trade")
    k = int(keep.sum())
    score = _metric(metric, pnl)

    base = score(trades)
    filtered = score(trades[keep])
    rng = np.random.default_rng(seed)
    sims = np.empty(n)
    for i in range(n):
        pick = np.sort(rng.choice(len(trades), size=k, replace=False))  # same count, chosen at random / 같은 수를 무작위로
        sims[i] = score(trades.iloc[pick])

    return {"all": base, "filtered": filtered, "random": sims, "kept": k, "of": len(trades),
            "p": float((sims >= filtered).mean()),
            "from_fewer": float(sims.mean() - base), "from_selection": float(filtered - sims.mean())}


def _metric(metric, pnl: str) -> Callable[[pd.DataFrame], float]:
    if callable(metric):
        return metric
    if metric == "sum":
        return lambda t: float(t[pnl].sum())
    if metric == "mean":
        return lambda t: float(t[pnl].mean()) if len(t) else 0.0
    raise ValueError(f"unknown metric {metric!r}")
