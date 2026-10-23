"""Is the average trade above zero, or could it be luck? Resample the trades.

Draw as many trades as there are, with replacement, n times (default 10,000), and count how often the average comes
out at or below zero. The t statistic of the mean is reported alongside.

    res = bootstrap_mean(trades["pnl"])
    res["p"]        # share of resamples whose mean is <= 0
    res["t"]        # mean / standard error

This treats trades as independent draws. It ignores their order and clustering, so a losing streak in one market
regime is mixed in with everything else. A low p says the average is unlikely to be zero *for this sample of trades*;
it says nothing about the next ones, and nothing about how many variants were tried before this one (see overfit).

[한국어] 평균 거래가 0보다 큰가, 운일 수 있나? 거래를 다시 뽑아 본다.
거래 수만큼을 복원 추출로 n번(기본 1만 번) 다시 뽑아, 평균이 0 이하로 나오는 비율을 센다. 평균의 t 통계도 함께 낸다.
    res["p"]  다시 뽑은 평균이 0 이하인 비율
    res["t"]  평균 ÷ 표준오차
거래를 서로 독립인 표본으로 본다. 순서와 몰림을 무시하므로, 한 시장 국면의 연패도 나머지와 섞인다.
p가 낮다는 것은 *이 거래들에서는* 평균이 0일 가능성이 낮다는 뜻일 뿐이다. 다음 거래에 대해서는 아무것도 말하지 않고,
이 설정을 고르기 전에 변형을 몇 개 시도했는지도 반영하지 않는다(그것은 overfit).
"""
from __future__ import annotations

import numpy as np

CHUNK = 1000  # resamples per batch, so long trade lists do not need one huge array / 한 번에 다시 뽑는 수. 거래가 많아도 큰 배열 하나가 필요 없게


def bootstrap_mean(pnl, n: int = 10_000, seed: int = 0) -> dict:
    x = np.asarray(pnl, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise ValueError("need at least two trades")
    rng = np.random.default_rng(seed)
    means = np.concatenate([rng.choice(x, size=(min(CHUNK, n - i), len(x)), replace=True).mean(axis=1)
                            for i in range(0, n, CHUNK)])
    sd = x.std(ddof=1)
    t = x.mean() / (sd / np.sqrt(len(x))) if sd > 0 else float("inf") * np.sign(x.mean())
    return {"mean": float(x.mean()), "t": float(t), "p": float((means <= 0).mean()),
            "trades": len(x), "n": n, "means": means}
