"""How much of the result rests on a few trades?

Remove the best 1, 3, 5 ... trades and see what is left of the total. Trend following lives on rare large wins, so
some dependence is by design; but a gain that disappears without its top five trades means the result over a short
sample says little, because the next sample may simply not contain those five.

    res = top_dependence(trades["pnl"])
    res["table"]                # removed, total left, share of the original total left
    res["negative_after"]       # fewest top trades whose removal makes the total <= 0 (0 if it already is, None if never)

[한국어] 결과가 몇 건의 거래에 얼마나 기대고 있나?
가장 좋은 거래 1, 3, 5 …건을 빼고 합계가 얼마나 남는지 본다. 추세 추종은 드문 큰 이익으로 버티므로 어느 정도 기대는 것은
설계상 자연스럽다. 하지만 상위 5건이 없으면 이익이 사라진다면, 짧은 표본의 결과는 별 뜻이 없다.
다음 표본에는 그 5건이 없을 수도 있기 때문이다.
    res["table"]           뺀 건수, 남은 합계, 원래 합계 중 남은 비율
    res["negative_after"]  합계를 0 이하로 만드는 데 필요한 상위 거래의 최소 건수(이미 0 이하면 0, 끝까지 안 되면 None)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def top_dependence(pnl, tops: tuple[int, ...] = (1, 3, 5)) -> dict:
    x = np.asarray(pnl, dtype=float)
    x = np.sort(x[np.isfinite(x)])[::-1]  # best first / 좋은 거래부터
    total = x.sum()
    left = total - np.concatenate([[0.0], np.cumsum(x)])  # left[k] = total without the top k trades / 상위 k건을 뺀 합계
    rows = [{"removed": k, "total_left": float(left[k]),
             "share_left": float(left[k] / total) if total > 0 else np.nan}
            for k in (0, *tops) if k <= len(x)]
    neg = np.flatnonzero(left <= 0)
    return {"table": pd.DataFrame(rows), "negative_after": int(neg[0]) if len(neg) else None, "trades": len(x)}
