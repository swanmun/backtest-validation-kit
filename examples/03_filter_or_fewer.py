"""A filter that "improves" a losing strategy: is it selecting, or just trading less?

On random walks with costs, the toy strategy's average trade loses money. The filter
skips the next trade after two losses in a row. It usually looks better. Compare it with
removing the same number of trades at random, on 20 independent markets with no edge:
the gain from trading less shows up every time; the gain from "selection" is noise around
zero, and on one market in twenty or so it will pass p < 0.05 by luck alone.

[한국어] 손실 전략을 '개선'하는 필터: 고르는 것인가, 덜 거래할 뿐인가?
비용이 있는 무작위 행보에서 장난감 전략의 평균 거래는 손실이다. 필터는 2연패 뒤 다음 거래를 건너뛴다.
대개 결과가 좋아 보인다. 우위가 없는 서로 다른 시장 20개에서, 같은 수의 거래를 무작위로 뺀 결과와 비교한다.
덜 거래한 몫은 매번 나타나고, '고른 몫'은 0 주변의 잡음이며, 스무 시장에 하나쯤은 운만으로 p < 0.05를 넘는다.
"""
import _path  # noqa: F401
import numpy as np

from toy_strategy import ma_breakout
from vkit import removal_test
from vkit.synth import random_walk


def skip_after_two_losses(pnl) -> np.ndarray:
    """True for trades the filter keeps. / 필터가 남기는 거래가 True."""
    keep, losses = np.ones(len(pnl), dtype=bool), 0
    for i, p in enumerate(pnl):
        if losses >= 2:
            keep[i], losses = False, 0
            continue
        losses = losses + 1 if p < 0 else 0
    return keep


rows = []
print("market  trades  avg/trade     all  filtered  from fewer  from selection      p")
for seed in range(1, 21):
    prices = random_walk(n_bars=12000, seed=seed)  # a new market with no edge / 우위가 없는 새 시장
    trades = ma_breakout(prices, ma=30, lookback=10, cost=0.001)
    r = removal_test(trades, skip_after_two_losses(trades["pnl"]), n=500, seed=seed)
    rows.append(r)
    print(f"{seed:>6}  {r['of']:>6}  {trades['pnl'].mean():>+8.3%}  {r['all']:>+6.1%}  {r['filtered']:>+8.1%}"
          f"  {r['from_fewer']:>+10.1%}  {r['from_selection']:>+14.1%}  {r['p']:>5.3f}")

fewer = np.array([r["from_fewer"] for r in rows])
sel = np.array([r["from_selection"] for r in rows])
print(f"\nfrom fewer:     positive on {np.sum(fewer > 0)} of {len(rows)} markets, mean {fewer.mean():+.1%}")
print(f"from selection: positive on {np.sum(sel > 0)} of {len(rows)}, mean {sel.mean():+.1%}, "
      f"p < 0.05 on {sum(r['p'] < 0.05 for r in rows)}")
print("(from fewer = average of removing the same number of trades at random; from selection = what the filter adds beyond that)")
