"""Two questions about a trade list: could the average trade be zero, and how much rests on a few trades?

Two synthetic markets: a random walk with no edge, and the market of example 02 with a trending stretch in the middle.
The toy strategy runs on both. Bootstrap asks whether its average trade could be zero; top_dependence asks how much
of the total is left without its best trades (only meaningful when the total is positive).

In the trend market the strategy makes money, yet the bootstrap cannot rule out a zero average, and without its three
best trades the total turns negative. A positive total over one sample can rest on a handful of trades that the next
sample may not contain.

[한국어] 거래 목록에 던질 두 질문: 평균 거래가 0일 수 있나, 그리고 몇 건에 얼마나 기대나?
합성 시장 둘: 우위가 없는 무작위 행보, 그리고 예제 02의 가운데 추세 구간이 있는 시장. 장난감 전략을 둘 다에 돌린다.
부트스트랩은 평균 거래가 0일 수 있는지 묻고, top_dependence는 가장 좋은 거래들을 빼면 합계가 얼마나 남는지 묻는다
(합계가 양수일 때만 뜻이 있다).
추세 시장에서 전략은 돈을 벌지만, 부트스트랩으로는 평균이 0일 가능성을 배제하지 못하고, 가장 좋은 3건을 빼면 합계가
음수가 된다. 한 표본에서 난 이익은, 다음 표본에는 없을 수도 있는 몇 건에 기대고 있을 수 있다.
"""
import _path  # noqa: F401

from toy_strategy import ma_breakout
from vkit import bootstrap_mean, top_dependence
from vkit.synth import random_walk

markets = {
    "random walk, no edge": random_walk(n_bars=12000, seed=3),
    "trend in the middle (example 02)": random_walk(n_bars=12000, regimes=[(0.4, 0.0), (0.2, 0.0004), (0.4, 0.0)], seed=7),
}
for name, prices in markets.items():
    trades = ma_breakout(prices)
    b = bootstrap_mean(trades["pnl"], n=10_000, seed=1)
    print(f"{name}: {b['trades']} trades, total {trades['pnl'].sum():+.1%}")
    print(f"  bootstrap  t = {b['t']:.2f}, p(mean <= 0) = {b['p']:.3f}")
    if trades["pnl"].sum() <= 0:
        print("  top_dependence: the total is not positive, so there is no gain to depend on")
        continue
    d = top_dependence(trades["pnl"], tops=(1, 3, 5, 10))
    left = ", ".join(f"top {r.removed}: {r.share_left:.0%}" for r in d["table"].itertuples() if r.removed)
    print(f"  without the best trades, share of the total left: {left}; the total turns <= 0 after removing "
          f"{d['negative_after']}")
