"""Is the toy strategy better than a coin flip? Three tests, criteria fixed first.

The market is synthetic: a random walk with a trending stretch in the middle. A trend
strategy will make money in that stretch. The question is whether its timing is the
reason, or whether being in the market during the trend would have been enough.

[한국어] 장난감 전략이 동전 던지기보다 나은가? 시험 세 가지, 통과 기준은 먼저 정한다.
시장은 합성 데이터다. 가운데에 추세 구간이 있는 무작위 행보라, 추세 전략은 그 구간에서 돈을 번다.
질문은 그 이익이 타이밍 덕인지, 아니면 추세 동안 시장에 있기만 했어도 충분했는지다.
"""
import _path  # noqa: F401
import pandas as pd

from toy_strategy import ma_breakout
from vkit import Book, deflated_sharpe, random_entry_test, shuffle_test, trial_sharpe_variance
from vkit.synth import random_walk

# 1. Pass criteria, written down before any result is seen / 통과 기준, 결과를 보기 전에 적는다
P_MAX, DSR_MIN = 0.05, 0.95
print(f"Pass only if: random-entry p < {P_MAX}, shuffled-price p < {P_MAX}, and deflated Sharpe >= {DSR_MIN}\n")

prices = random_walk(n_bars=12000, regimes=[(0.4, 0.0), (0.2, 0.0004), (0.4, 0.0)], seed=7)
trades = ma_breakout(prices)
print(f"toy strategy: {len(trades)} trades, total P&L {trades['pnl'].sum():+.1%} (sum of trade returns)")

# 2. Random entry times, same trades otherwise / 진입 시점만 무작위, 나머지는 같은 거래
re_ = random_entry_test(trades, prices, n=1000, seed=1)
print(f"random entry   Sharpe {re_['actual']:.2f}, random median {pd.Series(re_['random']).median():.2f}, p = {re_['p']:.3f}")

# 3. Same strategy on shuffled prices / 같은 전략을 섞은 가격에
sh = shuffle_test(ma_breakout, prices, n=200, seed=2)
print(f"shuffled bars  Sharpe {sh['actual']:.2f}, fake median {pd.Series(sh['fake']).median():.2f}, p = {sh['p']:.3f}")

# 4. Deflated Sharpe: this setting was picked from a grid of 20, so allow for that
#    보정 샤프: 이 설정은 20개 중에서 고른 것이므로 그만큼을 빼고 본다
grid = [(ma, lb) for ma in (20, 30, 50, 80, 120) for lb in (10, 20, 40, 60)]
curves = pd.DataFrame({g: Book(ma_breakout(prices, *g), prices).returns() for g in grid})
best = curves.mean().div(curves.std()).idxmax()
d = deflated_sharpe(curves[best], n_trials=len(grid), sr_variance=trial_sharpe_variance(curves))
print(f"deflated Sharpe best of {len(grid)} (ma {best[0]}, lookback {best[1]}): DSR = {d['dsr']:.2f}")

passed = re_["p"] < P_MAX and sh["p"] < P_MAX and d["dsr"] >= DSR_MIN
print("\nverdict:", "passes all three" if passed else "indistinguishable from a coin flip")
