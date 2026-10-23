"""The same tests on real prices: BTC and ETH, H4 bars, 2020-01 to 2026-06 (public data, see public_data.py).

Here the toy strategy is long only (spot) and pays 0.2% per round trip (0.1% fee on each side), no slippage.
Both coins rose a lot in this window, so a long-only strategy that is often in the market will likely make money.
The question is the same as in example 02: is the timing the reason, or would holding have done as well?
The first run downloads the bars (network needed once).

[한국어] 같은 시험을 실제 가격으로: BTC와 ETH의 4시간 봉, 2020-01 ~ 2026-06(공개 데이터, public_data.py 참고).
여기서 장난감 전략은 매수만 하고(현물), 거래 한 번(진입 + 청산)에 0.2%(양쪽 수수료 0.1%씩)를 내며 슬리피지는 없다고 둔다.
이 기간에 두 코인 모두 크게 올랐으므로, 시장에 자주 들어가 있는 매수 전략은 돈을 벌 가능성이 크다.
질문은 예제 02와 같다. 이익이 타이밍 덕인가, 아니면 그냥 들고 있어도 그만큼 됐나?
처음 실행할 때 봉을 내려받는다(네트워크는 한 번만 필요).
"""
import _path  # noqa: F401
import pandas as pd

from public_data import END, START, h4
from toy_strategy import ma_breakout
from vkit import Book, check_bars, deflated_sharpe, random_entry_test, shuffle_test, trial_sharpe_variance

# 1. Pass criteria, written down before any result is seen / 통과 기준, 결과를 보기 전에 적는다
P_MAX, DSR_MIN = 0.05, 0.95
COST = 0.002
print(f"Pass only if: random-entry p < {P_MAX}, shuffled-price p < {P_MAX} on every market, and deflated Sharpe >= {DSR_MIN}\n")

# 2. Data, checked before use / 데이터는 쓰기 전에 점검한다
prices = {s: h4(s) for s in ("BTCUSDT", "ETHUSDT")}
for s, df in prices.items():
    issues = check_bars(df, "4h")
    found = ", ".join(f"{k} {n}" for k, n in issues["kind"].value_counts().items()) or "no problems"
    print(f"{s}: {len(df):,} bars, {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}; data_check: {found}")
    if len(issues):
        print(issues.to_string(index=False, max_colwidth=70))
print(f"(window {START} to {END}, UTC)\n")

# 3. The strategy, and holding both coins for the whole window for comparison / 전략, 그리고 비교용으로 기간 내내 들고 있기
trades = pd.concat([ma_breakout(df, cost=COST, both_sides=False, symbol=s) for s, df in prices.items()], ignore_index=True)
hold = pd.DataFrame([{"symbol": s, "side": 1, "entry_time": df.index[0], "exit_time": df.index[-1], "cost": COST}
                     for s, df in prices.items()])
print(f"toy strategy: {len(trades)} trades ({', '.join(f'{s} {n}' for s, n in trades['symbol'].value_counts().items())}), "
      f"sum of trade returns {trades['pnl'].sum():+.0%}")
print(f"Sharpe: strategy {Book(trades, prices).sharpe():.2f}, holding both {Book(hold, prices).sharpe():.2f}\n")

# 4. Random entry times, same trades otherwise / 진입 시점만 무작위, 나머지는 같은 거래
re_ = random_entry_test(trades, prices, n=1000, seed=1)
print(f"random entry   Sharpe {re_['actual']:.2f}, random median {pd.Series(re_['random']).median():.2f}, p = {re_['p']:.3f}")

# 5. Same strategy on shuffled prices, one market at a time / 같은 전략을 섞은 가격에, 종목 하나씩
sh = {s: shuffle_test(lambda d: ma_breakout(d, cost=COST, both_sides=False), df, n=200, seed=2) for s, df in prices.items()}
for s, r in sh.items():
    print(f"shuffled bars  {s} Sharpe {r['actual']:.2f}, fake median {pd.Series(r['fake']).median():.2f}, p = {r['p']:.3f}")

# 6. Deflated Sharpe: the setting is the best of 20 / 보정 샤프: 이 설정은 20개 중 1등
grid = [(ma, lb) for ma in (20, 30, 50, 80, 120) for lb in (10, 20, 40, 60)]
curves = pd.DataFrame({g: Book(pd.concat([ma_breakout(df, *g, cost=COST, both_sides=False, symbol=s)
                                          for s, df in prices.items()], ignore_index=True), prices).returns()
                       for g in grid})
best = curves.mean().div(curves.std()).idxmax()
d = deflated_sharpe(curves[best], n_trials=len(grid), sr_variance=trial_sharpe_variance(curves))
print(f"deflated Sharpe best of {len(grid)} (ma {best[0]}, lookback {best[1]}): DSR = {d['dsr']:.2f}")

passed = re_["p"] < P_MAX and all(r["p"] < P_MAX for r in sh.values()) and d["dsr"] >= DSR_MIN
print("\nverdict:", "passes all three" if passed else "indistinguishable from a coin flip")
