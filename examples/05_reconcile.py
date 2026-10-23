"""Match a bot's trades against its backtest.

Here the "bot" is the backtest with four realistic faults planted in it: one entry a bar
late (at the next bar's price), one stop set differently, one trade missed, and one extra
trade. In practice the bot list comes from running the real bot on the same past prices.

[한국어] 봇의 거래를 백테스트 거래와 맞춰 본다.
여기서 '봇'은 백테스트에 실제로 있을 법한 문제 네 가지를 일부러 심은 것이다.
한 봉 늦은 진입(다음 봉 가격에), 다르게 잡힌 손절, 빠진 거래, 더 생긴 거래.
실제로는 진짜 봇을 같은 과거 가격으로 돌려 거래 목록을 얻는다.
"""
import _path  # noqa: F401
import pandas as pd

from toy_strategy import ma_breakout
from vkit import reconcile, summary
from vkit.synth import random_walk

prices = random_walk(n_bars=6000, seed=6)
backtest = ma_breakout(prices, symbol="SYN")

bot = backtest.copy()
late = prices.index[prices.index.get_loc(bot.loc[3, "entry_time"]) + 1]
bot.loc[3, ["entry_time", "entry_price"]] = [late, prices.loc[late, "close"]]  # entered one bar late / 한 봉 늦게 진입
bot.loc[7, "stop"] = bot.loc[7, "stop"] * 1.002                                  # stop computed differently / 손절을 다르게 계산
bot = bot.drop(index=11)                                                         # trade missed / 거래를 빠뜨림
extra = bot.iloc[[5]].assign(entry_time=bot["entry_time"].iloc[5] + pd.Timedelta(days=1))
bot = pd.concat([bot, extra], ignore_index=True)                                 # trade the backtest never made / 백테스트에 없는 거래

table = reconcile(backtest, bot, time_tol="1D")
s = summary(table)
print(f"backtest {len(backtest)} trades, bot {len(bot)} trades")
print(f"match {s['match']}, differs {s['differs']}, missing {s['missing']}, extra {s['extra']} "
      f"-> match rate {s['match_rate']:.1%}\n")
pd.set_option("display.width", 160)
print(table[table["status"] != "match"].to_string(index=False))
