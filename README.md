# backtest-validation-kit

Small tools for two questions every backtest should answer before anyone trusts it:

1. **Is the backtest testing what I think it is testing?** (data, and live code vs. research code)
2. **Is the result better than chance?** (random entries, shuffled prices, removing trades at random, overfitting)

[한국어 README](README.ko.md)

## Why this exists

I ran a rule-based trading bot on live accounts with my own money for three years. In 2026 I retested those rules from scratch on a longer history and many more markets. Most of my "improvements" turned out not to be real, and one data error had been quietly changing results. Each lesson below became one tool here.

| Lesson | Tool |
|---|---|
| Data can be wrong without raising an error. An H4 price file had daily bars mixed into its older years, and the code ran fine. | `data_check` |
| In a strategy whose average trade loses money after costs, trading less loses less, whatever you remove. A filter can look smart just by cutting trades. | `trade_removal` |
| If you pick the best of many variants, the winner looks good from luck alone. | `random_entry`, `shuffle`, `overfit` |
| A bot can trade differently from its backtest without any error. | `reconcile` |

This repository contains only the **methods**, rewritten from scratch: the original research scripts are mixed with live-account data and the live rules, so they stay private. It has no real strategy parameters and no broker data. Examples 1–5 run on synthetic prices, example 6 on real public prices.

## What these methods found in my research

I used these methods in a write-up of the three years (section numbers below; the write-up itself is not public because it contains account figures).

| Section | What it found | Tool |
|---|---|---|
| 9.1 | The older years of the H4 price files were daily bars, for 17 markets. Nothing crashed, so the results alone did not show it | `data_check` |
| 9.4 | Most of the "improvements" from rules that skip trades came from trading less, not from picking better trades | `trade_removal` |
| 9.5 | The top-ranked strategy over ten years: random-entry p = 0.067 and deflated Sharpe 0.83, both short of the criteria set in advance; the best of 80 trend variants fell below the median out of sample with probability 0.56 (PBO) | `random_entry`, `shuffle`, `overfit` |
| 9.6 | The live bot file, replayed on past prices against a fake trading server, made 99 of the backtest's 101 trades with the same bar, entry price and stop | `reconcile` |

## Tools

| Module | Question | Method | Output |
|---|---|---|---|
| `data_check` | Is the price data what the file says it is? | Checks the median bar spacing month by month against the expected timeframe (catches, e.g., daily bars inside an H4 file), gaps, and impossible prices (non-positive, high below open/close, low above open/close), and isolated spikes | A table of problems by date range |
| `random_entry` | Does the entry timing add anything? | Keeps each market's number of trades and each trade's side, size, cost, and holding length, and draws only the entry bar at random, N times (default 1,000) | p = share of random runs whose Sharpe is at least the strategy's |
| `shuffle` | Does the strategy use price patterns, or just the drift? | Shuffles the order of whole bars to build fake prices with the same moves but no order, then reruns the strategy | p = share of fake histories where the strategy did at least as well |
| `trade_removal` | Is a filter picking better trades, or just fewer? | Compares the filter with removing the same number of trades at random, N times | p, and the improvement split into "from trading less" and "from selection" |
| `overfit` | After many trials, is the best result still meaningful? | PBO (probability of backtest overfitting, by combinatorially symmetric cross-validation) and the deflated Sharpe ratio (Bailey and López de Prado, 2014) | PBO (0.5 = coin flip), DSR (probability that the true Sharpe is above zero) |
| `reconcile` | Does the live code make the same trades as the backtest? | Pairs two trade lists by symbol, side, and nearest entry time within a tolerance, then compares entry time, entry price, and stop | Match rate and a table of every mismatch |

Helpers: `core` (trade list format, mark-to-market returns, Sharpe ratio) and `synth` (synthetic prices, a deliberately corrupted file, a bad tick). A toy moving-average strategy in `examples/` produces trade lists to test.

### Trade list format

One row per trade. `random_entry` and `shuffle` mark every trade to the bar closes the same way, so real and random trades differ only in timing.

| Column | Meaning |
|---|---|
| `symbol` | market name (optional with one market) |
| `side` | +1 long, −1 short |
| `entry_time`, `exit_time` | bar times; the trade enters and exits at those bars' closes |
| `size` | exposure as a fraction of equity (default 1) |
| `cost` | total cost of the trade as a fraction of equity (default 0) |
| `pnl` | used by `trade_removal` |
| `entry_price`, `stop` | used by `reconcile` |

## Quick start

```
pip install -r requirements.txt
cd examples
python 01_data_check.py
python 02_coin_flip.py
python 03_filter_or_fewer.py
python 04_pbo.py          # about half a minute
python 05_reconcile.py
python 06_real_prices.py  # downloads public BTC and ETH H4 bars once (network), about a minute
cd ..
python -m pytest tests
```

```python
from vkit import check_bars, random_entry_test, removal_test, pbo, deflated_sharpe, reconcile

check_bars(ohlc, "4h")                               # empty = no problems found
random_entry_test(trades, ohlc, n=1000)["p"]
removal_test(trades, keep=mask)["from_selection"]
pbo(returns_by_variant)["pbo"]                       # periods x variants
reconcile(backtest_trades, bot_trades, time_tol="1h")
```

## What the examples show

Examples 1–5 on synthetic data, example 6 on real public prices; fixed seeds.

1. **`01_data_check`**: a file that looks like H4 from 2016 has daily bars until the end of 2018, plus one bad tick. Both are found.
2. **`02_coin_flip`**: the toy strategy makes +24% on a market with a trending stretch. Against random entries p = 0.13, against shuffled prices p = 0.09, and as the best of 20 settings its deflated Sharpe is 0.85. Verdict: indistinguishable from a coin flip.
3. **`03_filter_or_fewer`**: on 20 markets with no edge, a "skip after two losses" filter improves the result. Trading less accounts for the gain on all 20; "selection" averages +2% and passes p < 0.05 on 3 of 20 by luck alone.
4. **`04_pbo`**: on 6 markets with no edge, PBO averages 0.52, but single markets range from 0.32 to 0.80.
5. **`05_reconcile`**: four planted faults (an entry one bar late, a different stop, a missed trade, an extra trade) are all listed.
6. **`06_real_prices`**: BTC and ETH, H4, 2020-01 to 2026-06, long only, 0.2% per round trip. `data_check` flags one "spike" in each, at 2020-03-12 20:00: the real COVID crash, not a data error. The tool finds candidates; a person decides. The strategy makes money (408 trades) and its deflated Sharpe is 0.99, but holding both coins had a higher Sharpe (0.87 vs 0.72), random entries did as well often enough (p = 0.18), and so did shuffled prices (BTC p = 0.07, ETH p = 0.36). Verdict: indistinguishable from a coin flip.

## How to read the results

- **Set the pass criteria before you run anything** (for example, p < 0.05 on every dataset, and DSR ≥ 0.95). Writing them down first is what keeps the tests honest.
- Start from the assumption that **every strategy is a coin flip**. These tests can reject that assumption; they cannot prove an edge.
- A low p against random entries means the timing beats random timing *on this data*. It says nothing about the future.
- PBO near 0.5 means the selection process itself is overfitting: the in-sample winner is no better than a random pick out of sample.
- **Making money is not timing skill.** A high deflated Sharpe says the returns are unlikely to be zero even after the trials; it does not say the entries are well timed. In example 6 the DSR is 0.99, yet holding the same coins did better.
- **One p-value or one PBO is one draw.** On markets with no edge at all, a single market's PBO ranged from 0.32 to 0.80 (example 4), and a useless filter passed p < 0.05 on 3 of 20 (example 3). Require several datasets, including ones you did not use to choose the rules.

## What this does not do

- It does not model your costs. Commissions, spreads, slippage, and overnight financing (swap) must already be in your trade list. In my own tests, swap alone turned a profit before costs into a loss.
- It is not a backtest engine. It takes prices and trade lists that you produce.
- Random trades may overlap each other, and every trade is marked at bar closes; intrabar stops are not modelled.
- The examples show how the tools work, not whether any strategy works. Example 6 is one window and two coins.

## Related

[bot-ops-mcp](https://github.com/swanmun/bot-ops-mcp): a read-only MCP server for my live trading bots, with an evaluation of 10 AI assistants on 38 operator questions.

## License

MIT. See [LICENSE](LICENSE).
