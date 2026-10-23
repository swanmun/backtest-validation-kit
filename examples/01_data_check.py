"""Find daily bars hidden in an H4 file, and one bad tick.

The file below looks like 4-hour data from 2016. In fact everything before 2019 is daily
bars, the way some exports fill history they do not have at the requested timeframe.
A backtest would run without errors and test a daily strategy for those years.

[한국어] H4 파일 속에 숨은 일봉과 잘못된 틱 하나를 찾는다.
아래 파일은 2016년부터의 4시간 단위 데이터처럼 보이지만, 실제로는 2019년 전이 모두 일봉이다.
요청한 시간 단위의 과거가 없을 때 일부 내보내기가 이렇게 채운다.
백테스트는 오류 없이 돌고, 그 구간에서는 일봉 전략을 시험하게 된다.
"""
import _path  # noqa: F401
import pandas as pd

from vkit import check_bars
from vkit.synth import add_spike, mix_in_daily, random_walk

h4 = random_walk(n_bars=12000, freq="4h", start="2016-01-04", seed=1)
bad = mix_in_daily(h4, "2019-01-01")                    # older years become daily bars / 오래된 구간을 일봉으로
bad = add_spike(bad, at=len(bad) - 500, factor=1.15)    # one bad tick / 잘못된 틱 하나

print(f"bars: {len(bad):,}, from {bad.index[0]:%Y-%m-%d} to {bad.index[-1]:%Y-%m-%d}\n")
issues = check_bars(bad, "4h")
pd.set_option("display.width", 160)
print(issues.to_string(index=False) if len(issues) else "no problems found")

first_ok = issues.loc[issues["kind"] == "timeframe", "end"].max()
print(f"\nUse this file only after {first_ok:%Y-%m-%d}, or re-export the older years at the right timeframe.")
