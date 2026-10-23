"""Is the price data what the file says it is?

The failure this is built for: an exported "H4" file whose older years were quietly
filled with daily bars. Nothing crashed; the backtest simply tested a daily strategy
for those years under an H4 name. Checking the median bar spacing month by month
finds that kind of segment, along with gaps and impossible prices.

    issues = check_bars(df, "4h")      # empty DataFrame = no problems found
    assert_clean(df, "4h")             # raise instead, e.g. inside a data loader

df: OHLC DataFrame (open, high, low, close) with a DatetimeIndex.

[한국어] 가격 데이터가 파일 이름대로인가?
이 도구가 겨냥하는 사고: 내보낸 'H4' 파일의 오래된 구간이 조용히 일봉으로 채워져 있었다.
아무것도 멈추지 않았고, 백테스트는 그 구간에서 H4라는 이름으로 일봉 전략을 시험했다.
달마다 봉 간격의 중앙값을 보면 그런 구간을 찾을 수 있고, 빈 구간과 있을 수 없는 가격도 함께 찾는다.
    check_bars(df, "4h")   빈 표면 문제 없음
    assert_clean(df, "4h") 문제가 있으면 오류를 낸다(데이터를 읽는 곳에 넣어 쓴다)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["kind", "start", "end", "bars", "detail"]


def check_bars(df: pd.DataFrame, timeframe: str, max_gap: str = "4D", spike_z: float = 12.0) -> pd.DataFrame:
    """Return one row per problem: kind, start, end, bars, detail.

    kinds
      order      index not sorted, or duplicate times
      timeframe  months whose median bar spacing is not the expected timeframe
      gap        spacing longer than max_gap (weekends and short holidays are below the default)
      price      non-positive prices, high below open/close, low above open/close
      spike      a single bar that jumps far from its neighbours and comes straight back

    [한국어] 문제 하나에 한 줄(종류, 시작, 끝, 봉 수, 설명)을 돌려준다.
      order      시각이 정렬되지 않았거나 같은 시각이 두 번
      timeframe  봉 간격의 중앙값이 기대한 시간 단위가 아닌 달
      gap        max_gap보다 긴 빈 구간(주말 · 짧은 휴일은 기본값보다 짧다)
      price      0 이하 가격, 고가가 시가 · 종가보다 낮음, 저가가 시가 · 종가보다 높음
      spike      봉 하나가 앞뒤에서 크게 튀었다가 바로 돌아옴
    """
    tf = pd.Timedelta(timeframe)
    issues: list[dict] = []
    idx = df.index

    if not idx.is_monotonic_increasing:
        issues.append(_row("order", idx.min(), idx.max(), len(idx), "index is not sorted"))
    dup = idx.duplicated()
    if dup.any():
        issues.append(_row("order", idx[dup][0], idx[dup][-1], int(dup.sum()), "duplicate times"))
    df = df[~dup].sort_index()
    idx = df.index
    if len(idx) < 3:
        return pd.DataFrame(issues, columns=COLUMNS)

    # timeframe: median spacing per month, consecutive months with the same wrong value merged
    # 시간 단위: 달마다 봉 간격의 중앙값, 같은 틀린 값이 이어지는 달은 한 줄로 합친다
    spacing = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
    month = spacing.groupby(spacing.index.to_period("M")).median()
    wrong = month[(month > tf * 1.5) | (month < tf * 0.5)]
    run = None
    for period, med in wrong.items():
        if run and run["med"] == med and period == run["last"] + 1:
            run["last"] = period
            continue
        if run:
            issues.append(_timeframe_row(df, run, tf))
        run = {"first": period, "last": period, "med": med}
    if run:
        issues.append(_timeframe_row(df, run, tf))

    # gaps / 빈 구간
    gap = spacing[spacing > pd.Timedelta(max_gap)]
    for t, d in gap.items():
        before = idx[idx.get_loc(t) - 1]
        issues.append(_row("gap", before, t, 0, f"no bars for {d}"))

    # impossible prices / 있을 수 없는 가격
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    eps = 1e-12
    bad = (np.minimum.reduce([o, h, l, c]) <= 0) | (h + eps < np.maximum(o, c)) | (l - eps > np.minimum(o, c))
    for t in idx[bad]:
        issues.append(_row("price", t, t, 1, _price_detail(df.loc[t])))

    # isolated spikes: a big log return followed by a big one the other way
    # 홀로 튀는 값: 큰 로그 수익 바로 뒤에 반대 방향으로 큰 수익(robust z, 중앙값 절대 편차 기준)
    lr = np.diff(np.log(np.where(c > 0, c, np.nan)))
    mad = np.nanmedian(np.abs(lr - np.nanmedian(lr))) * 1.4826
    if mad > 0:
        z = (lr - np.nanmedian(lr)) / mad
        hit = (np.abs(z[:-1]) > spike_z) & (np.abs(z[1:]) > spike_z / 2) & (np.sign(z[:-1]) != np.sign(z[1:]))
        for i in np.flatnonzero(hit):
            t = idx[i + 1]
            issues.append(_row("spike", t, t, 1, f"close {c[i + 1]:.6g} between {c[i]:.6g} and {c[i + 2]:.6g}"))

    return pd.DataFrame(issues, columns=COLUMNS)


def assert_clean(df: pd.DataFrame, timeframe: str, **kwargs) -> None:
    """Raise ValueError listing every problem. Use it where the data is read, so no test runs on bad data.

    [한국어] 문제를 모두 나열하며 ValueError를 낸다. 데이터를 읽는 곳에 두면 틀린 데이터로는 어떤 시험도 돌지 않는다.
    """
    issues = check_bars(df, timeframe, **kwargs)
    if len(issues):
        raise ValueError("price data failed the check:\n" + issues.to_string(index=False))


def _row(kind, start, end, bars, detail) -> dict:
    return {"kind": kind, "start": start, "end": end, "bars": bars, "detail": detail}


def _timeframe_row(df: pd.DataFrame, run: dict, tf: pd.Timedelta) -> dict:
    part = df[(df.index.to_period("M") >= run["first"]) & (df.index.to_period("M") <= run["last"])]
    return _row("timeframe", part.index[0], part.index[-1], len(part),
                f"median bar spacing {run['med']}, expected {tf}")


def _price_detail(bar: pd.Series) -> str:
    return "open {:.6g} high {:.6g} low {:.6g} close {:.6g}".format(*(bar[k] for k in ("open", "high", "low", "close")))
