"""Does the live code make the same trades as the backtest?

A bot can trade differently from its backtest without raising any error: a missed
condition, a different price feed, an exchange that merges positions. Run the bot on the
same past prices (against a fake trading server, or in its own replay mode), then match
its trades with the backtest's.

    table = reconcile(backtest_trades, bot_trades, time_tol="0h", rel_tol=1e-6)
    summary(table)      # counts per status and the match rate

Trades are paired within the same symbol and side, nearest entry time first, within time_tol.
status
  match        paired at the same entry time, and every field in `fields` agrees within rel_tol
  differs      paired, but the entry time or a field disagrees (listed in `detail`)
  missing      in expected only (the bot did not make this trade)
  extra        in actual only (the bot made a trade the backtest did not)
Every row that is not a match is a question to answer: which side is wrong, and why.

[한국어] 실거래 코드가 백테스트와 같은 거래를 내는가?
봇은 오류 하나 없이 백테스트와 다르게 매매할 수 있다. 빠뜨린 조건, 다른 가격 피드, 포지션을 합치는 거래소 등.
같은 과거 가격으로 봇을 돌리고(가짜 거래 서버 위에서, 또는 봇의 재생 모드로), 그 거래를 백테스트 거래와 맞춘다.
거래는 같은 종목 · 같은 방향 안에서, time_tol 안의 가장 가까운 진입 시점부터 짝짓는다.
  match    같은 진입 시점에 짝지어졌고, fields의 값이 rel_tol 안에서 모두 같음
  differs  짝지어졌지만 진입 시점이나 값이 다름(detail에 적음)
  missing  기대한 쪽에만 있음(봇이 이 거래를 하지 않음)
  extra    실제 쪽에만 있음(백테스트에 없는 거래를 봇이 함)
match가 아닌 줄은 모두 답해야 할 질문이다. 어느 쪽이 틀렸고, 왜인가.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reconcile(expected: pd.DataFrame, actual: pd.DataFrame, time_tol: str = "0h", rel_tol: float = 1e-6,
              fields: tuple[str, ...] = ("entry_price", "stop"), keys: tuple[str, ...] = ("symbol", "side")) -> pd.DataFrame:
    exp, act = _prep(expected, keys), _prep(actual, keys)
    fields = tuple(f for f in fields if f in exp and f in act)
    tol = pd.Timedelta(time_tol)
    rows = []
    used = np.zeros(len(act), dtype=bool)  # actual trades already paired / 이미 짝지어진 실제 거래
    for i, e in exp.iterrows():
        same = (~used) & np.all([act[k].to_numpy() == e[k] for k in keys], axis=0)
        gap = (act["entry_time"] - e["entry_time"]).abs()
        cand = np.flatnonzero(same & (gap <= tol).to_numpy())
        if len(cand) == 0:
            rows.append(_row(e, None, "missing", ""))
            continue
        j = cand[np.argmin(gap.to_numpy()[cand])]  # nearest entry time first / 진입 시점이 가장 가까운 것부터
        used[j] = True
        a = act.iloc[j]
        off = [f"{f} {e[f]:.6g} vs {a[f]:.6g}" for f in fields if not np.isclose(a[f], e[f], rtol=rel_tol, atol=0)]
        late = a["entry_time"] - e["entry_time"]
        if late != pd.Timedelta(0):
            off.insert(0, f"entry {'+' if late > pd.Timedelta(0) else '-'}{abs(late)}")
        rows.append(_row(e, a, "differs" if off else "match", ", ".join(off)))
    for j in np.flatnonzero(~used):
        rows.append(_row(None, act.iloc[j], "extra", ""))
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> dict:
    """Counts per status, and matches as a share of the expected trades. / 상태별 개수, 그리고 기대한 거래 중 match의 비율."""
    counts = table["status"].value_counts().to_dict()
    expected = sum(counts.get(s, 0) for s in ("match", "differs", "missing"))
    return {**{s: counts.get(s, 0) for s in ("match", "differs", "missing", "extra")},
            "match_rate": counts.get("match", 0) / expected if expected else float("nan")}


def _prep(t: pd.DataFrame, keys) -> pd.DataFrame:
    t = t.copy()
    for k in keys:
        if k not in t:
            t[k] = "_"
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    return t.sort_values("entry_time").reset_index(drop=True)


def _row(e, a, status: str, detail: str) -> dict:
    src = e if e is not None else a
    return {"symbol": src["symbol"], "side": src["side"],
            "expected_entry": None if e is None else e["entry_time"],
            "actual_entry": None if a is None else a["entry_time"],
            "status": status, "detail": detail}
