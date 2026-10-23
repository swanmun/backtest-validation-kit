"""Tests for every tool, on synthetic data. Run: python -m pytest tests

[한국어] 도구마다의 테스트(합성 데이터). 실행: python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from toy_strategy import ma_breakout  # noqa: E402
from vkit import (Book, assert_clean, bootstrap_mean, top_dependence, check_bars, deflated_sharpe, expected_max_sharpe, mtm_returns, pbo,  # noqa: E402
                  random_entry_test, reconcile, removal_test, sharpe, shuffle_bars, shuffle_test, summary)
from vkit.synth import add_spike, mix_in_daily, random_walk  # noqa: E402


@pytest.fixture(scope="module")
def h4():
    return random_walk(n_bars=6000, seed=11)


# data_check / 데이터 점검

def test_clean_data_has_no_problems(h4):
    assert check_bars(h4, "4h").empty


def test_daily_bars_inside_h4_file_are_found(h4):
    until = h4.index[3000]
    issues = check_bars(mix_in_daily(h4, until), "4h")
    tf = issues[issues["kind"] == "timeframe"]
    assert len(tf) == 1
    assert tf["start"].iloc[0] == h4.index[0].normalize()
    assert tf["end"].iloc[0] < until


def test_spike_and_bad_price_are_found(h4):
    bad = add_spike(h4, at=2000, factor=1.2)
    bad.iloc[4000, bad.columns.get_loc("high")] = bad["low"].iloc[4000] * 0.99
    kinds = set(check_bars(bad, "4h")["kind"])
    assert {"spike", "price"} <= kinds
    with pytest.raises(ValueError):
        assert_clean(bad, "4h")


# core / 공통 부품

def test_mark_to_market_one_trade():
    idx = pd.date_range("2024-01-01", periods=4, freq="1D")
    close = pd.Series([100.0, 110.0, 121.0, 121.0], index=idx)
    trades = pd.DataFrame({"side": [1], "entry_time": [idx[0]], "exit_time": [idx[2]], "cost": [0.01]})
    r = mtm_returns(trades, close, freq=None).to_numpy()
    assert np.allclose(r, [0.0, 0.10, 0.10 - 0.01, 0.0])


def test_short_and_size():
    idx = pd.date_range("2024-01-01", periods=3, freq="1D")
    close = pd.Series([100.0, 90.0, 99.0], index=idx)
    trades = pd.DataFrame({"side": [-1], "entry_time": [idx[0]], "exit_time": [idx[2]], "size": [0.5]})
    assert np.allclose(mtm_returns(trades, close, freq=None), [0.0, 0.05, -0.05])


def test_sharpe_basics():
    assert sharpe([1.0, 1.0, 1.0]) == 0.0
    assert sharpe([0.01, 0.03]) == pytest.approx(0.02 / np.std([0.01, 0.03], ddof=1))


# random entry and shuffle / 무작위 진입과 가격 섞기

def test_random_entry_is_reproducible_and_bounded(h4):
    trades = ma_breakout(h4)
    a = random_entry_test(trades, h4, n=50, seed=3)
    b = random_entry_test(trades, h4, n=50, seed=3)
    assert np.array_equal(a["random"], b["random"])
    assert 0.0 <= a["p"] <= 1.0
    assert a["actual"] == pytest.approx(Book(trades, h4).sharpe())


def test_random_entry_respects_allowed_bars(h4):
    trades = ma_breakout(h4)
    allowed = pd.Series(False, index=h4.index)
    allowed.iloc[:200] = True  # only the first 200 bars may be used / 앞 200봉에서만 진입
    res = random_entry_test(trades, h4, n=5, seed=1, allowed=allowed)
    assert np.isfinite(res["random"]).all()


def test_shuffle_keeps_bars_and_validity(h4):
    fake = shuffle_bars(h4, np.random.default_rng(0))
    real_moves = np.sort((h4["close"] / h4["close"].shift()).dropna().to_numpy())
    fake_moves = np.sort((fake["close"] / fake["close"].shift()).dropna().to_numpy())
    assert np.allclose(real_moves, fake_moves)
    assert (fake["high"] >= fake[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (fake["low"] <= fake[["open", "close"]].min(axis=1) + 1e-9).all()
    res = shuffle_test(ma_breakout, h4, n=3, seed=0)
    assert len(res["fake"]) == 3


# trade removal / 거래 무작위 제거

def test_removal_with_identical_trades_has_no_selection():
    trades = pd.DataFrame({"pnl": [-1.0] * 100})
    keep = np.arange(100) % 2 == 0
    res = removal_test(trades, keep, n=100, seed=0)
    assert res["from_fewer"] == pytest.approx(50.0)
    assert res["from_selection"] == pytest.approx(0.0)
    assert res["p"] == 1.0


def test_removal_that_drops_the_losers_is_selection():
    trades = pd.DataFrame({"pnl": [1.0] * 50 + [-1.0] * 50})
    res = removal_test(trades, trades["pnl"] > 0, n=200, seed=0)
    assert res["from_selection"] > 40 and res["p"] == 0.0


# overfit / 과적합

def test_pbo_noise_versus_real_edge():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((1600, 30))
    assert 0.1 < pbo(noise)["pbo"] < 0.9
    edge = noise.copy()
    edge[:, 0] += 0.3  # one variant is genuinely better everywhere / 변형 하나가 어디서나 정말 낫다
    assert pbo(edge)["pbo"] < 0.05


def test_deflated_sharpe_falls_with_more_trials():
    r = np.random.default_rng(1).normal(0.05, 1.0, 2000)
    few, many = deflated_sharpe(r, 2, 0.001), deflated_sharpe(r, 500, 0.001)
    assert many["dsr"] < few["dsr"]
    assert expected_max_sharpe(500, 0.001) > expected_max_sharpe(10, 0.001) > 0


# reconcile / 거래 대조

def test_reconcile_statuses():
    t0 = pd.Timestamp("2024-01-01")
    exp = pd.DataFrame({"symbol": "A", "side": [1, 1, -1, 1],
                        "entry_time": [t0, t0 + pd.Timedelta("1D"), t0 + pd.Timedelta("2D"), t0 + pd.Timedelta("3D")],
                        "entry_price": [10.0, 11.0, 12.0, 13.0], "stop": [9.0, 10.0, 13.0, 12.0]})
    act = exp.copy()
    act.loc[1, "stop"] = 10.5                                       # differs / 손절이 다름
    act.loc[2, "entry_time"] = act.loc[2, "entry_time"] + pd.Timedelta("4h")  # differs (late) / 늦은 진입
    act = act.drop(index=3)                                         # missing / 빠진 거래
    act = pd.concat([act, exp.iloc[[0]].assign(entry_time=t0 + pd.Timedelta("9D"))])  # extra / 더 생긴 거래
    s = summary(reconcile(exp, act, time_tol="12h"))
    assert (s["match"], s["differs"], s["missing"], s["extra"]) == (1, 2, 1, 1)
    assert s["match_rate"] == pytest.approx(0.25)


# bootstrap and concentration / 부트스트랩과 상위 거래 의존도

def test_bootstrap_bounds_and_t():
    assert bootstrap_mean([1.0, 2.0, 3.0, 4.0], n=500)["p"] == 0.0  # every resample is positive / 다시 뽑아도 모두 양수
    r = np.random.default_rng(2).normal(0.0, 1.0, 400)
    res = bootstrap_mean(r - r.mean(), n=2000, seed=1)  # mean exactly zero / 평균이 정확히 0
    assert 0.3 < res["p"] < 0.7
    x = np.array([1.0, -0.5, 2.0, 0.5])
    assert bootstrap_mean(x, n=10)["t"] == pytest.approx(x.mean() / (x.std(ddof=1) / 2))
    assert np.array_equal(bootstrap_mean(x, n=50, seed=3)["means"], bootstrap_mean(x, n=50, seed=3)["means"])


def test_top_dependence():
    res = top_dependence([10.0, 5.0, 1.0, -2.0], tops=(1, 2))
    table = res["table"].set_index("removed")
    assert table.loc[0, "total_left"] == 14.0
    assert table.loc[1, "total_left"] == 4.0 and table.loc[1, "share_left"] == pytest.approx(4 / 14)
    assert res["negative_after"] == 2  # without 10 and 5, 1 - 2 < 0 / 10과 5를 빼면 1 - 2 < 0
    assert top_dependence([1.0, 1.0])["negative_after"] == 2
