"""backtest-validation-kit: checks to run before trusting a backtest. See README.md.

[한국어] 백테스트를 믿기 전에 돌려 볼 점검 도구 모음. 설명은 README.ko.md.
"""
from .core import Book, mtm_returns, sharpe
from .data_check import assert_clean, check_bars
from .overfit import deflated_sharpe, expected_max_sharpe, pbo, trial_sharpe_variance
from .random_entry import random_entry_test
from .reconcile import reconcile, summary
from .shuffle import shuffle_bars, shuffle_test
from .trade_removal import removal_test

__all__ = [
    "Book", "mtm_returns", "sharpe",
    "check_bars", "assert_clean",
    "random_entry_test", "shuffle_bars", "shuffle_test", "removal_test",
    "pbo", "deflated_sharpe", "expected_max_sharpe", "trial_sharpe_variance",
    "reconcile", "summary",
]
