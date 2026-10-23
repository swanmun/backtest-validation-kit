"""After many trials, is the best result still meaningful?

Two measures, both from Bailey and López de Prado.

PBO, probability of backtest overfitting (combinatorially symmetric cross-validation):
split the history into n_blocks pieces; for every way of choosing half of them as the
"selection" period, pick the variant with the best Sharpe there and check its rank in the
other half. PBO is how often the selected variant lands below the median. 0.5 means
selecting on past results is no better than picking at random.

    res = pbo(returns)          # returns: periods x variants (DataFrame or 2-D array)
    res["pbo"]

Deflated Sharpe ratio: the probability that the true Sharpe ratio is above zero, after
allowing for the best of n_trials looking good by luck. It needs the variance of the
Sharpe ratios across the trials (per-period units, as from trial_sharpe_variance).

    dsr = deflated_sharpe(best_returns, n_trials=len(trials), sr_variance=trial_sharpe_variance(trials))
    dsr["dsr"]                  # e.g. require >= 0.95, decided before looking

References: Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest Overfitting" (2015);
Bailey, López de Prado, "The Deflated Sharpe Ratio" (2014).

[한국어] 여러 번 시도한 뒤에도 1등이 의미 있는가? 두 지표 모두 Bailey · López de Prado의 방법이다.
PBO(백테스트 과적합 확률, 조합 대칭 교차 검증): 기간을 n_blocks 조각으로 나누고, 그 절반을 '고르는 기간'으로
삼는 모든 경우마다 거기서 샤프가 가장 좋은 변형을 골라, 나머지 절반에서 순위를 본다.
고른 변형이 중간 아래로 떨어지는 비율이 PBO다. 0.5면 과거 성적으로 고르는 것이 아무거나 고르는 것보다 낫지 않다.
보정 샤프(DSR): n_trials개 중 1등이 운으로 좋아 보이는 몫을 뺀 뒤, 진짜 샤프가 0보다 클 확률.
시도들 사이 샤프의 분산이 필요하다(기간 단위, trial_sharpe_variance로 잰다).
예: 결과를 보기 전에 'DSR ≥ 0.95'를 통과 기준으로 정한다.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def pbo(returns, n_blocks: int = 16) -> dict:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("returns must be periods x variants with at least two variants")
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    blocks = np.array_split(np.arange(len(x)), n_blocks)
    count = np.array([len(b) for b in blocks], dtype=float)
    # per-block sums, so each split costs no more than a few adds / 조각별 합을 미리 구해, 나누는 경우마다 더하기 몇 번이면 된다
    s1 = np.array([x[b].sum(0) for b in blocks])
    s2 = np.array([(x[b] ** 2).sum(0) for b in blocks])

    def block_sharpe(sel):
        m_ = count[sel].sum()
        mean = s1[sel].sum(0) / m_
        var = s2[sel].sum(0) / m_ - mean ** 2
        return np.where(var > 1e-18, mean / np.sqrt(np.maximum(var, 1e-18)), 0.0)

    n_var = x.shape[1]
    every = np.arange(n_blocks)
    logits, ranks = [], []
    for chosen in itertools.combinations(range(n_blocks), n_blocks // 2):
        sel = np.array(chosen)
        ins, oos = block_sharpe(sel), block_sharpe(np.setdiff1d(every, sel))
        best = int(np.argmax(ins))
        w = ((oos < oos[best]).sum() + 1) / (n_var + 1)  # relative rank out of sample, 1 = top / 확인 기간의 상대 순위, 1 = 1등
        ranks.append(w)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "oos_rank_median": float(np.median(ranks)),
            "splits": len(logits), "logits": logits}


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Sharpe ratio the best of n_trials would reach by luck alone (per-period units).

    [한국어] n_trials개 중 1등이 운만으로 닿을 샤프(기간 단위).
    """
    if n_trials < 2:
        return 0.0
    return float(np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                                         + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(returns, n_trials: int, sr_variance: float) -> dict:
    """Probability that the true Sharpe is above zero, after allowing for n_trials. / 시도 횟수를 고려한 뒤 진짜 샤프가 0보다 클 확률."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else 0.0
    sk, ku = skew(r), kurtosis(r, fisher=False)  # skewness and (non-excess) kurtosis / 왜도와 첨도(초과 아님)
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    denom = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    return {"dsr": float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom)), "sharpe": float(sr),
            "sharpe_by_luck": sr0, "periods": t, "n_trials": n_trials}


def trial_sharpe_variance(returns) -> float:
    """Variance of the per-period Sharpe ratios across trials (columns).

    [한국어] 시도(열)들 사이에서 기간 단위 샤프의 분산.
    """
    x = pd.DataFrame(returns)
    srs = x.mean() / x.std(ddof=1)
    return float(srs.replace([np.inf, -np.inf], np.nan).dropna().var(ddof=1))
