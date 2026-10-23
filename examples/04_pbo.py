"""Pick the best of 80 variants: does the winner stay a winner?

80 settings of the toy strategy on random walks, so no setting has a real edge anywhere.
For each market, choose the best setting on half the history and check it on the other
half, over every way of splitting 16 blocks in two. PBO is how often the chosen setting
lands below the median.

Two things to see: averaged over markets, PBO sits near 0.5 (selecting on the past is no
better than a random pick), and on any one market it can land far from 0.5 in either
direction. One PBO is an estimate with its own noise; read it with the other tests.

[한국어] 변형 80개 중 1등을 고르면, 그 1등은 계속 1등인가?
무작위 행보 위의 장난감 전략 설정 80개라, 어느 설정에도 진짜 우위는 없다.
시장마다 기간의 절반에서 1등 설정을 고르고 나머지 절반에서 확인하기를, 16조각을 둘로 나누는 모든 경우에 대해 한다.
PBO는 고른 설정이 중간 아래로 떨어지는 비율이다.
볼 것 두 가지: 시장들을 평균하면 PBO는 0.5 가까이(과거로 고르는 것이 아무거나 고르는 것보다 낫지 않다)이고,
시장 하나만 보면 0.5에서 어느 쪽으로든 멀리 떨어질 수 있다. PBO 하나도 잡음이 있는 추정치이니 다른 시험과 함께 읽는다.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd

from toy_strategy import ma_breakout
from vkit import Book, pbo
from vkit.synth import random_walk

grid = [(ma, lb) for ma in (10, 20, 30, 40, 50, 70, 90, 120, 160, 200) for lb in (5, 10, 15, 20, 30, 40, 60, 80)]
values = []
for seed in range(20, 26):
    prices = random_walk(n_bars=12000, seed=seed)  # a new market with no edge / 우위가 없는 새 시장
    curves = pd.DataFrame({g: Book(ma_breakout(prices, *g), prices).returns() for g in grid})  # days x variants / 날짜 × 변형
    res = pbo(curves, n_blocks=16)
    values.append(res["pbo"])
    print(f"market {seed}: PBO = {res['pbo']:.2f}, "
          f"median out-of-sample rank of the in-sample winner {res['oos_rank_median']:.2f} (1 = top)")
print(f"\nmean PBO {np.mean(values):.2f} over {len(values)} markets with no edge "
      f"({len(grid)} variants, {res['splits']:,} splits each)")
