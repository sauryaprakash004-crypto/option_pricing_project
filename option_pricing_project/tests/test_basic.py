"""A few basic tests. Run from the project folder with:  python -m pytest -q"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.black_scholes import call_price, put_price
from src.implied_volatility import implied_volatility

S, K, T, R, SIGMA = 100.0, 105.0, 0.5, 0.05, 0.25


def test_call_price_positive():
    assert call_price(S, K, T, R, SIGMA) > 0


def test_put_price_positive():
    assert put_price(S, K, T, R, SIGMA) > 0


def test_put_call_parity():
    # C - P = S - K * exp(-rT)
    lhs = call_price(S, K, T, R, SIGMA) - put_price(S, K, T, R, SIGMA)
    rhs = S - K * np.exp(-R * T)
    assert abs(lhs - rhs) < 1e-8


def test_implied_vol_recovers_original_vol():
    for option_type, pricer in [("call", call_price), ("put", put_price)]:
        price = pricer(S, K, T, R, SIGMA)
        iv = implied_volatility(price, option_type, S, K, T, R)
        assert abs(iv - SIGMA) < 1e-6
