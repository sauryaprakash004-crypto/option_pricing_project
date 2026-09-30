"""Option Pricing and Volatility Analysis using Black-Scholes.

Run with:  python main.py

Pipeline: synthetic option data -> Black-Scholes -> implied volatility -> smile /
surface -> LSTM volatility forecast -> option price with forecasted volatility.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.black_scholes import call_price, put_price, delta, gamma, vega
from src.implied_volatility import implied_volatility

# ------------------------------------------------------------------ settings
R = 0.05                               # risk-free rate (constant, assumed)
SPOT = 100.0                           # spot price (constant, synthetic data)
STRIKES = np.arange(80, 125, 5)        # 80, 85, ..., 120
MATURITIES = [0.25, 0.5, 1.0]          # years
N_OBS = 800
FIG_DIR = "figures"


def price_array(option_types, S, K, T, r, sigma):
    """Black-Scholes price for arrays where each row is a call or a put."""
    return np.where(option_types == "call",
                    call_price(S, K, T, r, sigma),
                    put_price(S, K, T, r, sigma))


# ------------------------------------------------------------ 1. synthetic data
def make_synthetic_data(seed=42):
    """SYNTHETIC options data (not real market data).

    Rows are in time order. Each row is one option. Its true volatility is
        slowly moving base volatility (mean-reverting random series)
        + skew/smile term depending on strike
        + small term-structure term depending on maturity.
    The option price is the Black-Scholes price at that volatility.
    """
    rng = np.random.default_rng(seed)

    base = np.empty(N_OBS)
    base[0] = 0.25
    for t in range(1, N_OBS):
        base[t] = base[t - 1] + 0.05 * (0.25 - base[t - 1]) + 0.01 * rng.standard_normal()
    base = np.clip(base, 0.10, 0.60)

    strike = rng.choice(STRIKES, N_OBS).astype(float)
    maturity = rng.choice(MATURITIES, N_OBS)
    option_type = rng.choice(["call", "put"], N_OBS)

    m = strike / SPOT                                   # moneyness K / S
    vol = base - 0.25 * (m - 1) + 1.0 * (m - 1) ** 2 + 0.04 * (maturity - 0.5)
    vol = np.round(vol, 6)

    price = price_array(option_type, SPOT, strike, maturity, R, vol)

    return pd.DataFrame({
        "spot_price": SPOT,
        "strike": strike,
        "maturity": maturity,
        "option_type": option_type,
        "option_price": np.round(price, 8),
        "volatility": vol,
    })


# ---------------------------------------------------------------------- figures
def fig_price_vs_strike():
    K = np.linspace(70, 130, 100)
    plt.figure(figsize=(7, 4.5))
    plt.plot(K, call_price(SPOT, K, 0.5, R, 0.25), label="Call")
    plt.plot(K, put_price(SPOT, K, 0.5, R, 0.25), label="Put")
    plt.xlabel("Strike")
    plt.ylabel("Black-Scholes price")
    plt.title("Option Price vs Strike (S=100, T=0.5, vol=25%, r=5%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/1_option_price_vs_strike.png", dpi=150)
    plt.close()


def fig_smile(df):
    plt.figure(figsize=(7, 4.5))
    for i, T in enumerate(MATURITIES):
        sub = df[df.maturity == T]
        color = f"C{i}"
        plt.scatter(sub.strike, sub.implied_vol, s=8, alpha=0.2, color=color)
        avg = sub.groupby("strike").implied_vol.mean()
        plt.plot(avg.index, avg.values, marker="o", color=color, label=f"T = {T} yr (average)")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility")
    plt.title("Implied Volatility Smile (synthetic data)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/2_implied_vol_smile.png", dpi=150)
    plt.close()


def fig_surface(df):
    grid = df.pivot_table(index="maturity", columns="strike",
                          values="implied_vol", aggfunc="mean")
    X, Y = np.meshgrid(grid.columns.values, grid.index.values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, grid.values, cmap="viridis")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title("Implied Volatility Surface (synthetic data)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/3_implied_vol_surface.png", dpi=150)
    plt.close()


def fig_actual_vs_pred(actual, pred):
    plt.figure(figsize=(9, 4.5))
    plt.plot(actual, label="Actual volatility")
    plt.plot(pred, label="LSTM predicted volatility")
    plt.xlabel("Test observation")
    plt.ylabel("Volatility")
    plt.title("Actual vs LSTM Predicted Volatility (test set)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/4_actual_vs_lstm_volatility.png", dpi=150)
    plt.close()


def fig_price_comparison(test):
    plt.figure(figsize=(6, 6))
    for kind, color in [("call", "C0"), ("put", "C1")]:
        sub = test[test.option_type == kind]
        plt.scatter(sub.price_original_vol, sub.price_lstm_vol, s=15, alpha=0.7,
                    color=color, label=kind.capitalize())
    lim = [0, test[["price_original_vol", "price_lstm_vol"]].values.max() * 1.05]
    plt.plot(lim, lim, "k--", linewidth=1, label="Equal price")
    plt.xlabel("Price using original volatility")
    plt.ylabel("Price using LSTM predicted volatility")
    plt.title("Option Price: Original vs LSTM Volatility")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/5_price_comparison.png", dpi=150)
    plt.close()


# ------------------------------------------------------------------------ main
def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    results = {}

    # 1. Data
    df = make_synthetic_data()
    df.to_csv("data/options.csv", index=False)
    print(f"Created SYNTHETIC dataset: {len(df)} options -> data/options.csv")
    results["n_observations"] = len(df)

    # 2. Black-Scholes (+ Greeks for one example option: ATM, T = 0.5, vol = 25%)
    print("\nExample ATM option (S=100, K=100, T=0.5, r=5%, vol=25%):")
    print(f"  Call = {call_price(100, 100, 0.5, R, 0.25):.4f}   "
          f"Put = {put_price(100, 100, 0.5, R, 0.25):.4f}")
    print(f"  Call delta = {delta('call', 100, 100, 0.5, R, 0.25):.4f}   "
          f"Gamma = {gamma(100, 100, 0.5, R, 0.25):.4f}   "
          f"Vega = {vega(100, 100, 0.5, R, 0.25):.4f}")
    fig_price_vs_strike()

    # 3. Implied volatility (solve from the option price with brentq)
    df["implied_vol"] = [
        implied_volatility(row.option_price, row.option_type, row.spot_price,
                           row.strike, row.maturity, R)
        for row in df.itertuples()
    ]
    err = (df.implied_vol - df.volatility).abs()
    results["iv_failed_to_solve"] = int(df.implied_vol.isna().sum())
    results["iv_max_abs_error"] = float(err.max())
    results["iv_mean_abs_error"] = float(err.mean())
    print(f"\nImplied vol solved for {df.implied_vol.notna().sum()}/{len(df)} options; "
          f"max |IV - true vol| = {err.max():.2e}")

    # 4. Smile / surface
    smile = df[df.maturity == 0.5].groupby("strike").implied_vol.mean()
    results["avg_iv_T0.5_K80"] = float(smile[80.0])
    results["avg_iv_T0.5_K100"] = float(smile[100.0])
    results["avg_iv_T0.5_K120"] = float(smile[120.0])
    term = df[df.strike == 100.0].groupby("maturity").implied_vol.mean()
    for T in MATURITIES:
        results[f"avg_iv_K100_T{T}"] = float(term[T])
    print("Average IV at T=0.5:  K=80: %.3f   K=100: %.3f   K=120: %.3f"
          % (smile[80.0], smile[100.0], smile[120.0]))
    fig_smile(df)
    fig_surface(df)

    # 5-7. LSTM forecast, evaluation, final pricing (needs PyTorch)
    try:
        from src.lstm_model import train_and_predict
    except ImportError:
        print("\nPyTorch is not installed -> skipping the LSTM stage. Run: pip install torch")
        results["lstm_status"] = "skipped (PyTorch not installed)"
    else:
        print("\nTraining LSTM (previous 20 volatilities -> next volatility):")
        actual, pred, start, _ = train_and_predict(df.volatility.values)
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        results.update({"lstm_status": "ok", "test_observations": len(actual),
                        "lstm_MAE": mae, "lstm_RMSE": rmse})
        print(f"\nTest MAE = {mae:.4f}   Test RMSE = {rmse:.4f}  (volatility units)")
        fig_actual_vs_pred(actual, pred)

        # Re-price the test options with the LSTM-predicted volatility
        test = df.iloc[start:].copy()
        test["pred_vol"] = pred
        test["price_original_vol"] = price_array(
            test.option_type.values, test.spot_price.values, test.strike.values,
            test.maturity.values, R, test.volatility.values)
        test["price_lstm_vol"] = price_array(
            test.option_type.values, test.spot_price.values, test.strike.values,
            test.maturity.values, R, test.pred_vol.values)
        diff = (test.price_lstm_vol - test.price_original_vol).abs()
        results["mean_price_original_vol"] = float(test.price_original_vol.mean())
        results["mean_abs_price_diff"] = float(diff.mean())
        results["mean_abs_price_diff_pct"] = float(100 * diff.mean() / test.price_original_vol.mean())
        print(f"Mean option price (original vol): {results['mean_price_original_vol']:.3f}")
        print(f"Mean |price(LSTM vol) - price(original vol)|: {diff.mean():.3f} "
              f"({results['mean_abs_price_diff_pct']:.1f}% of mean price)")
        fig_price_comparison(test)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results.json and figures in ./figures")


if __name__ == "__main__":
    main()
