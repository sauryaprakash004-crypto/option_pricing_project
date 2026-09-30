# Option Pricing and Volatility Analysis using Black-Scholes

## 1. Project objective
A small Python project that prices European options with Black-Scholes, recovers implied volatility from option prices, visualises the volatility smile and surface, and uses a simple LSTM to forecast volatility, which is then fed back into Black-Scholes.

**The data is synthetic** (generated in `main.py`). No real market data is used.

## 2. Methodology
```
Synthetic option data -> Black-Scholes -> Implied volatility -> Smile / surface
                      -> LSTM volatility forecast -> Option price with forecasted volatility
```
- 800 synthetic options in time order (`data/options.csv`): columns `spot_price, strike, maturity, option_type, option_price, volatility`.
- Spot = 100, risk-free rate = 5%, strikes 80-120, maturities 0.25 / 0.5 / 1.0 years.
- True volatility = slowly moving base level + skew/smile in strike + small term-structure effect. Option prices are Black-Scholes prices at that volatility.

## 3. Black-Scholes
`src/black_scholes.py` implements d1, d2, call price, put price, and the Greeks Delta, Gamma and Vega for European options without dividends.

## 4. Implied volatility
`src/implied_volatility.py` solves `BS_price(sigma) = market_price` for sigma using `scipy.optimize.brentq`. Prices below intrinsic value return NaN.

## 5. Volatility smile / surface
- **Smile:** implied volatility plotted against strike for one maturity.
- **Skew:** the smile is tilted; here, low strikes have higher IV than high strikes.
- **Term structure:** how implied volatility changes with maturity.
- **Surface:** strike (x), maturity (y), implied volatility (z), built from average IV per strike/maturity cell. No interpolation or fitted model.

## 6. LSTM volatility forecasting
`src/lstm_model.py` (PyTorch): previous 20 volatility values -> next volatility value.
1 LSTM layer, hidden size 16, Linear output, MSE loss, Adam, 15 epochs. First 80% of windows (in time order) for training, last 20% for testing. The test predictions are plugged back into Black-Scholes to re-price the test options.

## 7. Results
Synthetic data, single run, seed 42.

| Item | Result |
|---|---|
| Options in dataset | 800 |
| Options where IV was solved | 800 / 800 |
| Max abs error, recovered IV vs true vol | 1.4e-09 |
| Average IV, T=0.5, K=80 / 100 / 120 | 0.326 / 0.245 / 0.234 |
| Average IV, K=100, T=0.25 / 0.5 / 1.0 | 0.230 / 0.245 / 0.270 |
| LSTM test MAE / RMSE | written to `results.json` by `main.py` |
| Price change from LSTM vol | written to `results.json` by `main.py` |

The LSTM numbers are produced when you run `python main.py` with PyTorch installed; nothing is hard-coded here.

Figures are saved in `figures/`:
1. `1_option_price_vs_strike.png`
2. `2_implied_vol_smile.png`
3. `3_implied_vol_surface.png`
4. `4_actual_vs_lstm_volatility.png`
5. `5_price_comparison.png`

## 8. Limitations
- Synthetic data: the smile is built in by construction, so it is not evidence about real markets.
- Black-Scholes assumes constant volatility, no dividends, European exercise.
- The LSTM forecasts a single volatility series that mixes the smile effect with the base level; it is a demonstration, not a tuned or benchmarked forecaster (no baseline model comparison).
- One train/test split, one run, no hyperparameter tuning.
- Not a trading model.

## 9. How to run
```bash
pip install -r requirements.txt
python main.py
python -m pytest -q
```
