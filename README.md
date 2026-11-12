# Taxi Demand Forecasting

Forecasts the number of taxi orders per hour so a fleet can put enough cars on the road at the right time. A Random Forest with time-series features beats a naive baseline and the target error.

## Results
| Model | RMSE |
|---|---|
| Baseline (predict the previous hour) | 58.00 |
| Random Forest (tuned) | 37.65 |

The goal was an RMSE below 40. The final model also reaches an MAE of 25.94 and an R-squared of 0.598 on a held-out, time-ordered test set (the last 10% of the data, never shuffled).

## Approach
1. Load 10-minute order counts and resample them to hourly (and 30-minute) intervals
2. Decompose the series to see trend, daily seasonality and rush-hour patterns
3. Build features: hour, day of week, weekend flag, lag features (including 1-week and 2-week lags) and rolling means
4. Evaluate a naive baseline, then tune a Random Forest with `RandomizedSearchCV` and `TimeSeriesSplit` so validation never uses future data

## Data
`taxi_2024.csv` has 26,496 rows of order counts at 10-minute intervals from 1 March to 31 August 2024.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook taxi_orders_random_forest.ipynb
```

## Tech stack
Python, pandas, NumPy, scikit-learn, statsmodels, Matplotlib
