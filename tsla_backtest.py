import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & ENGINEER FEATURES
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("tsla_2025.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# --- Target: did price go UP the next day? ---
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# --- Returns ---
df["Return_1d"] = df["Close"].pct_change()
df["Return_2d"] = df["Close"].pct_change(2)
df["Return_5d"] = df["Close"].pct_change(5)
df["Return_10d"] = df["Close"].pct_change(10)

# --- Moving averages & ratio ---
df["SMA_5"]    = df["Close"].rolling(5).mean()
df["SMA_20"]   = df["Close"].rolling(20).mean()
df["SMA_50"]   = df["Close"].rolling(50).mean()
df["SMA_ratio_5_20"]  = df["SMA_5"]  / df["SMA_20"]
df["SMA_ratio_20_50"] = df["SMA_20"] / df["SMA_50"]

# --- Volatility ---
df["Volatility_5"]  = df["Return_1d"].rolling(5).std()
df["Volatility_10"] = df["Return_1d"].rolling(10).std()

# --- Volume ---
df["Volume_change"]   = df["Volume"].pct_change()
df["Volume_SMA_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()

# --- Candlestick body features ---
df["High_Low_range"] = (df["High"] - df["Low"]) / df["Close"]
df["Open_Close"]     = (df["Close"] - df["Open"]) / df["Open"]

# --- RSI (14) ---
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + gain / loss))

# --- MACD ---
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]        = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

# --- Bollinger Bands (20, 2σ) ---
bb_mid   = df["Close"].rolling(20).mean()
bb_std   = df["Close"].rolling(20).std()
df["BB_upper"] = bb_mid + 2 * bb_std
df["BB_lower"] = bb_mid - 2 * bb_std
df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
df["BB_width"]    = (df["BB_upper"] - df["BB_lower"]) / bb_mid

# --- Momentum ---
df["Momentum_5"]  = df["Close"] - df["Close"].shift(5)
df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

# --- Rate of Change ---
df["ROC_5"]  = (df["Close"] / df["Close"].shift(5) - 1) * 100
df["ROC_10"] = (df["Close"] / df["Close"].shift(10) - 1) * 100

# Drop NaN rows and the last row (no target)
df = df.dropna().iloc[:-1].reset_index(drop=True)

FEATURES = [
    "Return_1d", "Return_2d", "Return_5d", "Return_10d",
    "SMA_ratio_5_20", "SMA_ratio_20_50",
    "Volatility_5", "Volatility_10",
    "Volume_change", "Volume_SMA_ratio",
    "High_Low_range", "Open_Close",
    "RSI",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_position", "BB_width",
    "Momentum_5", "Momentum_10",
    "ROC_5", "ROC_10",
]

X, y = df[FEATURES], df["Target"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. HYPERPARAMETER TUNING (on training portion only)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_END = int(len(df) * 0.75)   # 75 % for tuning + training
TEST_START = int(len(df) * 0.88)  # last ~12 % for walk-forward test

X_tune, y_tune = X.iloc[:TRAIN_END], y.iloc[:TRAIN_END]

param_grid = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.10],
    "subsample":     [0.7, 0.8, 1.0],
    "min_samples_split": [5, 10],
}

tscv = TimeSeriesSplit(n_splits=5)

print("Tuning hyperparameters with TimeSeriesSplit CV … (this may take ~30 s)")
gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=tscv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
gs.fit(X_tune, y_tune)

best_params = gs.best_params_
print(f"Best params : {best_params}")
print(f"CV accuracy : {gs.best_score_:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. WALK-FORWARD BACKTEST
#    Each test day: retrain on ALL data before that day, predict next day.
#    We batch this into weekly retrains for speed.
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # only trade when model is ≥ 60 % confident
RETRAIN_EVERY        = 5      # retrain model every N test days

test_indices = list(range(TEST_START, len(df)))

predictions  = []
probabilities = []
retrain_dates = []

model = None
for step, idx in enumerate(test_indices):
    # Retrain on expanding window
    if step % RETRAIN_EVERY == 0 or model is None:
        model = GradientBoostingClassifier(**best_params, random_state=42)
        model.fit(X.iloc[:idx], y.iloc[:idx])
        retrain_dates.append(df.loc[idx, "Date"].strftime("%Y-%m-%d"))

    prob = model.predict_proba(X.iloc[[idx]])[0][1]   # P(UP)
    pred = 1 if prob >= CONFIDENCE_THRESHOLD else (0 if prob <= (1 - CONFIDENCE_THRESHOLD) else -1)
    predictions.append(pred)
    probabilities.append(prob)

df_test = df.iloc[TEST_START:].copy().reset_index(drop=True)
df_test["Prediction"]   = predictions
df_test["Probability"]  = probabilities

test_acc = accuracy_score(
    df_test["Target"],
    df_test["Prediction"].replace(-1, 0)   # treat "no signal" as 0 for accuracy
)
print(f"Walk-forward test accuracy : {test_acc:.4f}")
print(f"Retrain events             : {len(retrain_dates)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4. BACKTEST SIMULATION WITH CONFIDENCE-SCALED POSITION SIZING
# ─────────────────────────────────────────────────────────────────────────────
STARTING_BALANCE = 1000.0
MAX_POSITION_PCT = 1.0    # can use up to 100 % of balance
MIN_POSITION_PCT = 0.5    # minimum 50 % allocation when signal fires

balance     = STARTING_BALANCE
in_position = False
buy_price   = 0.0
shares      = 0.0
position_pct = 0.0

trades = []
equity = []

for i, row in df_test.iterrows():
    pred = row["Prediction"]
    prob = row["Probability"]

    if pred == 1 and not in_position:
        # Scale position size with confidence (linear between MIN and MAX)
        position_pct = MIN_POSITION_PCT + (MAX_POSITION_PCT - MIN_POSITION_PCT) * \
                       ((prob - CONFIDENCE_THRESHOLD) / (1.0 - CONFIDENCE_THRESHOLD))
        position_pct = min(position_pct, MAX_POSITION_PCT)

        allocated   = balance * position_pct
        shares      = allocated / row["Close"]
        buy_price   = row["Close"]
        in_position = True

    elif pred == 0 and in_position:
        sell_price = row["Close"]
        proceeds   = shares * sell_price
        cost       = shares * buy_price
        profit     = proceeds - cost
        balance    = balance - (shares * buy_price) + proceeds   # unallocated + proceeds
        trades.append({
            "Sell Date"   : row["Date"].strftime("%Y-%m-%d"),
            "Buy Price"   : round(buy_price, 2),
            "Sell Price"  : round(sell_price, 2),
            "Position %"  : round(position_pct * 100, 1),
            "Confidence"  : round(prob, 3),
            "Profit $"    : round(profit, 2),
            "Return %"    : round((sell_price - buy_price) / buy_price * 100, 2),
        })
        in_position  = False
        shares       = 0.0
        position_pct = 0.0

    # Daily equity = unrealised value of open position + unallocated cash
    unrealised    = shares * row["Close"] if in_position else 0.0
    unallocated   = balance - (shares * buy_price) if in_position else balance
    current_value = unrealised + unallocated
    equity.append({
        "Date"        : row["Date"].strftime("%Y-%m-%d"),
        "Portfolio $" : round(current_value, 2),
    })

# Close open position at period end
if in_position:
    final_price  = df_test.iloc[-1]["Close"]
    proceeds     = shares * final_price
    cost         = shares * buy_price
    profit       = proceeds - cost
    balance      = balance - cost + proceeds
    trades.append({
        "Sell Date"  : df_test.iloc[-1]["Date"].strftime("%Y-%m-%d"),
        "Buy Price"  : round(buy_price, 2),
        "Sell Price" : round(final_price, 2),
        "Position %" : round(position_pct * 100, 1),
        "Confidence" : round(df_test.iloc[-1]["Probability"], 3),
        "Profit $"   : round(profit, 2),
        "Return %"   : round((final_price - buy_price) / buy_price * 100, 2),
    })
    equity[-1]["Portfolio $"] = round(balance, 2)

# ─────────────────────────────────────────────────────────────────────────────
# 5. RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────
trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity)
equity_df["Portfolio $"] = equity_df["Portfolio $"].astype(float)

# Daily returns from equity curve
eq_vals = equity_df["Portfolio $"].values
daily_rets = np.diff(eq_vals) / eq_vals[:-1]

sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
          if np.std(daily_rets) > 0 else 0.0)

# Max drawdown
peak    = np.maximum.accumulate(eq_vals)
dd      = (eq_vals - peak) / peak
max_dd  = dd.min()

total_return   = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
buy_hold_return = (
    (df_test.iloc[-1]["Close"] - df_test.iloc[0]["Close"])
    / df_test.iloc[0]["Close"] * 100
)

winning_trades = len(trades_df[trades_df["Profit $"] > 0]) if len(trades_df) else 0
losing_trades  = len(trades_df[trades_df["Profit $"] <= 0]) if len(trades_df) else 0
win_rate       = round(winning_trades / len(trades_df) * 100, 1) if len(trades_df) else 0
best_trade     = trades_df["Return %"].max() if len(trades_df) else 0
worst_trade    = trades_df["Return %"].min() if len(trades_df) else 0
avg_trade      = trades_df["Return %"].mean() if len(trades_df) else 0
no_signal_days = (df_test["Prediction"] == -1).sum()

calmar = (total_return / 100) / abs(max_dd) if max_dd != 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 6. PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 52

print(SEP)
print("       FINE-TUNED BACKTEST RESULTS (Test Period)")
print(SEP)
print(f"  Starting balance    : ${STARTING_BALANCE:,.2f}")
print(f"  Final balance       : ${balance:,.2f}")
print(f"  Total return        : {total_return:.2f}%")
print(f"  Buy & hold return   : {buy_hold_return:.2f}%")
print(f"  Alpha               : {total_return - buy_hold_return:.2f}%")
print("-" * 52)
print(f"  Sharpe ratio (ann.) : {sharpe:.3f}")
print(f"  Max drawdown        : {max_dd * 100:.2f}%")
print(f"  Calmar ratio        : {calmar:.3f}")
print("-" * 52)
print(f"  Total trades        : {len(trades_df)}")
print(f"  Winning trades      : {winning_trades}")
print(f"  Losing trades       : {losing_trades}")
print(f"  Win rate            : {win_rate}%")
print(f"  Avg trade return    : {avg_trade:.2f}%")
print(f"  Best trade          : +{best_trade:.2f}%")
print(f"  Worst trade         : {worst_trade:.2f}%")
print(f"  No-signal days      : {no_signal_days}")
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(SEP)

print(f"\nBest hyperparameters found:\n  {best_params}")

if len(trades_df):
    print("\nAll trades:")
    print(trades_df.to_string(index=False))

print("\nEquity curve (first & last 5 days):")
print(equity_df.head(5).to_string(index=False))
print("  ...")
print(equity_df.tail(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
fi = pd.DataFrame({
    "Feature"   : FEATURES,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\nTop-10 feature importances (last trained model):")
print(fi.head(10).to_string(index=False))