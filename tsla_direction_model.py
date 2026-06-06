import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("tsla_2025.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
# Returns
df["Return_1d"]  = df["Close"].pct_change()
df["Return_2d"]  = df["Close"].pct_change(2)
df["Return_5d"]  = df["Close"].pct_change(5)
df["Return_10d"] = df["Close"].pct_change(10)

# Moving averages
df["SMA_5"]  = df["Close"].rolling(5).mean()
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["SMA_ratio_5_20"]  = df["SMA_5"]  / df["SMA_20"]
df["SMA_ratio_20_50"] = df["SMA_20"] / df["SMA_50"]

# Volatility
df["Volatility_5"]  = df["Return_1d"].rolling(5).std()
df["Volatility_10"] = df["Return_1d"].rolling(10).std()

# Volume
df["Volume_change"]    = df["Volume"].pct_change()
df["Volume_SMA_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()

# Candle
df["High_Low_range"] = (df["High"] - df["Low"]) / df["Close"]
df["Open_Close"]     = (df["Close"] - df["Open"]) / df["Open"]

# RSI (14)
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + gain / loss))

# MACD
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]        = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

# Bollinger Bands
bb_mid        = df["Close"].rolling(20).mean()
bb_std        = df["Close"].rolling(20).std()
df["BB_upper"]    = bb_mid + 2 * bb_std
df["BB_lower"]    = bb_mid - 2 * bb_std
df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
df["BB_width"]    = (df["BB_upper"] - df["BB_lower"]) / bb_mid

# Momentum & ROC
df["Momentum_5"]  = df["Close"] - df["Close"].shift(5)
df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
df["ROC_5"]       = (df["Close"] / df["Close"].shift(5) - 1) * 100
df["ROC_10"]      = (df["Close"] / df["Close"].shift(10) - 1) * 100

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
# 3. CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
split1 = int(len(df) * 0.75)
split2 = int(len(df) * 0.88)

X_train, y_train = X.iloc[:split1],       y.iloc[:split1]
X_val,   y_val   = X.iloc[split1:split2], y.iloc[split1:split2]
X_test,  y_test  = X.iloc[split2:],       y.iloc[split2:]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING (on train split only)
# ─────────────────────────────────────────────────────────────────────────────
param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [3, 4, 5],
    "learning_rate":     [0.03, 0.05, 0.10],
    "subsample":         [0.7, 0.8, 1.0],
    "min_samples_split": [5, 10],
}

tscv = TimeSeriesSplit(n_splits=5)

print("\nTuning hyperparameters … (may take ~30 s)")
gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=tscv,
    scoring="roc_auc",   # AUC is better than accuracy for imbalanced classes
    n_jobs=-1,
    verbose=0,
)
gs.fit(X_train, y_train)
best_params = gs.best_params_
print(f"Best params : {best_params}")
print(f"CV ROC-AUC  : {gs.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN BEST MODEL + PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
base_model = GradientBoostingClassifier(**best_params, random_state=42)
base_model.fit(X_train, y_train)

# Isotonic calibration on validation set → better probability estimates
calibrated = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
calibrated.fit(X_val, y_val)

# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTIMAL THRESHOLD (maximize F1 on validation set)
# ─────────────────────────────────────────────────────────────────────────────
val_probs = calibrated.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.30, 0.71, 0.01)
best_thresh, best_f1 = 0.5, 0.0
for t in thresholds:
    preds_t = (val_probs >= t).astype(int)
    score   = f1_score(y_val, preds_t, zero_division=0)
    if score > best_f1:
        best_f1, best_thresh = score, t

print(f"Optimal threshold (val F1={best_f1:.4f}): {best_thresh:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(name, X_eval, y_eval, threshold):
    probs = calibrated.predict_proba(X_eval)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(f"\n{'='*40}")
    print(f"  {name} set  (threshold={threshold:.2f})")
    print(f"{'='*40}")
    print(f"  Accuracy : {accuracy_score(y_eval, preds):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_eval, probs):.4f}")
    print(classification_report(y_eval, preds, target_names=["Down", "Up"]))
    print("  Confusion matrix:")
    print(confusion_matrix(y_eval, preds))

evaluate("Validation", X_val, y_val, best_thresh)
evaluate("Test",       X_test, y_test, best_thresh)

# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
fi = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": base_model.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\nFeature importances:")
print(fi.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 9. PREDICT ON NEW DATA
# ─────────────────────────────────────────────────────────────────────────────
# new_X must have the same FEATURES columns
#   probs  = calibrated.predict_proba(new_X)[:, 1]   → P(UP)
#   labels = (probs >= best_thresh).astype(int)       → 0 (Down) or 1 (Up)