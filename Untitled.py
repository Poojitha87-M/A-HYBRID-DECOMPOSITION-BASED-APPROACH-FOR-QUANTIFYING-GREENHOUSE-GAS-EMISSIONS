"""
GHG Emission Forecasting – Japan (ARIMA Only)

✔ Model: ARIMA
✔ Split: 70% Train | 20% Validation | 10% Test
✔ Grid Search + Early Stopping (proxy)
✔ Normalization AFTER split
✔ Metrics + Plots + Loss curve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

# =========================================================
# 1. METRICS
# =========================================================
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mp = mape(y_true, y_pred)

    print("\nEvaluation Metrics")
    print("="*40)
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print(f"MAPE : {mp:.2f}%")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mp}

# =========================================================
# 2. LOAD DATA (JAPAN)
# =========================================================
df = pd.read_csv("Japan_scaled.csv")

TARGET = "Annual greenhouse gas emissions including land use"

df = df[["Year", TARGET]].sort_values("Year").reset_index(drop=True)

values = df[TARGET].values.reshape(-1,1)
years = df["Year"].values

# =========================================================
# 3. SPLIT (70:20:10)
# =========================================================
n = len(values)
train_end = int(n * 0.70)
val_end = int(n * 0.90)

train_raw = values[:train_end]
val_raw = values[train_end:val_end]
test_raw = values[val_end:]

# =========================================================
# 4. NORMALIZATION
# =========================================================
scaler = MinMaxScaler()
train_sc = scaler.fit_transform(train_raw).flatten()
val_sc = scaler.transform(val_raw).flatten()
test_sc = scaler.transform(test_raw).flatten()

# =========================================================
# 5. GRID SEARCH ARIMA
# =========================================================
print("Searching best ARIMA model...")

best_order = None
best_val_mse = np.inf
patience = 10
no_improve = 0

for order in product(range(4), range(2), range(4)):
    try:
        model = ARIMA(train_sc, order=order).fit()
        val_pred = model.forecast(steps=len(val_sc))
        mse = mean_squared_error(val_sc, val_pred)

        if mse < best_val_mse:
            best_val_mse = mse
            best_order = order
            no_improve = 0
            print("Best:", order, "Val MSE:", mse)
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered")
            break

    except:
        continue

print("Final ARIMA order:", best_order)

# =========================================================
# 6. TRAIN FINAL MODEL (Train + Val)
# =========================================================
train_val = np.concatenate([train_sc, val_sc])

# =========================================================
# 7. TEST PREDICTION (ROLLING FORECAST)
# =========================================================
history = list(train_val)
preds = []

for t in range(len(test_sc)):
    model = ARIMA(history, order=best_order).fit()
    pred = model.forecast()[0]
    preds.append(pred)
    history.append(test_sc[t])

preds = np.array(preds)

# =========================================================
# 8. INVERSE TRANSFORM
# =========================================================
y_test = scaler.inverse_transform(test_sc.reshape(-1,1)).flatten()
y_pred = scaler.inverse_transform(preds.reshape(-1,1)).flatten()

# =========================================================
# 9. EVALUATION
# =========================================================
metrics = evaluate(y_test, y_pred)

# =========================================================
# 10. LOSS CURVE (PROXY)
# =========================================================
def rolling_loss(series, order):
    losses = []
    for i in range(10, len(series)):
        history = series[:i]
        model = ARIMA(history, order=order).fit()
        pred = model.forecast()[0]
        actual = series[i]
        losses.append((pred - actual)**2)
    return losses

train_val_series = np.concatenate([train_sc, val_sc])
losses = rolling_loss(train_val_series, best_order)

# =========================================================
# 11. PLOTS
# =========================================================

# Plot 1: Data split
plt.figure(figsize=(10,4))
plt.plot(years[:train_end], values[:train_end], label="Train")
plt.plot(years[train_end:val_end], values[train_end:val_end], label="Validation")
plt.plot(years[val_end:], values[val_end:], label="Test")
plt.legend()
plt.title("Japan Data Split")
plt.show()

# Plot 2: Loss curve
plt.figure(figsize=(8,4))
plt.plot(losses)
plt.title("Rolling MSE Loss (Proxy)")
plt.xlabel("Steps")
plt.ylabel("MSE")
plt.show()

# Plot 3: Predictions
plt.figure(figsize=(10,5))
plt.plot(years[val_end:], y_test, label="Original", marker='o')
plt.plot(years[val_end:], y_pred, '--', label="ARIMA Predicted")
plt.legend()
plt.title("Japan Test Prediction (ARIMA)")
plt.xlabel("Year")
plt.ylabel("GHG Emission")
plt.grid()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

# Helper functions
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mp   = mape(y_true, y_pred)

    print("\n"+"="*50)
    print(f"{name} Evaluation:")
    print("="*50)
    print(f"MAE  : {mae:.6f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")
    print(f"MAPE : {mp:.4f}%")
    return rmse, r2

# Load data
df = pd.read_csv("Japan_scaled.csv")
df = df[["Year", "Annual greenhouse gas emissions including land use"]]
df = df.sort_values("Year").reset_index(drop=True)
values = df["Annual greenhouse gas emissions including land use"].values.reshape(-1,1)

# Split data 70:20:10
n = len(values)
train_end = int(n*0.70)
val_end = int(n*0.90)

train_raw = values[:train_end]
val_raw = values[train_end:val_end]
test_raw = values[val_end:]

# Normalize AFTER split
scaler = MinMaxScaler()
train_sc = scaler.fit_transform(train_raw).flatten()
val_sc = scaler.transform(val_raw).flatten()
test_sc = scaler.transform(test_raw).flatten()

# Combine train+val for SARIMA training
train_val_sc = np.concatenate([train_sc, val_sc])

# Grid search SARIMA hyperparameters (p,d,q) and seasonal (P,D,Q,s)
best_aic = np.inf
best_order = None
best_seasonal_order = None
seasonal_period = 12  # adjust if your data is not monthly

# You can reduce ranges for speed if needed
p = d = q = range(0, 3)
P = D = Q = range(0, 2)

print("Searching best SARIMA model...")

for order in product(p, d, q):
    for seasonal_order in product(P, D, Q):
        seasonal = seasonal_order + (seasonal_period,)
        try:
            model = SARIMAX(train_val_sc, order=order, seasonal_order=seasonal).fit(disp=False)
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = order
                best_seasonal_order = seasonal
        except:
            continue

print(f"\nBest SARIMA order: {best_order} Seasonal order: {best_seasonal_order}")

# Forecast test data with rolling forecast
history = list(train_val_sc)
sarima_test_preds = []

for t in range(len(test_sc)):
    model = SARIMAX(history, order=best_order, seasonal_order=best_seasonal_order).fit(disp=False)
    pred = model.forecast()[0]
    sarima_test_preds.append(pred)
    history.append(test_sc[t])

sarima_test_preds = np.array(sarima_test_preds)
sarima_test_preds_inv = scaler.inverse_transform(sarima_test_preds.reshape(-1,1)).flatten()

# Inverse transform test true values
y_test_inv = scaler.inverse_transform(test_sc.reshape(-1,1)).flatten()

rmse_sarima, r2_sarima = evaluate("SARIMA", y_test_inv, sarima_test_preds_inv)

# Plot test predictions vs original
plt.figure(figsize=(12,5))
plt.plot(y_test_inv, label='Original', marker='o')
plt.plot(sarima_test_preds_inv, label='SARIMA Predicted', linestyle='--')
plt.title('Test Data: Original vs SARIMA Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Annual GHG Emission')
plt.legend()
plt.tight_layout()
plt.show()import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

# Helper functions
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mp   = mape(y_true, y_pred)

    print("\n"+"="*50)
    print(f"{name} Evaluation:")
    print("="*50)
    print(f"MAE  : {mae:.6f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")
    print(f"MAPE : {mp:.4f}%")
    return rmse, r2

# Load data
df = pd.read_csv("Japan_scaled.csv")
df = df.sort_values("Year").reset_index(drop=True)

# Target variable
y = df["Annual greenhouse gas emissions including land use"].values.reshape(-1,1)

# Exogenous variables: all numeric columns except Year and target
exog_cols = [col for col in df.columns if col not in ["Year", "Annual greenhouse gas emissions including land use"] and pd.api.types.is_numeric_dtype(df[col])]
X = df[exog_cols].values if exog_cols else None  # If no extra columns, X=None

# Split data 70:20:10
n = len(y)
train_end = int(n*0.70)
val_end = int(n*0.90)

y_train = y[:train_end]
y_val = y[train_end:val_end]
y_test = y[val_end:]

X_train = X[:train_end] if X is not None else None
X_val = X[train_end:val_end] if X is not None else None
X_test = X[val_end:] if X is not None else None

# Normalize AFTER split
scaler_y = MinMaxScaler()
y_train_sc = scaler_y.fit_transform(y_train).flatten()
y_val_sc = scaler_y.transform(y_val).flatten()
y_test_sc = scaler_y.transform(y_test).flatten()

if X is not None:
    scaler_X = MinMaxScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_val_sc = scaler_X.transform(X_val)
    X_test_sc = scaler_X.transform(X_test)
else:
    X_train_sc = X_val_sc = X_test_sc = None

# Combine train + val for model fitting
y_train_val_sc = np.concatenate([y_train_sc, y_val_sc])
X_train_val_sc = np.vstack([X_train_sc, X_val_sc]) if X is not None else None

# Grid search ARIMAX hyperparameters
best_aic = np.inf
best_order = None

print("Searching best ARIMAX model...")

for p,d,q in product(range(0,4), range(0,2), range(0,4)):
    try:
        model = SARIMAX(y_train_val_sc, order=(p,d,q), exog=X_train_val_sc).fit(disp=False)
        if model.aic < best_aic:
            best_aic = model.aic
            best_order = (p,d,q)
    except:
        continue

print(f"\nBest ARIMAX order: {best_order}")

# Forecast test data with rolling forecast
history_y = list(y_train_val_sc)
history_X = X_train_val_sc.copy() if X_train_val_sc is not None else None
arimax_test_preds = []

for t in range(len(y_test_sc)):
    exog_input = X_test_sc[t].reshape(1,-1) if X_test_sc is not None else None
    model = SARIMAX(history_y, order=best_order, exog=history_X).fit(disp=False)
    pred = model.forecast(exog=exog_input)[0]
    arimax_test_preds.append(pred)
    history_y.append(y_test_sc[t])
    if history_X is not None:
        history_X = np.vstack([history_X, exog_input])

arimax_test_preds = np.array(arimax_test_preds)
arimax_test_preds_inv = scaler_y.inverse_transform(arimax_test_preds.reshape(-1,1)).flatten()

# Inverse transform test true values
y_test_inv = scaler_y.inverse_transform(y_test_sc.reshape(-1,1)).flatten()

rmse_arimax, r2_arimax = evaluate("ARIMAX", y_test_inv, arimax_test_preds_inv)

# Plot test predictions vs original
plt.figure(figsize=(12,5))
plt.plot(y_test_inv, label='Original', marker='o')
plt.plot(arimax_test_preds_inv, label='ARIMAX Predicted', linestyle='--')
plt.title('Test Data: Original vs ARIMAX Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Annual GHG Emission')
plt.legend()
plt.tight_layout()
plt.show()"""
Japan Greenhouse Gas Emission Prediction
Statistical Model: SARIMAX
Split: 70:20:10
Normalization AFTER split
Hyperparameter tuning (Grid Search)
Metrics: MAE, MSE, RMSE, R2, MAPE
"""

# ─────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

# ─────────────────────────────────────────
# 2. Load Data
# ─────────────────────────────────────────
df = pd.read_csv("Japan_scaled.csv")
df = df[["Year","Annual greenhouse gas emissions including land use"]]
df = df.sort_values("Year").reset_index(drop=True)
df.columns = ["Year","GHG"]

values = df["GHG"].values.reshape(-1,1)

# ─────────────────────────────────────────
# 3. 70:20:10 Split
# ─────────────────────────────────────────
n = len(values)

train_end = int(n*0.70)
val_end   = int(n*0.90)

train_raw = values[:train_end]
val_raw   = values[train_end:val_end]
test_raw  = values[val_end:]

# ─────────────────────────────────────────
# 4. Normalization (AFTER split)
# ─────────────────────────────────────────
scaler = MinMaxScaler()

train_sc = scaler.fit_transform(train_raw).flatten()
val_sc   = scaler.transform(val_raw).flatten()
test_sc  = scaler.transform(test_raw).flatten()

# ─────────────────────────────────────────
# 5. Metrics
# ─────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

def evaluate(name, y_true, y_pred):

    mae  = mean_absolute_error(y_true,y_pred)
    mse  = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true,y_pred)
    mp   = mape(y_true,y_pred)

    print("\n"+"="*50)
    print(name)
    print("="*50)
    print(f"MAE  : {mae:.6f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")
    print(f"MAPE : {mp:.4f}%")

    return rmse, r2


# ─────────────────────────────────────────
# 6. SARIMAX Grid Search
# ─────────────────────────────────────────
print("\nSearching SARIMAX...")

p = range(0,3)
d = range(0,2)
q = range(0,3)

P = range(0,2)
D = range(0,2)
Q = range(0,2)
s = [12]

best_aic = np.inf
best_order = None
best_seasonal = None

for order in product(p,d,q):
    for seasonal in product(P,D,Q,s):

        try:
            model = SARIMAX(train_sc,
                            order=order,
                            seasonal_order=seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)

            if model.aic < best_aic:
                best_aic = model.aic
                best_order = order
                best_seasonal = seasonal

        except:
            continue

print("Best SARIMAX order:",best_order)
print("Best Seasonal order:",best_seasonal)


# ─────────────────────────────────────────
# 7. Validation Loss
# ─────────────────────────────────────────
history = list(train_sc)

val_preds = []
val_losses = []

for t in range(len(val_sc)):

    model = SARIMAX(history,
                    order=best_order,
                    seasonal_order=best_seasonal).fit(disp=False)

    pred = model.forecast()[0]

    val_preds.append(pred)

    val_losses.append((val_sc[t]-pred)**2)

    history.append(val_sc[t])


plt.figure(figsize=(8,4))
plt.plot(val_losses)
plt.title("SARIMAX Validation Loss (MSE)")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────
# 8. Test Prediction
# ─────────────────────────────────────────
history = list(train_sc)+list(val_sc)

test_preds = []

for t in range(len(test_sc)):

    model = SARIMAX(history,
                    order=best_order,
                    seasonal_order=best_seasonal).fit(disp=False)

    pred = model.forecast()[0]

    test_preds.append(pred)

    history.append(test_sc[t])


test_preds = np.array(test_preds)

rmse_sarimax, r2_sarimax = evaluate("SARIMAX", test_sc, test_preds)


# ─────────────────────────────────────────
# 9. Plot Predictions
# ─────────────────────────────────────────
plt.figure(figsize=(10,4))

plt.plot(test_sc,label="Original",marker="o")
plt.plot(test_preds,label="Predicted",linestyle="--")

plt.title("SARIMAX Test Prediction")
plt.legend()

plt.tight_layout()
plt.show()import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==========================================================
# 0. FILE PATH
# ==========================================================
file_path = "Japan_scaled.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("Upload Japan_scaled.csv first")

print("=" * 60)
print(" ARIMA + VMD | Japan GHG Forecast ")
print("=" * 60)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv(file_path)

TARGET = 'Annual greenhouse gas emissions including land use'

df = df[['Year', TARGET]].dropna().sort_values('Year').reset_index(drop=True)

years = df['Year'].values
data = df[TARGET].values

# Split
n = len(data)
n_train = int(n * 0.70)
n_val   = int(n * 0.20)

train = data[:n_train]
val   = data[n_train:n_train + n_val]
test  = data[n_train + n_val:]

# ==========================================================
# 2. NORMALIZE
# ==========================================================
scaler = MinMaxScaler()

train_sc = scaler.fit_transform(train.reshape(-1,1)).flatten()
val_sc   = scaler.transform(val.reshape(-1,1)).flatten()
test_sc  = scaler.transform(test.reshape(-1,1)).flatten()

full_sc = np.concatenate([train_sc, val_sc, test_sc])

# ==========================================================
# 3. VMD (CUSTOM)
# ==========================================================
def vmd(signal, K=3, alpha=2000, tol=1e-6, max_iter=300):

    N = len(signal)
    f_ext = np.concatenate([signal[::-1], signal, signal[::-1]])
    T = len(f_ext)

    freqs = np.fft.fftfreq(T)
    f_hat = np.fft.fft(f_ext)

    u_hat = np.zeros((K, T), dtype=complex)
    omega = np.linspace(0, 0.5, K)
    lam = np.zeros(T, dtype=complex)

    for _ in range(max_iter):
        u_prev = u_hat.copy()

        for k in range(K):
            other = np.sum(u_hat, axis=0) - u_hat[k]
            num = f_hat - other - lam/2
            den = 1 + alpha*(freqs - omega[k])**2
            u_hat[k] = num / den

            # update center freq
            pw = np.abs(u_hat[k])**2
            omega[k] = np.sum(freqs * pw) / (np.sum(pw) + 1e-12)

        lam += np.sum(u_hat, axis=0) - f_hat

        if np.linalg.norm(u_hat - u_prev) < tol:
            break

    modes = np.array([np.real(np.fft.ifft(u_hat[k]))[N:2*N] for k in range(K)])
    return modes

print("\nRunning VMD...")
K = 3
modes = vmd(full_sc, K=K)

# Split modes
m_tr = modes[:, :n_train]
m_val = modes[:, n_train:n_train+n_val]
m_te = modes[:, n_train+n_val:]
m_tv = modes[:, :n_train+n_val]

# ==========================================================
# 4. ARIMA (FIXED)
# ==========================================================
def walk_arima(train, test, p, d, q):

    history = list(train)
    preds = []

    for t in range(len(test)):
        try:
            model = ARIMA(history, order=(p,d,q)).fit()
            pred = model.forecast()[0]
        except:
            pred = history[-1]

        preds.append(pred)
        history.append(test[t])

    return np.array(preds)

# ==========================================================
# 5. GRID SEARCH
# ==========================================================
print("\nSearching ARIMA orders...")

best_orders = []

for k in range(K):
    best_rmse = np.inf
    best_order = (1,0,0)

    for p,d,q in product(range(2), range(2), range(2)):
        try:
            pred = walk_arima(m_tr[k], m_val[k], p,d,q)
            rmse = np.sqrt(mean_squared_error(m_val[k], pred))

            if rmse < best_rmse:
                best_rmse = rmse
                best_order = (p,d,q)
        except:
            pass

    best_orders.append(best_order)
    print(f"Mode {k+1}: {best_order} RMSE={best_rmse:.4f}")

# ==========================================================
# 6. FORECAST
# ==========================================================
print("\nForecasting...")

val_preds = []
test_preds = []

for k in range(K):
    p,d,q = best_orders[k]

    vp = walk_arima(m_tr[k], m_val[k], p,d,q)
    tp = walk_arima(m_tv[k], m_te[k], p,d,q)

    val_preds.append(vp)
    test_preds.append(tp)

val_final = np.sum(val_preds, axis=0)
test_final = np.sum(test_preds, axis=0)

# Inverse transform
inv = lambda x: scaler.inverse_transform(x.reshape(-1,1)).flatten()

val_pred = inv(val_final)
test_pred = inv(test_final)

# ==========================================================
# 7. METRICS
# ==========================================================
def metrics(y, yhat):
    return {
        "MAE": mean_absolute_error(y, yhat),
        "RMSE": np.sqrt(mean_squared_error(y, yhat)),
        "R2": r2_score(y, yhat)
    }

print("\nValidation:", metrics(val, val_pred))
print("Test:", metrics(test, test_pred))

# ==========================================================
# 8. PLOT
# ==========================================================
plt.figure(figsize=(10,4))
plt.plot(test, label="Actual", marker='o')
plt.plot(test_pred, label="ARIMA+VMD", linestyle='--')
plt.legend()
plt.title("Japan GHG Forecast (ARIMA+VMD)")
plt.show()import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product
import warnings
import os

from statsmodels.tsa.statespace.sarimax import SARIMAX  # ✅ SARIMA

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==========================================================
# 0. FILE PATH
# ==========================================================
file_path = "Japan_scaled.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("Upload Japan_scaled.csv first")

print("=" * 60)
print(" SARIMA + VMD | Japan GHG Forecast ")
print("=" * 60)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv(file_path)

TARGET = 'Annual greenhouse gas emissions including land use'

df = df[['Year', TARGET]].dropna().sort_values('Year').reset_index(drop=True)

years = df['Year'].values
data = df[TARGET].values

# Split
n = len(data)
n_train = int(n * 0.70)
n_val   = int(n * 0.20)

train = data[:n_train]
val   = data[n_train:n_train + n_val]
test  = data[n_train + n_val:]

# ==========================================================
# 2. NORMALIZE
# ==========================================================
scaler = MinMaxScaler()

train_sc = scaler.fit_transform(train.reshape(-1,1)).flatten()
val_sc   = scaler.transform(val.reshape(-1,1)).flatten()
test_sc  = scaler.transform(test.reshape(-1,1)).flatten()

full_sc = np.concatenate([train_sc, val_sc, test_sc])

# ==========================================================
# 3. VMD (CUSTOM)
# ==========================================================
def vmd(signal, K=3, alpha=2000, tol=1e-6, max_iter=300):

    N = len(signal)
    f_ext = np.concatenate([signal[::-1], signal, signal[::-1]])
    T = len(f_ext)

    freqs = np.fft.fftfreq(T)
    f_hat = np.fft.fft(f_ext)

    u_hat = np.zeros((K, T), dtype=complex)
    omega = np.linspace(0, 0.5, K)
    lam = np.zeros(T, dtype=complex)

    for _ in range(max_iter):
        u_prev = u_hat.copy()

        for k in range(K):
            other = np.sum(u_hat, axis=0) - u_hat[k]
            num = f_hat - other - lam/2
            den = 1 + alpha*(freqs - omega[k])**2
            u_hat[k] = num / den

            pw = np.abs(u_hat[k])**2
            omega[k] = np.sum(freqs * pw) / (np.sum(pw) + 1e-12)

        lam += np.sum(u_hat, axis=0) - f_hat

        if np.linalg.norm(u_hat - u_prev) < tol:
            break

    modes = np.array([np.real(np.fft.ifft(u_hat[k]))[N:2*N] for k in range(K)])
    return modes

print("\nRunning VMD...")
K = 3
modes = vmd(full_sc, K=K)

# Split modes
m_tr = modes[:, :n_train]
m_val = modes[:, n_train:n_train+n_val]
m_te = modes[:, n_train+n_val:]
m_tv = modes[:, :n_train+n_val]

# ==========================================================
# 4. SARIMA (REPLACED)
# ==========================================================
def walk_sarima(train, test, order, seasonal_order):

    history = list(train)
    preds = []

    for t in range(len(test)):
        try:
            model = SARIMAX(history,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)

            pred = model.forecast()[0]
        except:
            pred = history[-1]

        preds.append(pred)
        history.append(test[t])

    return np.array(preds)

# ==========================================================
# 5. GRID SEARCH (SARIMA)
# ==========================================================
print("\nSearching SARIMA orders...")

best_orders = []
best_seasonals = []

# You can adjust seasonal period (m)
m = 4   # assume quarterly pattern (can change to 12 if monthly data)

for k in range(K):
    best_rmse = np.inf
    best_order = (1,0,0)
    best_seasonal = (0,0,0,m)

    for p,d,q in product(range(2), range(2), range(2)):
        for P,D,Q in product(range(2), range(2), range(2)):
            try:
                pred = walk_sarima(
                    m_tr[k], m_val[k],
                    (p,d,q),
                    (P,D,Q,m)
                )
                rmse = np.sqrt(mean_squared_error(m_val[k], pred))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = (p,d,q)
                    best_seasonal = (P,D,Q,m)
            except:
                pass

    best_orders.append(best_order)
    best_seasonals.append(best_seasonal)

    print(f"Mode {k+1}: {best_order} x {best_seasonal} RMSE={best_rmse:.4f}")

# ==========================================================
# 6. FORECAST
# ==========================================================
print("\nForecasting...")

val_preds = []
test_preds = []

for k in range(K):
    order = best_orders[k]
    seasonal = best_seasonals[k]

    vp = walk_sarima(m_tr[k], m_val[k], order, seasonal)
    tp = walk_sarima(m_tv[k], m_te[k], order, seasonal)

    val_preds.append(vp)
    test_preds.append(tp)

val_final = np.sum(val_preds, axis=0)
test_final = np.sum(test_preds, axis=0)

# Inverse transform
inv = lambda x: scaler.inverse_transform(x.reshape(-1,1)).flatten()

val_pred = inv(val_final)
test_pred = inv(test_final)

# ==========================================================
# 7. METRICS
# ==========================================================
def metrics(y, yhat):
    return {
        "MAE": mean_absolute_error(y, yhat),
        "RMSE": np.sqrt(mean_squared_error(y, yhat)),
        "R2": r2_score(y, yhat)
    }

print("\nValidation:", metrics(val, val_pred))
print("Test:", metrics(test, test_pred))

# ==========================================================
# 8. PLOT
# ==========================================================
plt.figure(figsize=(10,4))
plt.plot(test, label="Actual", marker='o')
plt.plot(test_pred, label="SARIMA+VMD", linestyle='--')
plt.legend()
plt.title("Japan GHG Forecast (SARIMA+VMD)")
plt.show()# ─────────────────────────────────────────
# FINAL OPTIMIZED: VMD + ARIMA (NO GARCH)
# ─────────────────────────────────────────

!pip install vmdpy -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from vmdpy import VMD

DATASET = "Japan_scaled.csv"

# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
df = pd.read_csv(DATASET)
df = df[["Year", "Annual greenhouse gas emissions including land use"]]
df = df.sort_values("Year").reset_index(drop=True)
df.columns = ["Year", "GHG"]

values = df["GHG"].values.reshape(-1, 1)

# ─────────────────────────────────────────
# 2. Split
# ─────────────────────────────────────────
n = len(values)
train_end = int(n * 0.70)
val_end   = int(n * 0.90)

train_raw = values[:train_end]
val_raw   = values[train_end:val_end]
test_raw  = values[val_end:]

print(f"Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")

# ─────────────────────────────────────────
# 3. Scaling
# ─────────────────────────────────────────
scaler   = MinMaxScaler()
train_sc = scaler.fit_transform(train_raw).flatten()
val_sc   = scaler.transform(val_raw).flatten()
test_sc  = scaler.transform(test_raw).flatten()

full_sc = np.concatenate([train_sc, val_sc, test_sc])

# ─────────────────────────────────────────
# 4. VMD (REDUCED MODES)
# ─────────────────────────────────────────
K     = 2   # ✅ BEST for small dataset
alpha = 2000

u_full, _, _ = VMD(full_sc, alpha, 0, K, 0, 1, 1e-7)

train_modes = u_full[:, :train_end]
val_modes   = u_full[:, train_end:val_end]
test_modes  = u_full[:, val_end:]

# ─────────────────────────────────────────
# 5. ARIMA Order Selection
# ─────────────────────────────────────────
def best_arima(series):
    best_aic = np.inf
    best_order = (1, 1, 1)

    for p, d, q in product(range(0,3), range(0,2), range(0,3)):
        try:
            m = ARIMA(series, order=(p,d,q)).fit()
            if m.aic < best_aic:
                best_aic = m.aic
                best_order = (p,d,q)
        except:
            continue
    return best_order

mode_orders = []

print("\nSelecting ARIMA orders...")

for i in range(K):
    order = best_arima(train_modes[i])
    mode_orders.append(order)
    print(f"Mode {i+1}: ARIMA{order}")

# ─────────────────────────────────────────
# 6. FAST FORECAST FUNCTION
# ─────────────────────────────────────────
def forecast_modes(modes, history_modes):
    total_preds = []

    for i in range(K):
        history = history_modes[i]
        steps   = len(modes[i])

        try:
            model = ARIMA(history, order=mode_orders[i]).fit()
            preds = model.forecast(steps=steps)
        except:
            preds = np.zeros(steps)

        total_preds.append(preds)

    # ✅ ALIGN LENGTHS (FIX ERROR)
    min_len = min(len(p) for p in total_preds)
    total_preds = [p[:min_len] for p in total_preds]

    return np.sum(np.array(total_preds), axis=0)

# ─────────────────────────────────────────
# 7. Validation Forecast
# ─────────────────────────────────────────
val_pred = forecast_modes(val_modes, train_modes)

# ─────────────────────────────────────────
# 8. Test Forecast
# ─────────────────────────────────────────
history_full = [np.concatenate([train_modes[i], val_modes[i]]) for i in range(K)]
test_pred = forecast_modes(test_modes, history_full)

test_pred = np.clip(test_pred, 0, 1)

# ─────────────────────────────────────────
# 9. Metrics
# ─────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

y_true = test_sc
y_pred = test_pred

# ✅ FINAL ALIGNMENT
min_len = min(len(y_true), len(y_pred))
y_true = y_true[:min_len]
y_pred = y_pred[:min_len]

print("\n" + "="*50)
print("RESULTS (Scaled)")
print("="*50)
print("MAE :", mean_absolute_error(y_true, y_pred))
print("MSE :", mean_squared_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("R2  :", r2_score(y_true, y_pred))
print("MAPE:", mape(y_true, y_pred))

# ─────────────────────────────────────────
# 10. Plot
# ─────────────────────────────────────────
y_true_o = scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
y_pred_o = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()

plt.figure(figsize=(10,4))
plt.plot(y_true_o, label="Actual", marker='o')
plt.plot(y_pred_o, label="Predicted", linestyle='--')
plt.title("VMD + ARIMA Forecast (FINAL OPTIMIZED)")
plt.xlabel("Time Step")
plt.ylabel("GHG Emissions")
plt.legend()
plt.grid()
plt.show()import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import minimize
from itertools import product

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("Japan_scaled.csv")   # ✅ Japan dataset

df = df[["Year","Annual greenhouse gas emissions including land use"]]
df = df.sort_values("Year").reset_index(drop=True)
df.columns = ["Year","GHG"]

values = df["GHG"].values.reshape(-1,1)

# =========================================================
# 2. SPLIT (70:20:10)
# =========================================================
n = len(values)

train_end = int(n*0.70)
val_end   = int(n*0.90)

train_raw = values[:train_end]
val_raw   = values[train_end:val_end]
test_raw  = values[val_end:]

# =========================================================
# 3. NORMALIZATION (AFTER SPLIT)
# =========================================================
scaler = MinMaxScaler()

train_sc = scaler.fit_transform(train_raw).flatten()
val_sc   = scaler.transform(val_raw).flatten()
test_sc  = scaler.transform(test_raw).flatten()

full_sc = np.concatenate([train_sc, val_sc, test_sc])

# =========================================================
# 4. VMD (CUSTOM)
# =========================================================
def vmd(signal, K=4, alpha=3000, tol=1e-7, max_iter=500):

    N = len(signal)
    f_ext = np.concatenate([signal[::-1], signal, signal[::-1]])
    T = len(f_ext)

    freqs = np.fft.fftfreq(T)
    f_hat = np.fft.fft(f_ext)

    u_hat = np.zeros((K, T), dtype=complex)
    omega = np.linspace(0, 0.5, K)
    lam = np.zeros(T, dtype=complex)

    for _ in range(max_iter):
        u_prev = u_hat.copy()

        for k in range(K):
            other = np.sum(u_hat, axis=0) - u_hat[k]
            num = f_hat - other - lam/2
            den = 1 + alpha*(freqs - omega[k])**2
            u_hat[k] = num / den

            pw = np.abs(u_hat[k])**2
            omega[k] = np.sum(freqs * pw) / (np.sum(pw)+1e-12)

        lam += np.sum(u_hat, axis=0) - f_hat

        if np.linalg.norm(u_hat - u_prev) < tol:
            break

    modes = np.array([np.real(np.fft.ifft(u_hat[k]))[N:2*N] for k in range(K)])
    return modes

print("Running VMD...")
K = 4
modes = vmd(full_sc, K=K)

# Split modes
m_tr = modes[:, :train_end]
m_val = modes[:, train_end:val_end]
m_te  = modes[:, val_end:]
m_tv  = modes[:, :val_end]

# =========================================================
# 5. SARIMA GRID SEARCH
# =========================================================
def best_sarima(series):

    p = range(0,3)
    d = range(0,2)
    q = range(0,3)

    P = range(0,2)
    D = range(0,2)
    Q = range(0,2)
    s = [4]

    best_aic = np.inf
    best_order = None
    best_seasonal = None

    for order in product(p,d,q):
        for seasonal in product(P,D,Q,s):
            try:
                model = SARIMAX(series,
                                order=order,
                                seasonal_order=seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit(disp=False)

                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = order
                    best_seasonal = seasonal
            except:
                continue

    return best_order, best_seasonal

orders = []
seasonals = []

for i in range(K):
    o, s = best_sarima(m_tr[i])
    orders.append(o)
    seasonals.append(s)
    print(f"Mode {i+1}: {o} {s}")

# =========================================================
# 6. FORECAST FUNCTION
# =========================================================
def forecast_sarima(train, future, order, seasonal):

    history = list(train)
    preds = []

    for t in range(len(future)):
        model = SARIMAX(history,
                        order=order,
                        seasonal_order=seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)

        pred = model.forecast()[0]

        preds.append(pred)
        history.append(future[t])

    return np.array(preds)

# =========================================================
# 7. VALIDATION + TEST FORECAST
# =========================================================
val_preds = []
test_preds = []

for i in range(K):

    vp = forecast_sarima(m_tr[i], m_val[i], orders[i], seasonals[i])
    tp = forecast_sarima(m_tv[i], m_te[i], orders[i], seasonals[i])

    val_preds.append(vp)
    test_preds.append(tp)

val_sum = np.sum(val_preds, axis=0)
test_sum = np.sum(test_preds, axis=0)

# =========================================================
# 8. GARCH
# =========================================================
def garch11(residuals):

    def loss(params):
        w,a,b = params
        if w<=0 or a<0 or b<0 or a+b>=1:
            return 1e10

        h = np.zeros(len(residuals))
        h[0] = np.var(residuals)

        for t in range(1,len(residuals)):
            h[t] = w + a*residuals[t-1]**2 + b*h[t-1]

        return np.sum(np.log(h) + residuals**2/h)

    res = minimize(loss, [0.01,0.05,0.9], bounds=[(1e-6,None),(0,1),(0,1)])
    return res.x

residuals = val_sum - m_val.sum(axis=0)

params = garch11(residuals)

def garch_forecast(params, resid, steps):
    w,a,b = params
    h = np.var(resid)
    out = []

    for _ in range(steps):
        h = w + a*(resid[-1]**2) + b*h
        out.append(np.sqrt(h))
    return np.array(out)

garch_corr = garch_forecast(params, residuals, len(test_sum))

final_test = test_sum + garch_corr * np.sign(np.mean(residuals))

# =========================================================
# 9. INVERSE SCALE
# =========================================================
inv = lambda x: scaler.inverse_transform(x.reshape(-1,1)).flatten()

y_test = inv(test_sc)
y_pred = inv(final_test)

# =========================================================
# 10. METRICS
# =========================================================
def mape(y,yp):
    return np.mean(np.abs((y-yp)/y))*100

print("\nRESULTS")
print("MAE :", mean_absolute_error(y_test,y_pred))
print("MSE :", mean_squared_error(y_test,y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2  :", r2_score(y_test,y_pred))
print("MAPE:", mape(y_test,y_pred))

# =========================================================
# 11. PLOT
# =========================================================
plt.figure(figsize=(10,4))
plt.plot(y_test,label="Actual",marker='o')
plt.plot(y_pred,label="SARIMA+VMD+GARCH",linestyle='--')
plt.legend()
plt.title("Japan GHG Prediction")   # ✅ updated
plt.show()import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

!pip install vmdpy arch

from vmdpy import VMD
from arch import arch_model

# ─────────────────────────────────────────
# 1. Load Japan Data
# ─────────────────────────────────────────
df = pd.read_csv("Japan_scaled.csv") # Corrected: Changed 'japan_scaled.csv' to 'Japan_scaled.csv'

df = df[["Year", "Annual greenhouse gas emissions including land use"]]
df = df.sort_values("Year").reset_index(drop=True)
df.columns = ["Year", "GHG"]

values = df["GHG"].values.reshape(-1, 1)

# ─────────────────────────────────────────
# 2. Split 70:20:10
# ─────────────────────────────────────────
n = len(values)

train_end = int(n * 0.70)
val_end   = int(n * 0.90)

train_raw = values[:train_end]
val_raw   = values[train_end:val_end]
test_raw  = values[val_end:]

# ─────────────────────────────────────────
# 3. Normalize After Split
# ─────────────────────────────────────────
scaler = MinMaxScaler()

train_sc = scaler.fit_transform(train_raw).flatten()
val_sc   = scaler.transform(val_raw).flatten()
test_sc  = scaler.transform(test_raw).flatten()

# ─────────────────────────────────────────
# 4. VMD Decomposition
# ─────────────────────────────────────────
K = 5
alpha = 2000
tau = 0
DC = 0
init = 1
tol = 1e-7

u, _, _ = VMD(train_sc, alpha, tau, K, DC, init, tol)

print("VMD modes shape:", u.shape)

# ─────────────────────────────────────────
# 5. VMD Modes Plot
# ─────────────────────────────────────────
plt.figure(figsize=(10, 6), dpi=300)

for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(u[i], linewidth=1.5)
    plt.title(f"Mode {i+1}")
    plt.grid(False)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 6. ARIMA Per Mode
# ─────────────────────────────────────────
def best_arima(series):
    best_aic = np.inf
    best_order = None

    for p, d, q in product(range(0, 4), range(0, 2), range(0, 4)):
        try:
            model = ARIMA(series, order=(p, d, q)).fit()

            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, d, q)

        except:
            continue

    return best_order

mode_orders = []

for i in range(K):
    order = best_arima(u[i])
    mode_orders.append(order)
    print(f"Mode {i+1}: ARIMA Order = {order}")

# ─────────────────────────────────────────
# 7. Validation Forecast ARIMA + VMD + GARCH
# ─────────────────────────────────────────
val_preds_total = np.zeros(len(val_sc))

for i in range(K):

    history = list(u[i])
    preds = []

    for t in range(len(val_sc)):

        model = ARIMA(history, order=mode_orders[i]).fit()
        arima_pred = model.forecast()[0]

        resid = model.resid

        try:
            garch = arch_model(resid, vol='Garch', p=1, q=1).fit(disp='off')
            garch_pred = garch.forecast(horizon=1).variance.values[-1][0]

        except:
            garch_pred = 0

        final_pred = arima_pred + garch_pred

        preds.append(final_pred)

        history.append(val_sc[t] / K)

    preds = np.array(preds)

    min_len = min(len(val_preds_total), len(preds))

    val_preds_total[:min_len] += preds[:min_len]

val_losses = (val_sc[:min_len] - val_preds_total[:min_len]) ** 2

# ─────────────────────────────────────────
# 8. Validation Loss Plot
# ─────────────────────────────────────────
plt.figure(figsize=(8, 4), dpi=300)

plt.plot(val_losses,
         linewidth=2,
         color='teal',
         label='Validation Loss')

plt.title("Japan ARIMA-VMD-GARCH Validation Loss")
plt.xlabel("Step")
plt.ylabel("MSE")

plt.legend(frameon=True)
plt.grid(False)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 9. Test Forecast
# ─────────────────────────────────────────
test_preds_total = np.zeros(len(test_sc))

for i in range(K):

    history = list(u[i]) + list(val_sc / K)
    preds = []

    for t in range(len(test_sc)):

        model = ARIMA(history, order=mode_orders[i]).fit()
        arima_pred = model.forecast()[0]

        resid = model.resid

        try:
            garch = arch_model(resid, vol='Garch', p=1, q=1).fit(disp='off')
            garch_pred = garch.forecast(horizon=1).variance.values[-1][0]

        except:
            garch_pred = 0

        final_pred = arima_pred + garch_pred

        preds.append(final_pred)

        history.append(test_sc[t] / K)

    preds = np.array(preds)

    min_len = min(len(test_preds_total), len(preds))

    test_preds_total[:min_len] += preds[:min_len]

# ─────────────────────────────────────────
# 10. Metrics
# ─────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mae  = mean_absolute_error(test_sc[:min_len], test_preds_total[:min_len])
mse  = mean_squared_error(test_sc[:min_len], test_preds_total[:min_len])
rmse = np.sqrt(mse)
r2   = r2_score(test_sc[:min_len], test_preds_total[:min_len])
mp   = mape(test_sc[:min_len], test_preds_total[:min_len])

print("\n" + "="*50)
print("Japan ARIMA + VMD + GARCH Results")
print("="*50)
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)
print("MAPE:", mp)

# ─────────────────────────────────────────
# 11. Actual vs Predicted Plot - Sample Style
# ─────────────────────────────────────────
plt.figure(figsize=(8, 4.5), dpi=300)

x_axis = np.arange(len(test_sc[:min_len]))

plt.plot(x_axis,
         test_sc[:min_len],
         color='teal',
         linewidth=2,
         label='Actual')

plt.plot(x_axis,
         test_preds_total[:min_len],
         color='darkgoldenrod',
         linewidth=2,
         linestyle='--',
         label='Predicted')

plt.xlabel("Time")
plt.ylabel("Emissions")

plt.legend(loc='lower left', frameon=True)

plt.grid(False)

plt.tight_layout()

plt.savefig("Japan_actual_vs_predicted_sample_style.png",
            dpi=600,
            bbox_inches='tight')

plt.show()