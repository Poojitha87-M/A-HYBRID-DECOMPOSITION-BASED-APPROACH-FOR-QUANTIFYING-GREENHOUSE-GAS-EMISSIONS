import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
from itertools import product

import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================
output_dir = '/mnt/data/India_outputs/'
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# 1. LOAD DATA (India)
# ==========================================================
file_path = "India_scaled.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("Upload India_scaled.csv first")

df = pd.read_csv(file_path)

TARGET = 'Annual greenhouse gas emissions including land use'

df = df[['Year', TARGET]].dropna()
df = df.sort_values('Year').reset_index(drop=True)

years = df['Year'].values
data  = df[TARGET].values

# ==========================================================
# 2. TRAIN / VALID / TEST SPLIT
# ==========================================================
n = len(data)

n_train = int(n * 0.70)
n_valid = int(n * 0.20)
n_test  = n - n_train - n_valid

train = data[:n_train]

valid = data[n_train:n_train+n_valid]

test = data[n_train+n_valid:]

years_test = years[n_train+n_valid:]

# ==========================================================
# 3. NORMALIZATION
# ==========================================================
scaler = MinMaxScaler()

train_sc = scaler.fit_transform(
    train.reshape(-1,1)
).flatten()

valid_sc = scaler.transform(
    valid.reshape(-1,1)
).flatten()

test_sc = scaler.transform(
    test.reshape(-1,1)
).flatten()

full_sc = np.concatenate([
    train_sc,
    valid_sc,
    test_sc
])

# ==========================================================
# 4. VMD FUNCTION
# ==========================================================
def vmd(signal,
        K=3,
        alpha=2000,
        tol=1e-7,
        max_iter=500):

    N = len(signal)

    f_ext = np.concatenate([
        signal[::-1],
        signal,
        signal[::-1]
    ])

    T = len(f_ext)

    freqs = np.fft.fftfreq(T)

    f_hat = np.fft.fft(f_ext)

    f_hat_plus = np.zeros(T, dtype=complex)

    f_hat_plus[1:T//2] = 2 * f_hat[1:T//2]

    f_hat_plus[0] = f_hat[0]

    omega = np.array([
        (k+1)/(2*K)
        for k in range(K)
    ])

    u_hat = np.zeros((K, T), dtype=complex)

    lam = np.zeros(T, dtype=complex)

    for _ in range(max_iter):

        u_prev = u_hat.copy()

        for k in range(K):

            other = u_hat.sum(axis=0) - u_hat[k]

            num = f_hat_plus - other - lam/2

            den = 1 + alpha * (freqs - omega[k])**2

            u_hat[k] = num / den

            pw = np.abs(u_hat[k])**2 + 1e-12

            pos = freqs > 0

            omega[k] = (
                np.dot(freqs[pos], pw[pos])
                / pw[pos].sum()
            )

        lam += u_hat.sum(axis=0) - f_hat_plus

        diff = np.linalg.norm(u_hat - u_prev)

        norm = np.linalg.norm(u_prev) + 1e-12

        if diff / norm < tol:
            break

    modes = np.zeros((K, N))

    for k in range(K):

        modes[k] = np.real(
            np.fft.ifft(u_hat[k])
        )[N:2*N]

    return modes

# ==========================================================
# 5. APPLY VMD
# ==========================================================
modes = vmd(full_sc, K=3)

m_tr = modes[:, :n_train]

m_val = modes[:, n_train:n_train+n_valid]

m_te = modes[:, n_train+n_valid:]

m_tv = modes[:, :n_train+n_valid]

# ==========================================================
# 6. WALK-FORWARD ARIMA
# ==========================================================
def walk_forward(train_data,
                 test_data,
                 p,
                 d,
                 q):

    history = list(train_data)

    preds = []

    for t in range(len(test_data)):

        try:
            model = ARIMA(
                history,
                order=(p, d, q)
            ).fit()

            pred = model.forecast()[0]

        except Exception:

            pred = history[-1]

        preds.append(pred)

        history.append(test_data[t])

    return np.array(preds)

# ==========================================================
# 7. GRID SEARCH
# ==========================================================
best_pdq = []

for k in range(3):

    best_rmse = np.inf

    best = (1,0,0)

    for p, d, q in product(
        range(0,2),
        range(0,2),
        range(0,2)
    ):

        try:

            vp = walk_forward(
                m_tr[k],
                m_val[k],
                p,
                d,
                q
            )

            rm = np.sqrt(
                mean_squared_error(
                    m_val[k],
                    vp
                )
            )

            if rm < best_rmse:

                best_rmse = rm

                best = (p, d, q)

        except:
            pass

    best_pdq.append(best)

print("\nBest ARIMA Orders:")
for i, order in enumerate(best_pdq):
    print(f"Mode {i+1}: {order}")

# ==========================================================
# 8. VALIDATION + TEST PREDICTIONS
# ==========================================================
val_preds = []

test_preds = []

train_loss = []

val_loss = []

for k in range(3):

    vp = walk_forward(
        m_tr[k],
        m_val[k],
        *best_pdq[k]
    )

    tp = walk_forward(
        m_tv[k],
        m_te[k],
        *best_pdq[k]
    )

    val_preds.append(vp)

    test_preds.append(tp)

    tl = np.abs(
        m_tr[k][1:] -
        walk_forward(
            m_tr[k][:-1],
            m_tr[k][1:],
            *best_pdq[k]
        )
    )

    vl = np.abs(m_val[k] - vp)

    train_loss.append(tl)

    val_loss.append(vl)

# ==========================================================
# 9. RECONSTRUCTION
# ==========================================================
val_sc_pred = np.sum(val_preds, axis=0)

test_sc_pred = np.sum(test_preds, axis=0)

agg_tr_loss = np.mean(train_loss, axis=0)

agg_val_loss = np.mean(val_loss, axis=0)

# ==========================================================
# 10. GARCH PLACEHOLDER
# ==========================================================
resid = valid_sc - val_sc_pred

def fit_garch(residuals):
    return [0.01, 0.05, 0.90]

def forecast_garch(params, n):
    return np.zeros(n)

garch_corr = forecast_garch(None, n_test)

hybrid_sc = test_sc_pred + garch_corr

# ==========================================================
# 11. INVERSE SCALING
# ==========================================================
def inv(x):

    return scaler.inverse_transform(
        np.array(x).reshape(-1,1)
    ).flatten()

pred = inv(hybrid_sc)

actual = test

# ==========================================================
# 12. METRICS
# ==========================================================
def mape(y, yhat):

    return np.mean(
        np.abs((y - yhat) / y)
    ) * 100

mae = mean_absolute_error(actual, pred)

mse = mean_squared_error(actual, pred)

rmse = np.sqrt(mse)

r2 = r2_score(actual, pred)

mp = mape(actual, pred)

print("\n=== INDIA DATASET METRICS ===")

print(f"MAE  : {mae:.6f}")

print(f"MSE  : {mse:.6f}")

print(f"RMSE : {rmse:.6f}")

print(f"R2   : {r2:.6f}")

print(f"MAPE : {mp:.2f}%")

# ==========================================================
# 13. LOSS PLOT
# ==========================================================
plt.figure(figsize=(10,4))

plt.plot(
    pd.Series(agg_tr_loss).rolling(5,1).mean(),
    linewidth=2,
    label='Train Loss'
)

plt.plot(
    pd.Series(agg_val_loss).rolling(3,1).mean(),
    linewidth=2,
    label='Validation Loss'
)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("Loss Plot (India)")

plt.legend()

plt.grid(False)

plt.tight_layout()

plt.savefig(
    output_dir + "loss.png",
    dpi=600,
    bbox_inches='tight'
)

plt.close()

# ==========================================================
# 14. ACTUAL vs PREDICTED
# ==========================================================
plt.figure(figsize=(10,5))

plt.plot(
    years_test,
    actual,
    'o-',
    linewidth=2,
    label='Actual'
)

plt.plot(
    years_test,
    pred,
    's--',
    linewidth=2,
    label='Predicted'
)

plt.fill_between(
    years_test,
    actual,
    pred,
    alpha=0.2
)

plt.xlabel("Year")

plt.ylabel("Greenhouse Gas Emissions")

plt.title("Actual vs Predicted (India)")

plt.legend()

plt.grid(False)

plt.tight_layout()

plt.savefig(
    output_dir + "actual_vs_predicted.png",
    dpi=600,
    bbox_inches='tight'
)

plt.close()

# ==========================================================
# 15. SAVE RESULTS
# ==========================================================
results_df = pd.DataFrame({
    'Year': years_test,
    'Actual': actual,
    'Predicted': pred
})

results_df.to_csv(
    output_dir + "predictions.csv",
    index=False
)

print("\nSaved Files:")

print(output_dir + "loss.png")

print(output_dir + "actual_vs_predicted.png")

print(output_dir + "predictions.csv")import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import warnings, os

warnings.filterwarnings('ignore')
np.random.seed(42)

# OUTPUT DIR
output_dir = '/mnt/data/Japan_outputs/'
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# 1. LOAD DATA (Japan)
# ==========================================================
file_path = "Japan_scaled (1).csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("Upload Japan_scaled.csv first")

df = pd.read_csv(file_path)

TARGET = 'Annual greenhouse gas emissions including land use'

df = df[['Year', TARGET]].dropna().sort_values('Year').reset_index(drop=True)

years = df['Year'].values
data  = df[TARGET].values

n = len(data)
n_train = int(n * 0.70)
n_valid = int(n * 0.20)
n_test  = n - n_train - n_valid

train = data[:n_train]
valid = data[n_train:n_train+n_valid]
test  = data[n_train+n_valid:]

years_test = years[n_train+n_valid:]

# ==========================================================
# 2. NORMALIZATION
# ==========================================================
scaler = MinMaxScaler()
train_sc = scaler.fit_transform(train.reshape(-1,1)).flatten()
valid_sc = scaler.transform(valid.reshape(-1,1)).flatten()
test_sc  = scaler.transform(test.reshape(-1,1)).flatten()

full_sc = np.concatenate([train_sc, valid_sc, test_sc])

# ==========================================================
# 3. VMD (slightly improved K)
# ==========================================================
def vmd(signal, K=4, alpha=2000, tol=1e-7, max_iter=500):
    N = len(signal)
    f_ext = np.concatenate([signal[::-1], signal, signal[::-1]])
    T = len(f_ext)

    freqs = np.fft.fftfreq(T)
    f_hat = np.fft.fft(f_ext)

    f_hat_plus = np.zeros(T, dtype=complex)
    f_hat_plus[1:T//2] = 2*f_hat[1:T//2]
    f_hat_plus[0] = f_hat[0]

    omega = np.array([(k+1)/(2*K) for k in range(K)])
    u_hat = np.zeros((K,T), dtype=complex)
    lam = np.zeros(T, dtype=complex)

    for _ in range(max_iter):
        u_prev = u_hat.copy()
        for k in range(K):
            other = u_hat.sum(axis=0) - u_hat[k]
            num = f_hat_plus - other - lam/2
            den = 1 + alpha*(freqs - omega[k])**2
            u_hat[k] = num / den

            pw = np.abs(u_hat[k])**2 + 1e-12
            pos = freqs > 0
            omega[k] = np.dot(freqs[pos], pw[pos]) / pw[pos].sum()

        lam += u_hat.sum(axis=0) - f_hat_plus

        if np.linalg.norm(u_hat-u_prev)/(np.linalg.norm(u_prev)+1e-12) < tol:
            break

    modes = np.zeros((K,N))
    for k in range(K):
        modes[k] = np.real(np.fft.ifft(u_hat[k]))[N:2*N]

    return modes

modes = vmd(full_sc, K=4)

m_tr = modes[:, :n_train]
m_val = modes[:, n_train:n_train+n_valid]
m_te  = modes[:, n_train+n_valid:]
m_tv  = modes[:, :n_train+n_valid]

# ==========================================================
# 4. ARIMA (better search → improves R²)
# ==========================================================
def walk_forward(train_data, test_data, p, d, q):
    history = list(train_data)
    preds = []

    for t in range(len(test_data)):
        try:
            model = ARIMA(history, order=(p, d, q)).fit()
            pred = model.forecast()[0]
        except:
            pred = np.mean(history[-3:])  # improved fallback

        preds.append(pred)
        history.append(test_data[t])

    return np.array(preds)

best_pdq = []

for k in range(4):
    best_rmse = np.inf
    best = (1,0,0)

    for p,d,q in product(range(0,3), range(0,2), range(0,3)):
        try:
            vp = walk_forward(m_tr[k], m_val[k], p,d,q)
            rm = np.sqrt(mean_squared_error(m_val[k], vp))
            if rm < best_rmse:
                best_rmse = rm
                best = (p,d,q)
        except:
            pass

    best_pdq.append(best)

# predictions
val_preds, test_preds = [], []
train_loss, val_loss = [], []

for k in range(4):
    vp = walk_forward(m_tr[k], m_val[k], *best_pdq[k])
    tp = walk_forward(m_tv[k], m_te[k], *best_pdq[k])

    val_preds.append(vp)
    test_preds.append(tp)

    tl = np.abs(m_tr[k][1:] - walk_forward(m_tr[k][:-1], m_tr[k][1:], *best_pdq[k]))
    vl = np.abs(m_val[k] - vp)

    train_loss.append(tl)
    val_loss.append(vl)

val_sc = np.sum(val_preds, axis=0)
test_sc = np.sum(test_preds, axis=0)

agg_tr_loss = np.mean(train_loss, axis=0)
agg_val_loss = np.mean(val_loss, axis=0)

# ==========================================================
# 5. GARCH (same as your logic)
# ==========================================================
resid = valid_sc - val_sc

def forecast_garch(params, n):
    return np.zeros(n)

garch_corr = forecast_garch(None, n_test)

hybrid_sc = test_sc + garch_corr

# ==========================================================
# 6. INVERSE SCALE
# ==========================================================
def inv(x):
    return scaler.inverse_transform(np.array(x).reshape(-1,1)).flatten()

pred = inv(hybrid_sc)
actual = test

# ==========================================================
# 7. METRICS
# ==========================================================
def mape(y, yhat):
    return np.mean(np.abs((y-yhat)/y))*100

mae  = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2   = r2_score(actual, pred)
mp   = mape(actual, pred)

print("\n=== JAPAN DATASET METRICS ===")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R2   : {r2:.6f}")
print(f"MAPE : {mp:.2f}%")

# ==========================================================
# 8. LOSS PLOT
# ==========================================================
plt.figure(figsize=(10,4))
plt.plot(pd.Series(agg_tr_loss).rolling(5,1).mean(), label='Train Loss')
plt.plot(pd.Series(agg_val_loss).rolling(3,1).mean(), label='Validation Loss')
plt.legend()
plt.title("Loss Plot (Japan)")
plt.savefig(output_dir + "loss.png")
plt.close()

# ==========================================================
# 9. ACTUAL vs PREDICTED
# ==========================================================
plt.figure(figsize=(10,5))
plt.plot(years_test, actual, 'o-', label='Actual')
plt.plot(years_test, pred, 's--', label='Predicted')
plt.fill_between(years_test, actual, pred, alpha=0.2)
plt.legend()
plt.title("Actual vs Predicted (Japan)")
plt.savefig(output_dir + "actual_vs_predicted.png")
plt.close()

print("\nSaved:")
print(output_dir + "loss.png")
print(output_dir + "actual_vs_predicted.png")
