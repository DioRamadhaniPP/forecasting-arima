import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Forecasting ARIMA",
    page_icon="📈",
    layout="wide"
)

st.title("Forecasting Time Series dengan Metode ARIMA")
st.caption("ARIMA (Box–Jenkins) + Koefisien Dinamis + Evaluasi Model")

# ======================================================
# FUNGSI BANTU
# ======================================================
def adf_test(series):
    stat, pvalue, _, _, _, _ = adfuller(series.dropna())
    return stat, pvalue

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ======================================================
# SIDEBAR – INPUT DATA
# ======================================================
st.sidebar.header("Pengaturan Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV (tanggal & nilai)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    date_rng = pd.date_range(start="2020-01-01", periods=60, freq="M")
    df = pd.DataFrame({
        "tanggal": date_rng,
        "nilai": np.random.randint(30, 70, size=60)
    })

date_col = st.sidebar.selectbox("Kolom tanggal", df.columns, index=0)
value_col = st.sidebar.selectbox("Kolom nilai", df.columns, index=len(df.columns)-1)

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(by=date_col).set_index(date_col)

series_raw = df[value_col].astype(float)

# ======================================================
# TRANSFORMASI DATA
# ======================================================
st.sidebar.header("Transformasi Data")

transform_power = st.sidebar.number_input(
    "Transformasi Pangkat (k)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

series = series_raw ** transform_power if transform_power != 1 else series_raw.copy()

# ======================================================
# PARAMETER ARIMA
# ======================================================
st.sidebar.header("Parameter ARIMA")

p = st.sidebar.number_input("p (AR)", 0, 5, 1)
d = st.sidebar.number_input("d (Differencing)", 0, 3, 1)
q = st.sidebar.number_input("q (MA)", 0, 5, 0)

steps = st.sidebar.number_input("Periode Forecast", 1, 50, 12)

series_diff = series.diff(d).dropna() if d > 0 else series.copy()

# ======================================================
# KOEFISIEN DINAMIS (FIXED PARAMETERS)
# ======================================================
st.sidebar.subheader("Koefisien AR & MA (Opsional)")

use_custom_params = st.sidebar.checkbox("Gunakan koefisien manual", False)

ar_params, ma_params = [], []

if use_custom_params:
    st.sidebar.markdown("**Koefisien AR (φ)**")
    for i in range(p):
        ar_params.append(
            st.sidebar.number_input(
                f"φ{i+1}", value=0.1, step=0.05
            )
        )

    st.sidebar.markdown("**Koefisien MA (θ)**")
    for i in range(q):
        ma_params.append(
            st.sidebar.number_input(
                f"θ{i+1}", value=0.1, step=0.05
            )
        )

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "1. Data",
    "2. Stasioneritas & ACF/PACF",
    "3. Model, Evaluasi & Forecast"
])

# ======================================================
# TAB 1 – DATA
# ======================================================
with tab1:
    st.subheader("Data Time Series")
    st.dataframe(df.head())
    st.line_chart(series_raw)

# ======================================================
# TAB 2 – STASIONERITAS
# ======================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        stat, pval = adf_test(series)
        st.write("ADF Test (Data Asli)")
        st.write(f"ADF Statistic: {stat:.4f}")
        st.write(f"p-value: {pval:.4f}")

    with col2:
        stat_d, pval_d = adf_test(series_diff)
        st.write("ADF Test (Setelah Differencing)")
        st.write(f"ADF Statistic: {stat_d:.4f}")
        st.write(f"p-value: {pval_d:.4f}")

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots()
        plot_acf(series_diff, ax=ax, lags=20)
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        plot_pacf(series_diff, ax=ax, lags=20, method="ywm")
        st.pyplot(fig)

# ======================================================
# TAB 3 – MODEL & FORECAST
# ======================================================
with tab3:
    if st.button("Jalankan Model ARIMA"):
        try:
            model = ARIMA(series, order=(p, d, q))

            # -----------------------------
            # FIT MODEL (FIXED PARAMETER)
            # -----------------------------
            if use_custom_params and (p > 0 or q > 0):
                fixed_params = {}

                for i in range(p):
                    fixed_params[f"ar.L{i+1}"] = ar_params[i]

                for i in range(q):
                    fixed_params[f"ma.L{i+1}"] = ma_params[i]

                model_fit = model.fit_constrained(fixed_params)
            else:
                model_fit = model.fit()

            st.subheader("Ringkasan Model")
            st.text(model_fit.summary())

            # -----------------------------
            # KOEFISIEN
            # -----------------------------
            st.subheader("Koefisien Model")
            coef_df = pd.DataFrame({
                "Parameter": model_fit.params.index,
                "Nilai": model_fit.params.values
            })
            st.dataframe(coef_df)

            # -----------------------------
            # EVALUASI MODEL
            # -----------------------------
            fitted = model_fit.predict(typ="levels")
            fitted = fitted.reindex(series_raw.index)

            if transform_power != 1:
                fitted = fitted ** (1 / transform_power)

            eval_df = pd.concat([series_raw, fitted], axis=1).dropna()
            eval_df.columns = ["Actual", "Fitted"]

            rmse = np.sqrt(mean_squared_error(eval_df["Actual"], eval_df["Fitted"]))
            mape_val = mape(eval_df["Actual"], eval_df["Fitted"])

            st.subheader("Evaluasi In-Sample")
            c1, c2 = st.columns(2)
            c1.metric("RMSE", f"{rmse:.4f}")
            c2.metric("MAPE (%)", f"{mape_val:.2f}")

            # -----------------------------
            # DIAGNOSTIK RESIDUAL
            # -----------------------------
            st.subheader("Uji Ljung-Box")
            lb = acorr_ljungbox(model_fit.resid.dropna(), lags=[10], return_df=True)
            st.dataframe(lb)

            # -----------------------------
            # FORECAST
            # -----------------------------
            forecast = model_fit.forecast(steps=steps)
            if transform_power != 1:
                forecast = forecast ** (1 / transform_power)

            freq = pd.infer_freq(series_raw.index) or "M"
            forecast_index = pd.date_range(
                series_raw.index[-1], periods=steps+1, freq=freq
            )[1:]

            forecast = pd.Series(forecast.values, index=forecast_index)

            st.subheader("Hasil Forecast")
            st.dataframe(forecast.rename("Forecast"))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series_raw, label="Actual")
            ax.plot(eval_df["Fitted"], label="Fitted")
            ax.plot(forecast, "--o", label="Forecast")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
