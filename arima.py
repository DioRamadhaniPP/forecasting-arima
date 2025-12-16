import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error

# ------------------------------------
# Konfigurasi halaman
# ------------------------------------
st.set_page_config(
    page_title="Forecasting ARIMA",
    page_icon="📈",
    layout="wide"
)

st.title("Forecasting Time Series dengan Metode ARIMA")
st.caption("ARIMA (Box–Jenkins) + Transformasi + Evaluasi Model")

# ------------------------------------
# Fungsi bantu
# ------------------------------------
def adf_test(series):
    result = adfuller(series.dropna())
    return result[0], result[1]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ------------------------------------
# Sidebar: Input Data
# ------------------------------------
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

columns = list(df.columns)
date_col = st.sidebar.selectbox("Kolom tanggal", columns, index=0)
value_col = st.sidebar.selectbox("Kolom nilai", columns, index=len(columns) - 1)

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(by=date_col)
df.set_index(date_col, inplace=True)

series_raw = df[value_col].astype(float)

# ------------------------------------
# Transformasi Data
# ------------------------------------
st.sidebar.header("Transformasi Data")

transform_power = st.sidebar.number_input(
    "Nilai transformasi (k)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

if transform_power != 1.0:
    series = series_raw ** transform_power
else:
    series = series_raw.copy()

# ------------------------------------
# Parameter ARIMA
# ------------------------------------
st.sidebar.header("Parameter ARIMA")

p = st.sidebar.number_input("p (AR)", 0, 10, 1)
d = st.sidebar.number_input("d (Differencing)", 0, 3, 1)
q = st.sidebar.number_input("q (MA)", 0, 10, 0)
steps = st.sidebar.number_input("Periode forecast", 1, 100, 12)

series_diff = series.diff(d).dropna() if d > 0 else series.copy()

# ------------------------------------
# Tabs
# ------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["1. Data", "2. Stasioneritas & ACF/PACF", "3. Model, Evaluasi & Forecast"]
)

# ------------------------------------
# TAB 1: Data
# ------------------------------------
with tab1:
    st.subheader("Data Time Series")
    st.dataframe(df.head())
    st.line_chart(series_raw)

# ------------------------------------
# TAB 2: Stasioneritas
# ------------------------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        adf_stat, p_val = adf_test(series)
        st.write("ADF Test (Data Asli)")
        st.write(f"ADF Statistic : {adf_stat:.4f}")
        st.write(f"p-value       : {p_val:.4f}")

    with col2:
        adf_stat_d, p_val_d = adf_test(series_diff)
        st.write("ADF Test (Setelah Differencing)")
        st.write(f"ADF Statistic : {adf_stat_d:.4f}")
        st.write(f"p-value       : {p_val_d:.4f}")

    col3, col4 = st.columns(2)
    with col3:
        fig1, ax1 = plt.subplots()
        plot_acf(series_diff, ax=ax1, lags=20)
        st.pyplot(fig1)

    with col4:
        fig2, ax2 = plt.subplots()
        plot_pacf(series_diff, ax=ax2, lags=20, method="ywm")
        st.pyplot(fig2)

# ------------------------------------
# TAB 3: Model, Evaluasi & Forecast
# ------------------------------------
with tab3:
    if st.button("Jalankan Model ARIMA"):
        try:
            # -------------------------
            # Fit Model
            # -------------------------
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()

            st.subheader("Ringkasan Model")
            st.text(model_fit.summary())

            # -------------------------
            # Evaluasi Model (LEVEL DATA AKTUAL)
            # -------------------------
            fitted = model_fit.predict(typ="levels")
            fitted = fitted.reindex(series_raw.index)

            if transform_power != 1.0:
                fitted = fitted ** (1 / transform_power)

            eval_df = pd.concat([series_raw, fitted], axis=1).dropna()
            eval_df.columns = ["Actual", "Fitted"]

            rmse_value = np.sqrt(mean_squared_error(
                eval_df["Actual"], eval_df["Fitted"]
            ))

            mape_value = mape(
                eval_df["Actual"], eval_df["Fitted"]
            )

            st.subheader("Evaluasi Model (In-Sample)")
            col_e1, col_e2 = st.columns(2)
            col_e1.metric("RMSE", f"{rmse_value:.4f}")
            col_e2.metric("MAPE (%)", f"{mape_value:.2f}%")

            # -------------------------
            # Diagnostik Residual
            # -------------------------
            st.subheader("Uji Ljung-Box")
            residuals = model_fit.resid.dropna()
            lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
            st.dataframe(lb)

            # -------------------------
            # Forecast
            # -------------------------
            forecast = model_fit.forecast(steps=steps)

            if transform_power != 1.0:
                forecast = forecast ** (1 / transform_power)

            freq = pd.infer_freq(series_raw.index) or "M"
            forecast_index = pd.date_range(
                start=series_raw.index[-1],
                periods=steps + 1,
                freq=freq
            )[1:]

            forecast = pd.Series(forecast.values, index=forecast_index)

            st.subheader("Hasil Forecast")
            st.dataframe(
                pd.DataFrame({
                    "Tanggal": forecast.index,
                    "Forecast": forecast.values
                })
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series_raw.index, series_raw.values, label="Actual")
            ax.plot(eval_df.index, eval_df["Fitted"], label="Fitted")
            ax.plot(forecast.index, forecast.values, "--o", label="Forecast")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
