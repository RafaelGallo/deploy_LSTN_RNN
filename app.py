import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import timedelta
import plotly.express as px

# Configurações da página
st.set_page_config(page_title="Previsão de Fechamento LSTM", layout="wide")
st.title("📈 Previsão de Fechamento (Close) com LSTM")

# Funções de cache
@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    return load_model(path)

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["Datetime"])
    return df.sort_values("Datetime").reset_index(drop=True)

# Carregamento do modelo com spinner
MODEL_PATH = "models/model_lstm_rnn1.h5"
with st.spinner("🔄 Carregando modelo..."):
    model = load_model_cached(MODEL_PATH)

# Extrai a configuração de entrada
_, seq_length, n_features = model.input_shape

# Uploader de CSV
uploaded = st.file_uploader(
    "Faça upload do CSV com colunas: Datetime, Open, High, Low, Close, Volume, Ticker",
    type="csv"
)

if uploaded:
    df = load_data(uploaded)
    st.subheader("Dados carregados")
    st.dataframe(df.head())

    # Verifica colunas necessárias
    FEATURES = ["Open","High","Low","Close","Volume"]
    if not all(col in df.columns for col in FEATURES):
        st.error(f"CSV inválido! Precisa conter colunas: {FEATURES}")
        st.stop()

    # Prepara última janela de entrada
    data = df[FEATURES].values
    last_window = data[-seq_length:]
    X = last_window.reshape(1, seq_length, n_features)

    # Gera previsão
    pred = model.predict(X)[0,0]
    next_date = df["Datetime"].iloc[-1] + timedelta(days=1)
    st.markdown(f"**Data prevista:** {next_date.date()} &nbsp;&nbsp; **Fechamento previsto:** R$ {pred:.2f}")

    # Plota histórico + previsão
    hist = df[["Datetime","Close"]].rename(columns={"Close":"Preço"})
    fc = pd.DataFrame({"Datetime":[next_date], "Preço":[pred]})
    fig = px.line(hist, x="Datetime", y="Preço", title="Histórico de Fechamento + Previsão")
    fig.add_scatter(
        x=fc["Datetime"], y=fc["Preço"],
        mode="markers+text", text=[f"{pred:.2f}"],
        textposition="bottom center", name="Previsto"
    )
    st.plotly_chart(fig, use_container_width=True)
