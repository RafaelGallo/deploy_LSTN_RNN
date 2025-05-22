import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import timedelta

st.set_page_config(page_title="Previsão Close Price", layout="wide")
st.title("📈 Previsão de Fechamento (Close) com LSTM")

# ─── DEBUG: listar arquivos ─────────────────────────────────────────────────────
st.subheader("🧐 Debug: Conteúdo do repositório")
root_files = os.listdir(".")
models_files = os.listdir("models") if os.path.isdir("models") else []
st.write("Arquivos na raiz:", root_files)
st.write("Arquivos em models/:", models_files)
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_cached(path: str):
    return load_model(path)

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df

# Carregando modelo (path POSIX)
MODEL_PATH = "models/model_lstm1.keras"
if not os.path.isfile(MODEL_PATH):
    st.error(f"❌ Arquivo não encontrado em: {MODEL_PATH}")
    st.stop()

model = load_model_cached(MODEL_PATH)

# Extrai seq_length e n_features
_, seq_length, n_features = model.input_shape

uploaded = st.file_uploader("Faça upload do CSV com suas colunas", type=["csv"])
if uploaded:
    df = load_data(uploaded)
    st.subheader("Dados carregados")
    st.dataframe(df.head())

    FEATURES = ['Open','High','Low','Close','Volume']
    if not all(col in df.columns for col in FEATURES):
        st.error(f"Seu CSV precisa conter as colunas: {FEATURES}")
        st.stop()

    # Cria janela de entrada
    last_window = df[FEATURES].values[-seq_length:]
    X = last_window.reshape(1, seq_length, n_features)

    # Previsão
    pred = model.predict(X)[0][0]
    next_date = df['Datetime'].iloc[-1] + timedelta(days=1)
    st.markdown(f"**Data prevista:** {next_date.date()}  **Fechamento previsto:** {pred:.2f}")

    # Gráfico histórico + previsão
    import plotly.express as px
    hist = df[['Datetime','Close']].rename(columns={'Close':'Preço'})
    forecast = pd.DataFrame({'Datetime':[next_date], 'Preço':[pred]})
    fig = px.line(hist, x='Datetime', y='Preço', title="Histórico + Previsão")
    fig.add_scatter(x=forecast['Datetime'], y=forecast['Preço'],
                    mode='markers+text', text=[f"{pred:.2f}"],
                    textposition="bottom center", name='Previsto')
    st.plotly_chart(fig, use_container_width=True)
