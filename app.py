import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Previsão Close Price", layout="wide")
st.title("📈 Previsão de Fechamento (Close) com LSTM")

# ─── DEBUG (você pode remover depois) ─────────────────────────────────────────────
st.subheader("🧐 Debug: Estrutura de arquivos")
st.write("Raiz:", os.listdir("."))
if os.path.isdir("models"):
    lst = os.listdir("models")
    st.write("models/:", lst)
    for name in lst:
        p = os.path.join("models", name); info = os.stat(p)
        st.write(f"- {name}: file={os.path.isfile(p)}, size={info.st_size} bytes")
else:
    st.write("❌ Não existe pasta models/")
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_cached(path: str):
    return load_model(path)

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=['Datetime'])
    return df.sort_values('Datetime').reset_index(drop=True)

# **USE O .h5 AGORA**
MODEL_PATH = "models/model_lstm1.h5"
if not os.path.isfile(MODEL_PATH):
    st.error(f"❌ Arquivo não encontrado: {MODEL_PATH}")
    st.stop()

try:
    model = load_model_cached(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Erro ao carregar modelo HDF5: {e}")
    st.stop()

# Extrai sequência e features
_, seq_length, n_features = model.input_shape

uploaded = st.file_uploader("Faça upload do CSV com colunas Datetime,Open,High,Low,Close,Volume,Ticker", type="csv")
if uploaded:
    df = load_data(uploaded)
    st.subheader("Dados carregados")
    st.dataframe(df.head())

    FEATURES = ['Open','High','Low','Close','Volume']
    if not all(c in df.columns for c in FEATURES):
        st.error(f"Seu CSV precisa conter: {FEATURES}")
        st.stop()

    # Janela de input para LSTM
    last_w = df[FEATURES].values[-seq_length:]
    X = last_w.reshape(1, seq_length, n_features)

    # Previsão
    pred = model.predict(X)[0][0]
    next_date = df['Datetime'].iloc[-1] + timedelta(days=1)
    st.markdown(f"**Data prevista:** {next_date.date()}   **Fechamento previsto:** {pred:.2f}")

    # Plot histórico + previsão
    import plotly.express as px
    hist = df[['Datetime','Close']].rename(columns={'Close':'Preço'})
    fc = pd.DataFrame({'Datetime':[next_date],'Preço':[pred]})
    fig = px.line(hist, x='Datetime', y='Preço', title="Histórico de Fechamento + Previsão")
    fig.add_scatter(x=fc['Datetime'], y=fc['Preço'],
                    mode='markers+text', text=[f"{pred:.2f}"],
                    textposition="bottom center", name='Previsto')
    st.plotly_chart(fig, use_container_width=True)
