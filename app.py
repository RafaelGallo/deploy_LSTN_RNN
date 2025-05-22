import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Previsão Close Price", layout="wide")
st.title("📈 Previsão de Fechamento (Close) com LSTM")

# ─── DEBUG: estrutura e detalhes do arquivo ──────────────────────────────────────
st.subheader("🧐 Debug: Estrutura de arquivos")
# lista raiz
root = os.listdir(".")
st.write("Arquivos/pastas na raiz:", root)

# lista models/
if os.path.isdir("models"):
    lst = os.listdir("models")
    st.write("Arquivos em models/:", lst)
    # para cada item, mostre tipo e tamanho
    for name in lst:
        path = os.path.join("models", name)
        info = os.stat(path)
        st.write(f"- {name}: is_file={os.path.isfile(path)}, is_dir={os.path.isdir(path)}, size={info.st_size} bytes")
else:
    st.write("❌ Não existe pasta `models/`")

# ────────────────────────────────────────────────────────────────────────────────

# Funções de cache
@st.cache_resource
def load_model_cached(path: str):
    return load_model(path)

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df

# Caminho do modelo
MODEL_PATH = "models/model_lstm1.keras"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Caminho não existe: {MODEL_PATH}")
    st.stop()
if os.path.isdir(MODEL_PATH):
    st.error(f"❌ `{MODEL_PATH}` é um diretório, não um arquivo")
    st.stop()

# Tenta carregar
try:
    model = load_model_cached(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Erro ao carregar modelo: {e}")
    st.stop()

# Extrai seq_length e n_features
_, seq_length, n_features = model.input_shape

# Uploader e previsão
uploaded = st.file_uploader("Faça upload do CSV com suas colunas", type=["csv"])
if uploaded:
    df = load_data(uploaded)
    FEATURES = ['Open','High','Low','Close','Volume']
    if not all(col in df.columns for col in FEATURES):
        st.error(f"Seu CSV precisa conter as colunas: {FEATURES}")
        st.stop()
    last_window = df[FEATURES].values[-seq_length:]
    X = last_window.reshape(1, seq_length, n_features)
    pred = model.predict(X)[0][0]
    next_date = df['Datetime'].iloc[-1] + timedelta(days=1)
    st.markdown(f"**Data prevista:** {next_date.date()}  **Close previsto:** {pred:.2f}")
    import plotly.express as px
    hist = df[['Datetime','Close']].rename(columns={'Close':'Preço'})
    forecast = pd.DataFrame({'Datetime':[next_date], 'Preço':[pred]})
    fig = px.line(hist, x='Datetime', y='Preço', title="Histórico + Previsão")
    fig.add_scatter(x=forecast['Datetime'], y=forecast['Preço'],
                    mode='markers+text', text=[f"{pred:.2f}"],
                    textposition="bottom center", name='Previsto')
    st.plotly_chart(fig, use_container_width=True)
