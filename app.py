import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import timedelta

st.set_page_config(page_title="Previsão Close Price", layout="wide")
st.title("📈 Previsão de Fechamento (Close) com LSTM")

@st.cache_data
def load_model_cached(path: str):
    return load_model(path)

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df

# Carregando modelo 
model = load_model_cached("models\model_lstm1.keras")

# extrai tamanho da sequência (seq_length) e número de features
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

    # Seleção de janela para previsão
    last_window = df[FEATURES].values[-seq_length:]
    X = last_window.reshape(1, seq_length, n_features)

    # Gera a previsão do próximo dia
    pred = model.predict(X)[0][0]
    next_date = df['Datetime'].iloc[-1] + timedelta(days=1)

    st.markdown(f"**Data prevista:** {next_date.date()}  **Fechamento previsto:** {pred:.2f}")

    # Plota histórico + previsão
    import plotly.express as px
    hist = df[['Datetime','Close']].copy()
    hist = hist.rename(columns={'Close':'Preço'})
    forecast = pd.DataFrame({
        'Datetime': [next_date],
        'Preço': [pred]
    })
    fig = px.line(hist, x='Datetime', y='Preço', title="Histórico de Fechamento + Previsão")
    fig.add_scatter(x=forecast['Datetime'], y=forecast['Preço'],
                    mode='markers+text', text=[f"{pred:.2f}"], textposition="bottom center",
                    name='Previsto')
    st.plotly_chart(fig, use_container_width=True)
