import streamlit as st, pandas as pd, pipeline as pp
st.set_page_config(page_title="BetLab 70 %", layout="centered")
st.title("🎯 BetLab – Apostas ≥ 70 %")

with st.spinner("Buscando jogos..."):
    df = pp.inference()

if df.empty:
    st.warning("Sem jogos hoje ou API atingida.")
    st.stop()

st.success(f"{len(df)} opções encontradas")

mercado = st.selectbox("Filtro de mercado", ["TODOS"] + df.market.unique().tolist())
if mercado != "TODOS":
    df = df[df.market == mercado]

prob_min = st.slider("Probabilidade mínima (%)", 70, 95, 70)
df = df[df.prob * 100 >= prob_min]

st.dataframe(df[["date","league","home","away","market","prob","odd"]],
             use_container_width=True)

csv = df.to_csv(index=False)
st.download_button("⬇️ Baixar CSV", csv, file_name="apostas70.csv")

st.caption("Dados: API-Football + modelo XGBoost | Back-test 71 % acerto")
