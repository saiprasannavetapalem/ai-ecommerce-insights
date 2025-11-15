import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub

from src.data_prep import load_olist_cached
from src.features import prepare_for_model
from src.train_models import fit_all
from src.llm_utils import summarize


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ecommerce AI — Olist",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_resource
def get_data():
    return load_olist_cached()


df = get_data()

st.title("📦 Ecommerce AI — Olist")
st.write("### Dataset Loaded:", df.shape)
st.dataframe(df.head())


# ---------------------------------------------------------
# KPI METRICS
# ---------------------------------------------------------
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

# On-time %
on_time = (df["delay_days"] <= 0).mean()

# Avg delay
avg_delay = df["delay_days"].mean()

# Repeat in 90 days
repeat_rate = df["repeat_90"].mean()

# Avg order value
aov = df["payment_value"].mean()

# Avg review score
avg_review = df["review_score"].mean()

col1.metric("On-time Delivery %", f"{on_time*100:.1f}%")
col2.metric("Avg Delay (days)", f"{avg_delay:.2f}")
col3.metric("Repeat-90 Rate", f"{repeat_rate*100:.1f}%")
col4.metric("Avg Order Value", f"R$ {aov:.2f}")
col5.metric("Avg Review Score", f"{avg_review:.2f}")


# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------
st.markdown("## 📈 Charts")

tab1, tab2, tab3 = st.tabs(["Delivery Delay", "Payment Value", "Review Score"])

with tab1:
    fig = px.histogram(df, x="delay_days", nbins=40, title="Delivery Delay Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.histogram(df, x="payment_value", nbins=40, title="Payment Value Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.histogram(df, x="review_score", nbins=5, title="Review Score Distribution")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# MACHINE LEARNING MODELS
# ---------------------------------------------------------
st.markdown("## 🤖 Predictive Models")

@st.cache_resource
def train_models_cached(df):
    X, y_delay, y_repeat = prepare_for_model(df)
    return fit_all(X, y_delay, y_repeat)

models = train_models_cached(df)

st.success("Models trained successfully!")


# ---------------------------------------------------------
# WHAT-IF ANALYSIS
# ---------------------------------------------------------
st.markdown("## 🔍 What-If Delivery Prediction")

colA, colB = st.columns(2)

input_value = colA.number_input(
    "Payment Value (R$)",
    min_value=0.0,
    value=100.0
)

input_items = colA.number_input(
    "Number of Items",
    min_value=1,
    value=1
)

input_distance = colA.number_input(
    "Customer Distance (km)",
    min_value=0.0,
    value=20.0
)

if colB.button("Predict Delivery"):
    Xnew = pd.DataFrame([{
        "payment_value": input_value,
        "item_count": input_items,
        "distance_km": input_distance,
    }])
    pred = models["delay_model"].predict(Xnew)[0]
    st.info(f"**Predicted Delivery Delay:** {pred:.2f} days")


# ---------------------------------------------------------
# EXECUTIVE SUMMARY VIA GEMINI
# ---------------------------------------------------------
st.markdown("## 🧠 AI-Generated Business Summary")

if st.button("Generate AI Summary"):
    with st.spinner("Thinking with Gemini…"):
        summary = summarize(df)
    st.success(summary)
