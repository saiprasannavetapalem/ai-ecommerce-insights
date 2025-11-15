
import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub

from src.data_prep import load_olist_cached

st.set_page_config(page_title="Ecommerce AI — Olist", layout="wide")

@st.cache_resource
def get_data():
    return load_olist_cached()

df = get_data()

st.title("📦 Ecommerce AI — Olist")
st.write("Dataset Loaded:", df.shape)

st.dataframe(df.head())
