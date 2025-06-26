import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

st.title("Prediksi Penyakit Liver")
st.write("Aplikasi ini memprediksi apakah seorang pasien menderita penyakit liver berdasarkan data medis.")

uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.dataframe(data)

    # Preprocessing, prediction, and results can go here...
