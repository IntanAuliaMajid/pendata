import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Coba import SMOTE, tapi jika gagal, set jadi None
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

st.set_page_config(page_title="Analisis Penyakit Hati", page_icon="🩺", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    url = "https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset"
    try:
        df = pd.read_csv(url, header=None)
        df.columns = [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", 
            "Alkaline_Phosphotase", "Alamine_Aminotransferase", 
            "Aspartate_Aminotransferase", "Total_Protiens", "Albumin", 
            "Albumin_and_Globulin_Ratio", "Selector"
        ]
        df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)
        df['Selector'] = df['Selector'].apply(lambda x: 1 if x == 1 else 0)
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari URL. Error: {e}")
        return None


def show_introduction():
    st.title("UAS Penambangan Data: Analisis Penyakit Hati")
    st.markdown("**Nama:** Intan Aulia Majid  \n**NIM:** 230411100001")
    st.header("Dataset Pasien Hati India")
    st.write("""
    Dataset ini digunakan untuk membangun model klasifikasi untuk memprediksi apakah pasien menderita penyakit hati.
    """)
    st.info("Sumber Data: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)")

def show_eda(df):
    st.title("📊 Analisis & Visualisasi Data")
    st.subheader("Tampilan Awal Data")
    st.dataframe(df.head())
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

    st.subheader("Visualisasi Data")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Selector', palette='pastel', ax=ax1)
        ax1.set_title('Distribusi Kelas')
        ax1.set_xticklabels(['Tidak Sakit', 'Sakit Hati'])
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
        ax2.set_title("Heatmap Korelasi Fitur")
        st.pyplot(fig2)

def show_modeling_and_evaluation(X_train_scaled, X_test_scaled, y_train, y_test, X_columns):
    st.title("🧠 Pemodelan & Evaluasi")
    st.sidebar.header("Opsi Model")
    model_choice = st.sidebar.selectbox(
        "Pilih model untuk dievaluasi:",
        ("K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest + SMOTE")
    )
    st.header(f"Hasil Evaluasi: {model_choice}")

    model = None
    if model_choice == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    elif model_choice == "Random Forest + SMOTE":
        if SMOTE is None:
            st.warning("SMOTE tidak tersedia. Model dijalankan tanpa SMOTE.")
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            st.write("Menggunakan SMOTE untuk menyeimbangkan data latih.")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            st.write(f"Sebelum SMOTE: {Counter(y_train)}")
            st.write(f"Setelah SMOTE: {Counter(y_train_resampled)}")
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test_scaled)

    if model:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Tidak Sakit', 'Sakit Hati'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Akurasi Model")
            st.metric("Akurasi", f"{accuracy:.2%}")
            st.subheader("Classification Report")
            st.table(pd.DataFrame(report).transpose())
        with col2:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Prediksi Tidak Sakit', 'Prediksi Sakit'],
                        yticklabels=['Aktual Tidak Sakit', 'Aktual Sakit'])
            ax.set_ylabel('Aktual')
            ax.set_xlabel('Prediksi')
            st.pyplot(fig)

def show_conclusion(X_train_scaled, X_test_scaled, y_train, y_test):
    st.title("📊 Perbandingan & Kesimpulan")

    with st.spinner("Menghitung ulang akurasi model..."):
        acc_knn = accuracy_score(y_test, KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train).predict(X_test_scaled))
        acc_dt = accuracy_score(y_test, DecisionTreeClassifier(random_state=42).fit(X_train_scaled, y_train).predict(X_test_scaled))

        if SMOTE is not None:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
            acc_rf = accuracy_score(y_test, RandomForestClassifier(random_state=42).fit(X_res, y_res).predict(X_test_scaled))
        else:
            acc_rf = accuracy_score(y_test, RandomForestClassifier(random_state=42).fit(X_train_scaled, y_train).predict(X_test_scaled))

    df_akurasi = pd.DataFrame({
        'Model': ['KNN', 'Decision Tree', 'Random Forest + SMOTE' if SMOTE else 'Random Forest'],
        'Akurasi': [acc_knn, acc_dt, acc_rf]
    }).sort_values(by='Akurasi', ascending=False)

    st.subheader("Tabel Perbandingan Akurasi")
    st.table(df_akurasi)

    st.subheader("Visualisasi Perbandingan")
    fig, ax = plt.subplots()
    sns.barplot(data=df_akurasi, x='Akurasi', y='Model', palette='viridis', ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title('Perbandingan Akurasi Model')
    st.pyplot(fig)

    st.header("Kesimpulan")
    st.success(f"**Model Terbaik: {df_akurasi.iloc[0]['Model']}** dengan akurasi **{df_akurasi.iloc[0]['Akurasi']:.2%}**.")

def main():
    st.sidebar.title("Navigasi Aplikasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ("Pendahuluan Proyek", "Analisis & Visualisasi Data", "Pra-Pemrosesan & Pemodelan", "Kesimpulan")
    )

    df = load_and_preprocess_data()

    if df is not None:
        X = df.drop('Selector', axis=1)
        y = df['Selector']
        X_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if page == "Pendahuluan Proyek":
            show_introduction()
        elif page == "Analisis & Visualisasi Data":
            show_eda(df)
        elif page == "Pra-Pemrosesan & Pemodelan":
            show_modeling_and_evaluation(X_train_scaled, X_test_scaled, y_train, y_test, X_columns)
        elif page == "Kesimpulan":
            show_conclusion(X_train_scaled, X_test_scaled, y_train, y_test)
    else:
        st.error("Gagal memuat data.")

if __name__ == "__main__":
    main()
