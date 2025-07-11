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

from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Analisis Penyakit Hati", page_icon="🩺", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    """
    Memuat dan memproses data agar sesuai dengan logika di Jupyter Notebook.
    - Menggunakan nama kolom yang salah.
    - Menangani missing value pada kolom 'Protein'.
    - MENERAPKAN NORMALISASI (MinMaxScaler) pada fitur numerik sebelum splitting.
    """
    try:
        df = pd.read_csv("mybook/Indian Liver Patient Dataset (ILPD).csv", header=None)
        
        df.columns = [
            "Age", "Gender", "Urea", "Creatinine", "Hemoglobin", "WBC", "RBC",
            "pH", "Specific Gravity", "Protein", "Class"
        ]
        
        df['Protein'] = df['Protein'].fillna(df['Protein'].median())
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        
        # *** MENERAPKAN NORMALISASI SEPERTI DI NOTEBOOK ***
        scaler = MinMaxScaler()
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols.remove('Class') # Kolom target tidak dinormalisasi
        
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        return df
    except FileNotFoundError:
        st.error("File 'Indian Liver Patient Dataset (ILPD).csv' tidak ditemukan. Pastikan file berada di direktori yang benar.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat atau memproses dataset: {e}")
        return None

def show_introduction():
    st.title("UAS Penambangan Data: Analisis Penyakit Hati")
    st.markdown("**Nama:** Intan Aulia Majid  \n**NIM:** 230411100001")
    st.header("Dataset Pasien Hati India")
    st.write("Dataset ini digunakan untuk membangun model klasifikasi untuk memprediksi apakah pasien menderita penyakit hati.")
    st.info("Sumber Data: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)")

def show_eda(df):
    st.title("📊 Analisis & Visualisasi Data")
    st.subheader("Tampilan Awal Data")
    st.dataframe(df.head())
    st.subheader("Statistik Deskriptif")
    # Tampilkan statistik sebelum normalisasi untuk insight yang lebih baik
    # Jika ingin menampilkan setelah normalisasi, baris ini bisa di-uncomment
    # st.dataframe(df.describe())

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Class', palette='pastel', ax=ax1, order=[1, 2])
        ax1.set_title('Distribusi Kelas')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Sakit Hati (1)', 'Tidak Sakit (2)'])
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
        ax2.set_title("Heatmap Korelasi Fitur")
        st.pyplot(fig2)

def show_modeling_and_evaluation(X_train, X_test, y_train, y_test):
    """
    Fungsi untuk melatih dan mengevaluasi model pada data yang sudah dinormalisasi.
    """
    st.title("🧠 Pemodelan & Evaluasi")
    st.sidebar.header("Opsi Model")
    model_choice = st.sidebar.selectbox(
        "Pilih model untuk dievaluasi:",
        ("K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest + SMOTE")
    )
    st.header(f"Hasil Evaluasi: {model_choice}")

    y_train_np = np.asarray(y_train)

    if model_choice == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train_np)
        y_pred = model.predict(X_test)

    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train_np)
        y_pred = model.predict(X_test)

    elif model_choice == "Random Forest + SMOTE":
        st.write("Menggunakan SMOTE untuk menyeimbangkan data latih.")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_np)
        
        st.write(f"Distribusi kelas sebelum SMOTE: {Counter(y_train_np)}")
        st.write(f"Distribusi kelas setelah SMOTE: {Counter(y_train_resampled)}")
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_labels = [1, 2] 
    report_target_names = ['Sakit Hati', 'Tidak Sakit']
    report = classification_report(y_test, y_pred, labels=report_labels, target_names=report_target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=report_labels)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Akurasi Model")
        st.metric("Akurasi", f"{accuracy:.6f}") 
        st.subheader("Classification Report")
        st.table(pd.DataFrame(report).transpose())
    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm_xticklabels = ['Prediksi Sakit', 'Prediksi Tidak Sakit']
        cm_yticklabels = ['Aktual Sakit', 'Aktual Tidak Sakit']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=cm_xticklabels,
                      yticklabels=cm_yticklabels)
        ax.set_ylabel('Aktual')
        ax.set_xlabel('Prediksi')
        st.pyplot(fig)

def show_conclusion(X_train, X_test, y_train, y_test):
    st.title("📊 Perbandingan & Kesimpulan")

    y_train_np = np.asarray(y_train)

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train_np)
    acc_knn = accuracy_score(y_test, knn_model.predict(X_test))
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train_np)
    acc_dt = accuracy_score(y_test, dt_model.predict(X_test))

    # Random Forest + SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train_np)
    rf_model = RandomForestClassifier(random_state=42).fit(X_res, y_res)
    acc_rf = accuracy_score(y_test, rf_model.predict(X_test))

    df_akurasi = pd.DataFrame({
        'Model': ['KNN', 'Decision Tree', 'Random Forest + SMOTE'],
        'Akurasi': [acc_knn, acc_dt, acc_rf]
    })
    
    df_akurasi['Model'] = pd.Categorical(df_akurasi['Model'], ["KNN", "Decision Tree", "Random Forest + SMOTE"])
    df_akurasi = df_akurasi.sort_values('Model')

    st.subheader("Tabel Perbandingan Akurasi")
    st.table(df_akurasi.style.format({'Akurasi': "{:.6f}"}))

    st.subheader("Visualisasi Perbandingan")
    fig, ax = plt.subplots()
    sns.barplot(data=df_akurasi, x='Akurasi', y='Model', palette='viridis', ax=ax)
    ax.set_xlim(0.5, 0.8) 
    ax.set_title('Perbandingan Akurasi Model')
    st.pyplot(fig)

    st.header("Kesimpulan")
    df_akurasi_sorted = df_akurasi.sort_values(by='Akurasi', ascending=False)
    st.success(f"**Model Terbaik (sesuai notebook): {df_akurasi_sorted.iloc[0]['Model']}** dengan akurasi **{df_akurasi_sorted.iloc[0]['Akurasi']:.6f}**.")

def main():
    st.sidebar.title("Navigasi Aplikasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ("Pendahuluan Proyek", "Analisis & Visualisasi Data", "Pra-Pemrosesan & Pemodelan", "Kesimpulan")
    )

    # Data sudah dinormalisasi saat dimuat
    df = load_and_preprocess_data()

    if df is not None:
        X = df.drop('Class', axis=1)
        y = df['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if page == "Pendahuluan Proyek":
            show_introduction()
        elif page == "Analisis & Visualisasi Data":
            show_eda(df)
        elif page == "Pra-Pemrosesan & Pemodelan":
            # Data yang dikirim sudah ternormalisasi
            show_modeling_and_evaluation(X_train, X_test, y_train, y_test)
        elif page == "Kesimpulan":
            # Data yang dikirim sudah ternormalisasi
            show_conclusion(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
