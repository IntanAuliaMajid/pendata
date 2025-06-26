# app.py (your Streamlit application file)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

st.set_page_config(layout="wide")

st.title("Aplikasi Prediksi Penyakit Hati (Indian Liver Patient Dataset)")
st.markdown("---")

# Data Understanding
st.header("1. Data Understanding")
st.write("""
Dataset Indian Liver Patient digunakan untuk membangun model klasifikasi yang mampu memprediksi apakah seorang pasien menderita penyakit hati (liver disease) atau tidak, berdasarkan parameter medis seperti usia, jenis kelamin, kadar bilirubin, enzim hati, protein total, dan lain-lain. Tujuannya adalah untuk membantu diagnosis dini penyakit hati, mengevaluasi performa algoritma machine learning dalam klasifikasi medis, serta mendukung penelitian dan pengembangan sistem cerdas di bidang kesehatan.
""")

st.subheader("Keterkaitan Fitur-Fitur dalam ILPD:")
st.markdown("""
- **Age (Usia)**: Risiko penyakit hati meningkat seiring bertambahnya usia.
- **Gender (Jenis Kelamin)**: Mempengaruhi pola konsumsi alkohol, hormon, dan respons imun.
- **Total Bilirubin**: Kadar tinggi menandakan masalah pada hati dalam memproses dan membuang bilirubin.
- **Direct Bilirubin**: Peningkatan nilai ini menunjukkan adanya obstruksi atau kerusakan saluran empedu.
- **Alkaline Phosphatase (ALP)**: Enzim yang meningkat bila terjadi gangguan pada saluran empedu dan kerusakan jaringan hati.
- **Alanine Aminotransferase (SGPT/ALT)**: Indikator utama kerusakan hati akut atau kronis.
- **Aspartate Aminotransferase (SGOT/AST)**: Kadar tinggi sering terlihat pada hepatitis, sirosis, dan penyakit hati alkoholik.
- **Total Proteins**: Hati yang sehat memproduksi banyak protein, sehingga nilainya bisa menurun jika hati rusak.
- **Albumin**: Protein utama yang diproduksi oleh hati. Jika hati rusak, kemampuan produksinya menurun.
- **Albumin and Globulin Ratio (A/G Ratio)**: Ketidakseimbangan rasio ini bisa menjadi indikasi gangguan fungsi hati.
- **Dataset Column (Output)**: Kolom target yang menunjukkan apakah pasien menderita penyakit liver (1) atau tidak (2).
""")

# Load Data
st.subheader("Import dan Load Data")
file_path = "Indian Liver Patient Dataset (ILPD).csv"
try:
    df = pd.read_csv(file_path, header=None)
    df.columns = [
        "Age", "Gender", "Urea", "Creatinine", "Hemoglobin", "WBC", "RBC",
        "pH", "Specific Gravity", "Protein", "Class"
    ]
    for col in df.columns:
        if col != "Gender":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    st.success("Data berhasil dimuat!")
    st.write("Menampilkan 5 baris pertama data:")
    st.dataframe(df.head())
    st.write("Informasi umum dataset:")
    st.write(df.info())
    st.write("Statistik deskriptif dataset:")
    st.dataframe(df.describe())
except FileNotFoundError:
    st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# Data Visualization (moved here for better flow in Streamlit)
st.subheader("Visualisasi Data")

st.markdown("### Histogram Fitur Numerik (Distribusi Data)")
numeric_df = df.select_dtypes(include=[np.number])
fig, axes = plt.subplots(nrows=(len(numeric_df.columns) + 1) // 2, ncols=2, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_df.columns):
    axes[i].hist(numeric_df[col], bins=20, edgecolor='black', color='skyblue')
    axes[i].set_title(f'Distribusi: {col}')
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Frekuensi")
plt.tight_layout()
st.pyplot(fig)

st.markdown("### Heatmap Korelasi Fitur Numerik")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
ax_corr.set_title("Heatmap Korelasi Fitur Numerik")
st.pyplot(fig_corr)

st.markdown("### Scatterplot Hemoglobin vs Urea berdasarkan Kelas")
fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x='Hemoglobin', y='Urea', hue='Class', palette='Set1', ax=ax_scatter)
ax_scatter.set_title("Hemoglobin vs Urea berdasarkan Kelas")
ax_scatter.set_xlabel("Hemoglobin")
ax_scatter.set_ylabel("Urea")
ax_scatter.legend(title="Class")
ax_scatter.grid(True)
st.pyplot(fig_scatter)

st.markdown("### Visualisasi Missing Values (Heatmap)")
fig_missing, ax_missing = plt.subplots(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax_missing)
ax_missing.set_title("Visualisasi Missing Values (NaN) di Setiap Kolom")
ax_missing.set_xlabel("Kolom")
ax_missing.set_ylabel("Baris")
st.pyplot(fig_missing)

st.markdown("### Boxplot Fitur Numerik (Deteksi Outlier)")
fig_boxplot, axes_boxplot = plt.subplots(nrows=(len(numeric_df.columns) + 1) // 2, ncols=2, figsize=(16, len(numeric_df.columns)*1.5))
axes_boxplot = axes_boxplot.flatten()
for i, col in enumerate(numeric_df.columns):
    sns.boxplot(data=df, x=col, color='skyblue', ax=axes_boxplot[i])
    axes_boxplot[i].set_title(f'Boxplot: {col}')
    axes_boxplot[i].set_xlabel("")
plt.tight_layout()
st.pyplot(fig_boxplot)

# Pre-processing Data
st.header("2. Pre-processing Data")

st.subheader("Tangani Missing Values")
numerik_cols = df.select_dtypes(include='number').columns
kategori_cols = df.select_dtypes(include='object').columns

df[numerik_cols] = df[numerik_cols].fillna(df[numerik_cols].median())

for col in kategori_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna("unknown")
st.success("Missing values telah ditangani!")
st.write(df.info())

st.subheader("Encoding Fitur Kategorikal (Label Encoding)")
le = LabelEncoder()
for col in kategori_cols:
    df[col] = le.fit_transform(df[col])
st.success("Fitur kategorikal telah di-encode!")
st.dataframe(df[kategori_cols].head())

st.subheader("Normalisasi Fitur Numerik")
scaler = MinMaxScaler()
df[numerik_cols] = scaler.fit_transform(df[numerik_cols])
st.success("Fitur numerik telah dinormalisasi!")
st.dataframe(df[numerik_cols].head())

st.subheader("Split Data")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
st.success("Data telah dibagi menjadi training dan testing set!")
st.write(f"X_train shape: {X_train.shape}")
st.write(f"X_test shape: {X_test.shape}")
st.write(f"y_train shape: {y_train.shape}")
st.write(f"y_test shape: {y_test.shape}")

# Modelling
st.header("3. Modelling")

st.subheader("Model 1: K-Nearest Neighbors (KNN)")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
st.write("### Hasil KNN:")
st.write(f"Akurasi: {accuracy_score(y_test, y_pred_knn):.4f}")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred_knn))

st.subheader("Model 2: Decision Tree Classifier")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
st.write("### Hasil Decision Tree:")
st.write(f"Akurasi: {accuracy_score(y_test, y_pred_dt):.4f}")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred_dt))

st.subheader("Model 3: Random Forest + SMOTE")
st.write("Distribusi sebelum SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
st.write("Distribusi setelah SMOTE:", Counter(y_train_resampled))

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_resampled, y_train_resampled)
y_pred_rf_smote = rf_smote.predict(X_test)
st.write("### Hasil Random Forest + SMOTE:")
st.write(f"Akurasi: {accuracy_score(y_test, y_pred_rf_smote):.4f}")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred_rf_smote))

# Evaluation
st.header("4. Evaluasi")
st.subheader("Perbandingan Akurasi Model")

acc_knn = accuracy_score(y_test, y_pred_knn)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf_smote = accuracy_score(y_test, y_pred_rf_smote)

df_akurasi = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree', 'Random Forest + SMOTE'],
    'Akurasi': [acc_knn, acc_dt, acc_rf_smote]
})

st.dataframe(df_akurasi)

fig_eval, ax_eval = plt.subplots(figsize=(8, 5))
ax_eval.bar(df_akurasi['Model'], df_akurasi['Akurasi'], color=['skyblue', 'lightgreen', 'salmon'])
ax_eval.set_ylim(0, 1)
ax_eval.set_ylabel('Akurasi')
ax_eval.set_title('Perbandingan Akurasi Model Klasifikasi')
ax_eval.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig_eval)

st.subheader("Kesimpulan:")
st.write("""
Setelah dilakukan serangkaian percobaan dengan beberapa algoritma klasifikasi, diperoleh hasil performa dari tiga model berbeda, yaitu K-Nearest Neighbors (KNN), Decision Tree, dan Random Forest (dengan teknik pelatihan tambahan). Evaluasi dilakukan berdasarkan nilai akurasi prediksi terhadap data uji.

**Random Forest + SMOTE memberi hasil terbaik (65%)**
Berdasarkan hasil evaluasi, Random Forest adalah model terbaik yang digunakan dalam proses klasifikasi Chronic Kidney Disease pada dataset ini. Meskipun perbedaan akurasinya tidak terlalu besar dibandingkan KNN, konsistensi dan kestabilan model Random Forest memberikan keunggulan yang signifikan. Dengan hasil ini, Random Forest dapat dijadikan model dasar untuk pengembangan selanjutnya.
""")