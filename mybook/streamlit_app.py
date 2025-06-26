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

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Indian Liver Patient Prediction")

st.title("UAS IF4D: Prediksi Penyakit Hati (Indian Liver Patient Dataset)")
st.write("**Nama: Intan Aulia Majid**")
st.write("**NIM: 230411100001**")
st.write("**Mata Kuliah: Penambangan Data**")

st.header("1. Data Understanding")
st.subheader("Sumber Data")
st.write("Dataset diambil dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset).")

st.subheader("Tujuan Dataset")
st.write("Dataset Indian Liver Patient digunakan untuk membangun model klasifikasi yang mampu memprediksi apakah seorang pasien menderita penyakit hati (liver disease) atau tidak, berdasarkan parameter medis seperti usia, jenis kelamin, kadar bilirubin, enzim hati, protein total, dan lain-lain. Tujuannya adalah untuk membantu diagnosis dini penyakit hati, mengevaluasi performa algoritma machine learning dalam klasifikasi medis, serta mendukung penelitian dan pengembangan sistem cerdas di bidang kesehatan.")

st.subheader("Keterkaitan Fitur-Fitur dalam ILPD:")
st.markdown("""
1.  **Age (Usia)**: Risiko penyakit hati meningkat seiring bertambahnya usia. Usia yang lebih tua sering berkorelasi dengan penurunan fungsi organ, termasuk hati.
2.  **Gender (Jenis Kelamin)**: Beberapa penyakit hati lebih sering terjadi pada pria (misalnya: sirosis alkoholik), sedangkan lainnya mungkin lebih banyak menyerang wanita. Jenis kelamin bisa memengaruhi pola konsumsi alkohol, hormon, dan respons imun.
3.  **Total Bilirubin**: Bilirubin adalah produk samping pemecahan sel darah merah. Kadar tinggi menandakan masalah pada hati dalam memproses dan membuang bilirubin — gejala umum penyakit hati, terutama hepatitis.
4.  **Direct Bilirubin**: Merupakan bentuk terkonjugasi dari bilirubin. Peningkatan nilai ini menunjukkan adanya obstruksi atau kerusakan saluran empedu, yang umum dalam penyakit hati.
5.  **Alkaline Phosphatase (ALP)**: Enzim yang meningkat bila terjadi gangguan pada saluran empedu dan kerusakan jaringan hati. Nilai tinggi dapat menjadi penanda adanya penyakit hati kolestatik.
6.  **Alanine Aminotransferase (SGPT/ALT)**: Enzim ini dilepaskan ke dalam darah saat sel-sel hati rusak. Merupakan indikator utama kerusakan hati akut atau kronis.
7.  **Aspartate Aminotransferase (SGOT/AST)**: Mirip dengan ALT, namun juga ditemukan pada jantung dan otot. Kadar tinggi sering terlihat pada hepatitis, sirosis, dan penyakit hati alkoholik.
8.  **Total Proteins**: Mengukur jumlah total protein dalam darah, termasuk albumin dan globulin. Hati yang sehat memproduksi banyak protein, sehingga nilainya bisa menurun jika hati rusak.
9.  **Albumin**: Protein utama yang diproduksi oleh hati. Jika hati rusak, kemampuan produksinya menurun, sehingga kadar albumin bisa rendah.
10. **Albumin and Globulin Ratio (A/G Ratio)**: Rasio antara albumin dan globulin. Ketidakseimbangan rasio ini (terutama rasio rendah) bisa menjadi indikasi gangguan fungsi hati atau penyakit inflamasi kronis.
11. **Dataset Column (Output)**: Kolom target yang menunjukkan apakah pasien menderita penyakit liver (1) atau tidak (2). Semua fitur di atas digunakan untuk memprediksi nilai ini.
""")

st.header("2. Data Loading and Initial Exploration")

# Load Data
file_path = "Indian Liver Patient Dataset (ILPD).csv"
df = pd.read_csv(file_path, header=None)

df.columns = [
    "Age", "Gender", "Urea", "Creatinine", "Hemoglobin", "WBC", "RBC",
    "pH", "Specific Gravity", "Protein", "Class"
]

for col in df.columns:
    if col != "Gender":
        df[col] = pd.to_numeric(df[col], errors='coerce')

st.subheader("Data Info")
st.write(df.info())
st.subheader("First 5 Rows of Data")
st.dataframe(df.head())
st.subheader("Descriptive Statistics")
st.dataframe(df.describe())

st.subheader("Distribution of Numeric Features (Histograms)")
numeric_df = df.select_dtypes(include=[np.number])
fig_hist = numeric_df.hist(bins=20, figsize=(16, 12), edgecolor='black')
plt.suptitle("Distribusi Fitur Numerik", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
st.pyplot(fig_hist[0][0].figure) # Pass the figure object to st.pyplot

st.subheader("Heatmap of Numeric Feature Correlation")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
plt.title("Heatmap Korelasi Fitur Numerik")
st.pyplot(fig_corr)

st.subheader("Hemoglobin vs Urea by Class (Scatter Plot)")
fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x='Hemoglobin', y='Urea', hue='Class', palette='Set1', ax=ax_scatter)
plt.title("Hemoglobin vs Urea berdasarkan Kelas")
plt.xlabel("Hemoglobin")
plt.ylabel("Urea")
plt.legend(title="Class")
plt.grid(True)
st.pyplot(fig_scatter)

st.subheader("Missing Values")
missing_counts = df.isnull().sum()
st.write("Missing values per column:")
st.dataframe(missing_counts)

total_missing = missing_counts.sum()
st.write(f"Total missing values in the entire dataset: {total_missing}")

st.subheader("Visualisasi Missing Values (Heatmap)")
fig_missing, ax_missing = plt.subplots(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax_missing)
plt.title("Visualisasi Missing Values (NaN) di Setiap Kolom")
plt.xlabel("Kolom")
plt.ylabel("Baris")
st.pyplot(fig_missing)

st.header("3. Data Pre-processing")

# Handle Missing Values (Imputation)
numerik_cols = df.select_dtypes(include='number').columns
kategori_cols = df.select_dtypes(include='object').columns

df[numerik_cols] = df[numerik_cols].fillna(df[numerik_cols].median())

for col in kategori_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna("unknown")

st.subheader("Missing Values After Imputation")
st.write(df.info())

# Encoding Categorical Features (Label Encoding)
st.subheader("Encoding Categorical Features (Label Encoding)")
le = LabelEncoder()
for col in kategori_cols:
    df[col] = le.fit_transform(df[col])
st.write("Transformed 'Gender' column (first 5 rows):")
st.dataframe(df[kategori_cols].head())

# Normalisasi Fitur Numerik
st.subheader("Normalisasi Fitur Numerik")
scaler = MinMaxScaler()
df[numerik_cols] = scaler.fit_transform(df[numerik_cols])
st.write("Normalized numeric features (first 5 rows):")
st.dataframe(df[numerik_cols].head())

# Split Data
st.subheader("Data Splitting")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.write(f"X_train shape: {X_train.shape}")
st.write(f"X_test shape: {X_test.shape}")
st.write(f"y_train shape: {y_train.shape}")
st.write(f"y_test shape: {y_test.shape}")

st.header("4. Modelling")

st.subheader("Model 1: K-Nearest Neighbors (KNN)")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred_knn))

st.subheader("Model 2: Decision Tree Classifier")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred_dt))

st.subheader("Model 3: Random Forest + SMOTE")
st.write("Applying SMOTE to balance the training data...")
st.write(f"Distribution before SMOTE: {Counter(y_train)}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
st.write(f"Distribution after SMOTE: {Counter(y_train_resampled)}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_resampled, y_train_resampled)
y_pred_rf_smote = rf_smote.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf_smote):.4f}")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred_rf_smote))

st.header("5. Evaluation")
st.subheader("Model Accuracy Comparison")

acc_knn = accuracy_score(y_test, y_pred_knn)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf_smote = accuracy_score(y_test, y_pred_rf_smote)

df_accuracy = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree', 'Random Forest + SMOTE'],
    'Akurasi': [acc_knn, acc_dt, acc_rf_smote]
})

st.dataframe(df_accuracy)

st.subheader("Visualisasi Akurasi Model")
fig_acc_bar, ax_acc_bar = plt.subplots(figsize=(8, 5))
ax_acc_bar.bar(df_accuracy['Model'], df_accuracy['Akurasi'], color=['skyblue', 'lightgreen', 'salmon'])
ax_acc_bar.set_ylim(0, 1)
ax_acc_bar.set_ylabel('Akurasi')
ax_acc_bar.set_title('Perbandingan Akurasi Model Klasifikasi')
ax_acc_bar.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig_acc_bar)

st.header("Conclusion")
st.markdown("""
Setelah dilakukan serangkaian percobaan dengan beberapa algoritma klasifikasi, diperoleh hasil performa dari tiga model berbeda, yaitu K-Nearest Neighbors (KNN), Decision Tree, dan Random Forest (dengan teknik pelatihan tambahan). Evaluasi dilakukan berdasarkan nilai akurasi prediksi terhadap data uji.

**Kesimpulan:**
Random Forest + SMOTE memberi hasil terbaik (65%)
Berdasarkan hasil evaluasi, Random Forest adalah model terbaik yang digunakan dalam proses klasifikasi Chronic Kidney Disease pada dataset ini. Meskipun perbedaan akurasinya tidak terlalu besar dibandingkan KNN, konsistensi dan kestabilan model Random Forest memberikan keunggulan yang signifikan. Dengan hasil ini, Random Forest dapat dijadikan model dasar untuk pengembangan selanjutnya.
""")