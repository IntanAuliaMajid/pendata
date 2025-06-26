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
import streamlit as st

# --- Data Loading and Initial Understanding ---
@st.cache_data
def load_data():
    file_path = 'Indian Liver Patient Dataset (ILPD).csv'
    df = pd.read_csv(file_path, header=None)
    df.columns = [
        "Age", "Gender", "Urea", "Creatinine", "Hemoglobin", "WBC", "RBC",
        "pH", "Specific Gravity", "Protein", "Class"
    ]
    return df

df = load_data()

st.title("Indian Liver Patient Dataset - Classification Model Deployment")
st.write("This application demonstrates the classification models built on the Indian Liver Patient Dataset.")

st.subheader("Data Understanding")
st.write("The dataset is used to build a classification model that predicts whether a patient suffers from liver disease based on medical parameters.")
st.write("Here's a glimpse of the raw data:")
st.dataframe(df.head())

st.subheader("Data Information")
# Convert columns to numeric, coercing errors
for col in df.columns:
    if col != "Gender":
        df[col] = pd.to_numeric(df[col], errors='coerce')

st.text(df.info())

st.subheader("Descriptive Statistics")
st.dataframe(df.describe())

# --- Data Pre-processing ---
st.subheader("Data Pre-processing")

# Handle Missing Values
st.write("Handling missing values (imputation with median for numerical, mode for categorical).")
numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(include='object').columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna("unknown")

st.write("Missing values after imputation:")
st.dataframe(df.isnull().sum())

# Encoding Categorical Features
st.write("Encoding categorical features (Gender) using Label Encoding.")
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
st.write("Gender column after encoding (0 for Female, 1 for Male if the dataset follows that order or vice-versa depending on LabelEncoder's fit):")
st.dataframe(df['Gender'].head())

# Normalization of Numerical Features
st.write("Normalizing numerical features using MinMaxScaler.")
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
st.write("Normalized numerical features (first 5 rows):")
st.dataframe(df[numeric_cols].head())

# Split Data
st.write("Splitting data into training and testing sets (80% train, 20% test).")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
st.write(f"X_train shape: {X_train.shape}")
st.write(f"X_test shape: {X_test.shape}")
st.write(f"y_train shape: {y_train.shape}")
st.write(f"y_test shape: {y_test.shape}")

# --- Modeling ---
st.subheader("Classification Models")

# Model 1: K-Nearest Neighbors (KNN)
st.write("#### Model 1: K-Nearest Neighbors (KNN)")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred_knn))

# Model 2: Decision Tree Classifier
st.write("#### Model 2: Decision Tree Classifier")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred_dt))

# Model 3: Random Forest + SMOTE
st.write("#### Model 3: Random Forest + SMOTE")
st.write("Applying SMOTE for handling class imbalance in the training data.")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

st.write(f"Distribution before SMOTE: {Counter(y_train)}")
st.write(f"Distribution after SMOTE: {Counter(y_train_resampled)}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_resampled, y_train_resampled)
y_pred_rf_smote = rf_smote.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf_smote):.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred_rf_smote))

# --- Evaluation ---
st.subheader("Model Evaluation: Accuracy Comparison")

acc_knn = accuracy_score(y_test, y_pred_knn)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf_smote = accuracy_score(y_test, y_pred_rf_smote)

df_accuracy = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree', 'Random Forest + SMOTE'],
    'Accuracy': [acc_knn, acc_dt, acc_rf_smote]
})

st.dataframe(df_accuracy)

st.bar_chart(df_accuracy.set_index('Model'))

st.subheader("Conclusion")
st.write("""
After conducting a series of experiments with several classification algorithms, the performance results of three different models were obtained: K-Nearest Neighbors (KNN), Decision Tree, and Random Forest (with additional training techniques). The evaluation was carried out based on prediction accuracy values against test data.

**Key Findings:**
* **Random Forest + SMOTE** yielded the best results with an accuracy of approximately **65.81%**.
* **K-Nearest Neighbors (KNN)** showed a slightly lower accuracy of approximately **64.10%**.
* **Decision Tree Classifier** performed the lowest among the three, with an accuracy of approximately **61.54%**.

**Conclusion:**
Based on the evaluation results, **Random Forest with SMOTE** is the best model for classifying Chronic Kidney Disease on this dataset. Although the accuracy difference is not significantly large compared to KNN, the consistency and stability of the Random Forest model provide a significant advantage. With these results, Random Forest can be used as a baseline model for further development.
""")

# Optional: Add a section for user input for prediction (requires more setup)
# st.sidebar.subheader("Make a Prediction")
# age = st.sidebar.slider("Age", 4, 90, 40)
# gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
# # ... add more input fields for other features
#
# if st.sidebar.button("Predict"):
#     # Process inputs and make a prediction using the best model (rf_smote)
#     pass