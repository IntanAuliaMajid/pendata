import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b2553b5b",
   "metadata": {},
   "source": [
    "# UAS IF4D\n",
    "**Nama: Intan Aulia Majid** \n",
    "**NIM: 230411100001** \n",
    "**Mata Kuliah: Penambangan Data**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dec863ec",
   "metadata": {},
   "source": [
    "## **DATA PASIEN HATI INDIA (Indian Liver Patient Dataset)**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "182574b1",
   "metadata": {},
   "source": [
    "## Data Understanding\n",
    "Sumber Data\n",
    "Dataset diambil dari link dibawah ini:\n",
    "\n",
    "https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset\n",
    "\n",
    "**Tujuan Dataset** \n",
    "Dataset Indian Liver Patient digunakan untuk membangun model klasifikasi yang mampu memprediksi apakah seorang pasien menderita penyakit hati (liver disease) atau tidak, berdasarkan parameter medis seperti usia, jenis kelamin, kadar bilirubin, enzim hati, protein total, dan lain-lain. Tujuannya adalah untuk membantu diagnosis dini penyakit hati, mengevaluasi performa algoritma machine learning dalam klasifikasi medis, serta mendukung penelitian dan pengembangan sistem cerdas di bidang kesehatan."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ec9d117",
   "metadata": {},
   "source": [
    "**Keterkaitan Fitur-Fitur dalam ILPD :** \n",
    "1. Age (Usia)  \n",
    "Risiko penyakit hati meningkat seiring bertambahnya usia. Usia yang lebih tua sering berkorelasi dengan penurunan fungsi organ, termasuk hati.\n",
    "\n",
    "2. Gender (Jenis Kelamin)  \n",
    "Beberapa penyakit hati lebih sering terjadi pada pria (misalnya: sirosis alkoholik), sedangkan lainnya mungkin lebih banyak menyerang wanita. Jenis kelamin bisa memengaruhi pola konsumsi alkohol, hormon, dan respons imun.  \n",
    "\n",
    "3. Total Bilirubin  \n",
    "Bilirubin adalah produk samping pemecahan sel darah merah. Kadar tinggi menandakan masalah pada hati dalam memproses dan membuang bilirubin — gejala umum penyakit hati, terutama hepatitis.  \n",
    "\n",
    "4. Direct Bilirubin  \n",
    "Merupakan bentuk terkonjugasi dari bilirubin. Peningkatan nilai ini menunjukkan adanya obstruksi atau kerusakan saluran empedu, yang umum dalam penyakit hati.  \n",
    "\n",
    "5. Alkaline Phosphatase (ALP)  \n",
    "Enzim yang meningkat bila terjadi gangguan pada saluran empedu dan kerusakan jaringan hati. Nilai tinggi dapat menjadi penanda adanya penyakit hati kolestatik.  \n",
    "\n",
    "6. Alanine Aminotransferase (SGPT/ALT)  \n",
    "Enzim ini dilepaskan ke dalam darah saat sel-sel hati rusak. Merupakan indikator utama kerusakan hati akut atau kronis.  \n",
    "\n",
    "7. Aspartate Aminotransferase (SGOT/AST)  \n",
    "Mirip dengan ALT, namun juga ditemukan pada jantung dan otot. Kadar tinggi sering terlihat pada hepatitis, sirosis, dan penyakit hati alkoholik.  \n",
    "\n",
    "8. Total Proteins  \n",
    "Mengukur jumlah total protein dalam darah, termasuk albumin dan globulin. Hati yang sehat memproduksi banyak protein, sehingga nilainya bisa menurun jika hati rusak.  \n",
    "\n",
    "9. Albumin  \n",
    "Protein utama yang diproduksi oleh hati. Jika hati rusak, kemampuan produksinya menurun, sehingga kadar albumin bisa rendah.  \n",
    "\n",
    "10. Albumin and Globulin Ratio (A/G Ratio)  \n",
    "Rasio antara albumin dan globulin. Ketidakseimbangan rasio ini (terutama rasio rendah) bisa menjadi indikasi gangguan fungsi hati atau penyakit inflamasi kronis.  \n",
    "\n",
    "11. Dataset Column (Output)  \n",
    "Kolom target yang menunjukkan apakah pasien menderita penyakit liver (1) atau tidak (2). Semua fitur di atas digunakan untuk memprediksi nilai ini.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7518e312",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: ucimlrepo in c:\\users\\intan\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (0.0.7)\n",
      "Requirement already satisfied: pandas>=1.0.0 in c:\\users\\intan\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from ucimlrepo) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2020.12.5 in c:\\users\\intan\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from ucimlrepo) (2024.2.2)\n",
      "Requirement already satisfied: numpy>=1.23.2 in c:\\users\\intan\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas>=1.0.0->ucimlrepo) (1.26.1)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\intan\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas>=1.0.0->ucimlrepo) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\intan\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas>=1.0.0->ucimlrepo) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\intan\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas>=1.0.0->ucimlrepo) (2024.1)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\intan\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from python-dateutil>=2.8.2->pandas>=1.0.0->ucimlrepo) (1.16.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install ucimlrepo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2175bf1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'uci_id': 225, 'name': 'ILPD (Indian Liver Patient Dataset)', 'repository_url': 'https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset', 'data_url': 'https://archive.ics.uci.edu/static/public/225/data.csv', 'abstract': 'Death by liver cirrhosis continues to increase, given the increase in alcohol consumption rates, chronic hepatitis infections, and obesity-related liver disease. Notwithstanding the high mortality of this disease, liver diseases do not affect all sub-populations equally. The early detection of pathology is a determinant of patient outcomes, yet female patients appear to be marginalized when it comes to early diagnosis of liver pathology. \\nThe dataset comprises 584 patient records collected from the NorthEast of Andhra Pradesh, India.\\nThe prediction task is to determine whether a patient suffers from liver disease based on the information about several biochemical markers, including albumin and other enzymes required for metabolism.\\n', 'area': 'Health and Medicine', 'tasks': ['Classification'], 'characteristics': ['Multivariate'], 'num_instances': 583, 'num_features': 10, 'feature_types': ['Integer', 'Real'], 'demographics': ['Age', 'Gender'], 'target_col': ['Selector'], 'index_col': None, 'has_missing_values': 'no', 'missing_values_symbol': None, 'year_of_dataset_creation': 2022, 'last_updated': 'Fri Nov 03 2023', 'dataset_doi': '10.24432/C5D02C', 'creators': ['Bendi Ramana', 'N. Venkateswarlu'], 'intro_paper': {'ID': 242, 'type': 'NATIVE', 'title': 'Investigating for bias in healthcare algorithms: a sex-stratified analysis of supervised machine learning models in liver disease prediction', 'authors': 'I. Straw, Honghan Wu', 'venue': 'BMJ Health & Care Informatics', 'year': 2022, 'journal': None, 'DOI': '10.1136%2Fbmjhci-2021-100457', 'URL': 'https://www.semanticscholar.org/paper/df37b91a72fb4fb11dc9ac3d63c1479428e4f14d', 'sha': None, 'corpus': None, 'arxiv': None, 'mag': None, 'acl': None, 'pmid': None, 'pmcid': None}, 'additional_info': {'summary': \"This data set contains records of 416 patients diagnosed with liver disease and 167 patients without liver disease. This information is contained in the class label named 'Selector'.  There are 10 variables per patient: age, gender, total Bilirubin, direct Bilirubin, total proteins, albumin, A/G ratio, SGPT, SGOT and Alkphos. Of the 583 patient records, 441 are male, and 142 are female. \\n\\nThe current dataset has been used to study \\n- differences in patients across US and Indian patients that suffer from liver diseases.\\n- gender-based disparities in predicting liver disease, as previous studies have found that biochemical markers do not have the same effectiveness for male and female patients.\\n\", 'purpose': None, 'funded_by': None, 'instances_represent': 'Medical patients', 'recommended_data_splits': None, 'sensitive_data': 'Yes. The data contains information about the age and gender of the patients.', 'preprocessing_description': 'Any patient whose age exceeded 89 is listed as being of age \"90\".', 'variable_info': None, 'citation': 'The original dataset was first proposed by Ramana et al. (2012) as a critical comparison of patients across USA and India:\\nRamana, Bendi & Surendra, M & Babu, Prasad & Bala Venkateswarlu, Nagasuri. (2012). A Critical Comparative Study of Liver Patients from USA and INDIA: An Exploratory Analysis. International Journal of Computer Science. 9. '}}\n",
      "         name     role        type demographic  \\\n",
      "0         Age  Feature     Integer         Age   \n",
      "1      Gender  Feature      Binary      Gender   \n",
      "2          TB  Feature  Continuous        None   \n",
      "3          DB  Feature  Continuous        None   \n",
      "4     Alkphos  Feature     Integer        None   \n",
      "5        Sgpt  Feature     Integer        None   \n",
      "6        Sgot  Feature     Integer        None   \n",
      "7          TP  Feature  Continuous        None   \n",
      "8         ALB  Feature  Continuous        None   \n",
      "9   A/G Ratio  Feature  Continuous        None   \n",
      "10   Selector   Target      Binary        None   \n",
      "\n",
      "                                          description  units missing_values  \n",
      "0   Age of the patient. Any patient whose age exce...  years             no  \n",
      "1                               Gender of the patient   None             no  \n",
      "2                                     Total Bilirubin   None             no  \n",
      "3                                    Direct Bilirubin   None             no  \n",
      "4                                Alkaline Phosphotase   None             no  \n",
      "5                            Alamine Aminotransferase   None             no  \n",
      "6                          Aspartate Aminotransferase   None             no  \n",
      "7                                      Total Proteins   None             no  \n",
      "8                                             Albumin   None             no  \n",
      "9                          Albumin and Globulin Ratio   None             no  \n",
      "10  Selector field used to split the data into two...   None             no  \n"
     ]
    }
   ],
   "source": [
    "from ucimlrepo import fetch_ucirepo \n",
    "ilpd_indian_liver_patient_dataset = fetch_ucirepo(id=225) \n",
    "X = ilpd_indian_liver_patient_dataset.data.features \n",
    "y = ilpd_indian_liver_patient_dataset.data.targets \n",
    "\n",
    "print(ilpd_indian_liver_patient_dataset.metadata) \n",
    "print(ilpd_indian_liver_patient_dataset.variables) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3022a21d",
   "metadata": {},
   "source": [
    "**Import dan Load Data**\n",
    "\n",
    "Pertama, kita akan memuat dataset ke dalam DataFrame pandas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6026e3b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "file_path = \"Indian Liver Patient Dataset (ILPD).csv\"\n",
    "df = pd.read_csv(file_path)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "71715d50",
   "metadata": {},
   "source": [
    "**Menampilkan Info Data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "fcd6f22c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 582 entries, 0 to 581\n",
      "Data columns (total 11 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   65      582 non-null    int64  \n",
      " 1   Female  582 non-null    object \n",
      " 2   0.7     582 non-null    float64\n",
      " 3   0.1     582 non-null    float64\n",
      " 4   187     582 non-null    int64  \n",
      " 5   16      582 non-null    int64  \n",
      " 6   18      582 non-null    int64  \n",
      " 7   6.8     582 non-null    float64\n",
      " 8   3.3     582 non-null    float64\n",
      " 9   0.9     578 non-null    float64\n",
      " 10  1       582 non-null    int64  \n",
      "dtypes: float64(5), int64(5), object(1)\n",
      "memory usage: 50.1+ KB\n",
      "None\n",
      "   65 Female   0.7  0.1  187  16   18  6.8  3.3   0.9  1\n",
      "0  62   Male  10.9  5.5  699  64  100  7.5  3.2  0.74  1\n",
      "1  62   Male   7.3  4.1  490  60   68  7.0  3.3  0.89  1\n",
      "2  58   Male   1.0  0.4  182  14   20  6.8  3.4  1.00  1\n",
      "3  72   Male   3.9  2.0  195  27   59  7.3  2.4  0.40  1\n",
      "4  46   Male   1.8  0.7  208  19   14  7.6  4.4  1.30  1\n"
     ]
    }
   ],
   "source": [
    "print(df.info())\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dc7a5b87",
   "metadata": {},
   "source": [
    "**Visualisasi Data**\n",
    "\n",
    "Histogram Fitur Numerik (Distribusi Data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "912cd18b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 583 entries, 0 to 582\n",
      "Data columns (total 11 columns):\n",
      " #   Column            Non-Null Count  Dtype  \n",
      "---  ------            --------------  -----  \n",
      " 0   Age               583 non-null    int64  \n",
      " 1   Gender            583 non-null    object \n",
      " 2   Urea              583 non-null    float64\n",
      " 3   Creatinine        583 non-null    float64\n",
      " 4   Hemoglobin        583 non-null    int64  \n",
      " 5   WBC               583 non-null    int64  \n",
      " 6   RBC               583 non-null    int64  \n",
      " 7   pH                583 non-null    float64\n",
      " 8   Specific Gravity  583 non-null    float64\n",
      " 9   Protein           579 non-null    float64\n",
      " 10  Class             583 non-null    int64  \n",
      "dtypes: float64(5), int64(5), object(1)\n",
      "memory usage: 50.2+ KB\n",
      "None\n",
      "              Age        Urea  Creatinine   Hemoglobin          WBC  \\\n",
      "count  583.000000  583.000000  583.000000   583.000000   583.000000   \n",
      "mean    44.746141    3.298799    1.486106   290.576329    80.713551   \n",
      "std     16.189833    6.209522    2.808498   242.937989   182.620356   \n",
      "min      4.000000    0.400000    0.100000    63.000000    10.000000   \n",
      "25%     33.000000    0.800000    0.200000   175.500000    23.000000   \n",
      "50%     45.000000    1.000000    0.300000   208.000000    35.000000   \n",
      "75%     58.000000    2.600000    1.300000   298.000000    60.500000   \n",
      "max     90.000000   75.000000   19.700000  2110.000000  2000.000000   \n",
      "\n",
      "               RBC          pH  Specific Gravity     Protein       Class  \n",
      "count   583.000000  583.000000        583.000000  579.000000  583.000000  \n",
      "mean    109.910806    6.483190          3.141852    0.947064    1.286449  \n",
      "std     288.918529    1.085451          0.795519    0.319592    0.452490  \n",
      "min      10.000000    2.700000          0.900000    0.300000    1.000000  \n",
      "25%      25.000000    5.800000          2.600000    0.700000    1.000000  \n",
      "50%      42.000000    6.600000          3.100000    0.930000    1.000000  \n",
      "75%      87.000000    7.200000          3.800000    1.100000    2.000000  \n",
      "max    4929.000000    9.600000          5.500000    2.800000    2.000000  \n"
     ]
    }
   ],
   "source": [
    "df.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']\n",
    "\n",
    "for column in df.columns:\n",
    "    if df[column].dtype == 'object':\n",
    "        plt.figure(figsize=(8, 6))\n",
    "        sns.countplot(data=df, x=column, hue='Dataset', palette='viridis')\n",
    "        plt.title(f'Countplot of {column} by Dataset')\n",
    "        plt.xlabel(column)\n",
    "        plt.ylabel('Count')\n",
    "        plt.show()\n",
    "    else:\n",
    "        plt.figure(figsize=(10, 6))\n",
    "        sns.histplot(data=df, x=column, hue='Dataset', kde=True, palette='viridis')\n",
    "        plt.title(f'Distribution of {column} by Dataset')\n",
    "        plt.xlabel(column)\n",
    "        plt.ylabel('Frequency')\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e43b67e4",
   "metadata": {},
   "source": [
    "**Pre-processing Data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13143c7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "# Mengganti nilai 2 (pasien tidak berpenyakit hati) menjadi 0\n",
    "df['Dataset'] = df['Dataset'].replace(2, 0)\n",
    "\n",
    "# Handle missing values (NaN) pada Albumin_and_Globulin_Ratio\n",
    "# Imputasi dengan mean\n",
    "imputer = SimpleImputer(strategy='mean')\n",
    "df['Albumin_and_Globulin_Ratio'] = imputer.fit_transform(df[['Albumin_and_Globulin_Ratio']])\n",
    "\n",
    "# Label Encoding untuk Kolom 'Gender'\n",
    "le = LabelEncoder()\n",
    "df['Gender'] = le.fit_transform(df['Gender'])\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "673752e5",
   "metadata": {},
   "source": [
    "**Split Data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "81f1b0a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Memisahkan fitur (X) dan target (y)\n",
    "X = df.drop('Dataset', axis=1)\n",
    "y = df['Dataset']\n",
    "\n",
    "# Split data menjadi training dan testing set\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "226d7f99",
   "metadata": {},
   "source": [
    "**Normalisasi Data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d3e9114",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "scaler = MinMaxScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)\n",
    "X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97e16335",
   "metadata": {},
   "source": [
    "**Modeling**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5f0b5d5",
   "metadata": {},
   "source": [
    "**1. KNN (K-Nearest Neighbors)**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "921865a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "\n",
    "# Inisialisasi model KNN\n",
    "knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors bisa disesuaikan\n",
    "\n",
    "# Latih model\n",
    "knn.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Prediksi pada data uji\n",
    "y_pred_knn = knn.predict(X_test_scaled)\n",
    "\n",
    "# Evaluasi model\n",
    "accuracy_knn = accuracy_score(y_test, y_pred_knn)\n",
    "report_knn = classification_report(y_test, y_pred_knn)\n",
    "\n",
    "print(f\"Akurasi KNN: {accuracy_knn:.2f}\")\n",
    "print(\"Classification Report KNN:\")\n",
    "print(report_knn)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "18f9189b",
   "metadata": {},
   "source": [
    "**2. Decision Tree**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0631713d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "# Inisialisasi model Decision Tree\n",
    "dtree = DecisionTreeClassifier(random_state=42)\n",
    "\n",
    "# Latih model\n",
    "dtree.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Prediksi pada data uji\n",
    "y_pred_dtree = dtree.predict(X_test_scaled)\n",
    "\n",
    "# Evaluasi model\n",
    "accuracy_dtree = accuracy_score(y_test, y_pred_dtree)\n",
    "report_dtree = classification_report(y_test, y_pred_dtree)\n",
    "\n",
    "print(f\"Akurasi Decision Tree: {accuracy_dtree:.2f}\")\n",
    "print(\"Classification Report Decision Tree:\")\n",
    "print(report_dtree)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "21e42f56",
   "metadata": {},
   "source": [
    "**3. Random Forest (dengan teknik pelatihan tambahan)**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1675a34e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from collections import Counter\n",
    "\n",
    "# Cek distribusi kelas sebelum SMOTE\n",
    "print(f\"Distribusi kelas sebelum SMOTE: {Counter(y_train)}\")\n",
    "\n",
    "# Terapkan SMOTE pada data latih\n",
    "smote = SMOTE(random_state=42)\n",
    "X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)\n",
    "\n",
    "# Cek distribusi kelas setelah SMOTE\n",
    "print(f\"Distribusi kelas setelah SMOTE: {Counter(y_train_smote)}\")\n",
    "\n",
    "# Inisialisasi model Random Forest\n",
    "rf_smote = RandomForestClassifier(random_state=42)\n",
    "\n",
    "# Latih model dengan data yang sudah di-SMOTE\n",
    "rf_smote.fit(X_train_smote, y_train_smote)\n",
    "\n",
    "# Prediksi pada data uji\n",
    "y_pred_rf_smote = rf_smote.predict(X_test_scaled)\n",
    "\n",
    "# Evaluasi model\n",
    "accuracy_rf_smote = accuracy_score(y_test, y_pred_rf_smote)\n",
    "report_rf_smote = classification_report(y_test, y_pred_rf_smote)\n",
    "\n",
    "print(f\"Akurasi Random Forest + SMOTE: {accuracy_rf_smote:.2f}\")\n",
    "print(\"Classification Report Random Forest + SMOTE:\")\n",
    "print(report_rf_smote)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff1b2383",
   "metadata": {},
   "source": [
    "**Visualisasi Akurasi Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7662c199",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_names = ['KNN', 'Decision Tree', 'Random Forest + SMOTE']\n",
    "accuracies = [accuracy_knn, accuracy_dtree, accuracy_rf_smote]\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=model_names, y=accuracies, palette='viridis')\n",
    "plt.ylim(0, 1)  # Batasi akurasi antara 0 dan 1\n",
    "plt.xlabel('Model Klasifikasi')\n",
    "plt.ylabel('Akurasi')\n",
    "plt.title('Perbandingan Akurasi Model Klasifikasi')\n",
    "plt.grid(axis='y', linestyle='--', alpha=0.7)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "165aadc3",
   "metadata": {},
   "source": [
    "Setelah dilakukan serangkaian percobaan dengan beberapa algoritma klasifikasi, diperoleh hasil performa dari tiga model berbeda, yaitu K-Nearest Neighbors (KNN), Decision Tree, dan Random Forest (dengan teknik pelatihan tambahan). Evaluasi dilakukan berdasarkan nilai akurasi prediksi terhadap data uji.  \n",
    "\n",
    "**Kesimpulan:** \n",
    "Random Forest + SMOTE memberi hasil terbaik (65%)  \n",
    "Berdasarkan hasil evaluasi, Random Forest adalah model terbaik yang digunakan dalam proses klasifikasi Chronic Kidney Disease pada dataset ini. Meskipun perbedaan akurasinya tidak terlalu besar dibandingkan KNN, konsistensi dan kestabilan model Random Forest memberikan keunggulan yang signifikan. Dengan hasil ini, Random Forest dapat dijadikan model dasar untuk pengembangan selanjutnya."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

python_code = []
for cell in notebook_content["cells"]:
    if cell["cell_type"] == "code":
        python_code.extend(cell["source"])
        python_code.append("\n") # Add a newline between code cells

# Join all lines to form the final .py content
py_content = "".join(python_code)

print(py_content)