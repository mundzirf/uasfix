import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np


# Fungsi utama untuk multipages
def main():
    st.sidebar.title("Navigasi")
    pages = ["Home", "EDA", "Train Model", "Predict"]
    choice = st.sidebar.radio("Menu", pages)

    if choice == "Home":
        home_page()
    elif choice == "EDA":
        eda_page()
    elif choice == "Train Model":
        train_model_page()
    elif choice == "Predict":
        predict_page()


# Halaman Home
def home_page():
    st.title("Titanic Survival Prediction")
    st.write(
        """
    Selamat datang di aplikasi prediksi survival penumpang Titanic! 
    Gunakan menu di sidebar untuk mengakses fitur:
    - **EDA:** Analisis eksplorasi data
    - **Train Model:** Melatih model machine learning
    - **Predict:** Melakukan prediksi berdasarkan data baru
    
    dibuat oleh : 
    MUNDZIR FAUZAN 
       22191216
        STIN
    """
    )


# Fungsi Preprocessing
def preprocess_data(data, train_mode=True):
    """
    Fungsi untuk memproses data Titanic:
    - Mengisi nilai yang hilang
    - Melakukan encoding pada kolom kategorikal
    - Menghapus kolom yang tidak relevan
    """
    # Mengisi nilai kosong pada kolom numerik dan kategorikal
    st.write("Preprocessing data...")
    data.fillna({
        column: data[column].median() if data[column].dtype != 'object' else data[column].mode()[0]
        for column in data.columns
}, inplace=True)


    # Encoding kolom kategorikal
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)

    # Label encoding untuk age_group dan fare_group jika ada
    if "age_group" in data.columns:
        le = LabelEncoder()
        data["age_group"] = le.fit_transform(data["age_group"])
    if "fare_group" in data.columns:
        le = LabelEncoder()
        data["fare_group"] = le.fit_transform(data["fare_group"])

    # Hapus kolom yang tidak relevan
    data = data.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, errors="ignore")

    # Jika dalam mode pelatihan, pastikan target "Survived" ada
    if train_mode and "Survived" not in data.columns:
        raise ValueError("Kolom 'Survived' hilang dalam dataset!")

        # Jika dalam mode prediksi, pastikan "Survived" tidak ada
    if not train_mode and "Survived" in data.columns:
        data = data.drop("Survived", axis=1)

    return data


# Halaman EDA
def eda_page():
    st.title("Exploratory Data Analysis (EDA)")
    uploaded_file = st.file_uploader("Upload dataset Titanic (CSV):", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset:")
            st.dataframe(data)

            # Statistik deskriptif
            st.write("Statistik Deskriptif:")
            st.write(data.describe())

            # Visualisasi distribusi Survival
            st.write("Distribusi Survival:")
            fig, ax = plt.subplots()
            sns.countplot(data=data, x="Survived", ax=ax, palette="viridis")
            st.pyplot(fig)

            # Visualisasi hubungan fitur numerik
            if "Age" in data.columns and "Fare" in data.columns:
                st.write("Distribusi Age vs Fare:")
                fig, ax = plt.subplots()
                sns.scatterplot(data=data, x="Age", y="Fare", hue="Survived", palette="viridis", ax=ax)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error dalam membaca dataset: {e}")


# Halaman Train Model
def train_model_page():
    st.title("Train Model")
    uploaded_file = st.file_uploader("Upload dataset Titanic (CSV):", type="csv")
    if uploaded_file:
        try:
            # Membaca dataset
            data = pd.read_csv(uploaded_file)
            st.write("Dataset awal:")
            st.dataframe(data.head())

            # Preprocessing data
            st.write("Preprocessing Data...")
            data = preprocess_data(data, train_mode=True)
            if data.isnull().sum().sum() > 0:
                st.error("Error: Masih ada nilai kosong (NaN) dalam dataset setelah preprocessing.")
                st.write("Periksa kembali dataset Anda:")
                st.dataframe(data.isnull().sum())
                return

            # Pisahkan fitur (X) dan target (y)
            X = data.drop("Survived", axis=1, errors="ignore")
            y = data["Survived"]

            # Split data menjadi data latih dan uji
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if "Survived" in X_train.columns:
                X_train = X_train.drop("Survived", axis=1)
            if "Survived" in X_test.columns:
                 X_test = X_test.drop("Survived", axis=1)

            # Melatih model Random Forest
            st.write("Training Random Forest...")
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)
            st.write(f"Random Forest Accuracy: {rf_acc:.2f}")

            # Melatih model Logistic Regression menggunakan pipeline
            st.write("Training Logistic Regression dengan pipeline...")
            lr_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),  # Isi NaN dengan median
                ("scaler", StandardScaler()),  # Standardisasi fitur
                ("model", LogisticRegression(max_iter=200))
            ])

            # Fit model Logistic Regression
            lr_pipeline.fit(X_train, y_train)
            lr_pred = lr_pipeline.predict(X_test)
            lr_acc = accuracy_score(y_test, lr_pred)

            # Output hasil Logistic Regression
            st.write(f"Logistic Regression Accuracy (dengan pipeline): {lr_acc:.2f}")
            # Hasil benchmarking
            st.write("Hasil Benchmarking:")
            st.write(f"Random Forest Accuracy: {rf_acc:.2f}")
            st.write(f"Logistic Regression Accuracy: {lr_acc:.2f}")

            # Simpan model ke file
            joblib.dump(rf_model, "titanic_model.pkl")
            st.success("Model Random Forest telah disimpan sebagai `titanic_model.pkl`.")

        except Exception as e:
            st.error(f"Error saat memproses data: {e}")


# Halaman Predict
def predict_page():
    st.title("Predict")
    uploaded_file = st.file_uploader("Upload dataset untuk prediksi (CSV):", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset untuk prediksi:")
            st.dataframe(data.head())

            # Load model
            try:
                model = joblib.load("titanic_model.pkl")
            except FileNotFoundError:
                st.error("Model belum dilatih! Silakan latih model terlebih dahulu di menu Train Model.")
                return

            # Preprocessing data untuk prediksi
            data = preprocess_data(data, train_mode=False)

            # Prediksi
            predictions = model.predict(data)
            data["Prediction"] = predictions
            st.write("Hasil Prediksi:")
            st.dataframe(data[["Prediction"]])

        except Exception as e:
            st.error(f"Error saat memproses data: {e}")


if __name__ == "__main__":
    main()
