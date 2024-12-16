import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import streamlit as st

# Menampilkan Judul Aplikasi
st.title('Prediksi Bank Marketing')

# Upload file CSV melalui Streamlit
uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
if uploaded_file is not None:
    # Membaca dataset
    data = pd.read_csv(uploaded_file, sep=';')

    # Menampilkan Data Teratas dan Terbawah
    st.subheader("5 Data Teratas")
    st.write(data.head())

    st.subheader("5 Data Terbawah")
    st.write(data.tail())

    st.subheader("Ringkasan Statistik")
    st.write(data.describe())

    st.subheader("Informasi Tipe Data")
    st.write(data.dtypes)

    st.subheader("Jumlah Data yang Hilang")
    st.write(data.isnull().sum())

    st.subheader("Nama Kolom")
    st.write(data.columns.tolist())

    # Menambahkan dropdown untuk memilih target
    target_column = st.selectbox('Pilih kolom target', data.columns.tolist())

    # Menampilkan pilihan pengolahan target
    if target_column:
        st.subheader(f"Proses Pengolahan Target ({target_column})")

        # Cek apakah target adalah kategori biner (yes/no)
        if target_column in ['y']:
            target_map_option = st.radio("Pilih format pemetaan target", ('Biner (yes/no)', 'Nilai numerik lainnya'))

            if target_map_option == 'Biner (yes/no)':
                y = data[target_column].map({'yes': 1, 'no': 0})
                st.write(f"Target {target_column} dipetakan ke nilai numerik.")
            else:
                st.write(f"Target {target_column} digunakan seperti adanya.")
                y = data[target_column]
        else:
            y = data[target_column]
            st.write(f"Target {target_column} digunakan seperti adanya.")

        # Persiapan Data untuk Model
        X = data.drop(columns=target_column)

        # Meng-handle data kategorikal dengan get_dummies jika diperlukan
        if X.select_dtypes(include=['object']).shape[1] > 0:
            X = pd.get_dummies(X, drop_first=True)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Pembagian Data Train dan Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("Data Train (Features) - Sample")
        st.write(X_train[:5])

        st.subheader("Data Test (Features) - Sample")
        st.write(X_test[:5])

        st.subheader("Data Train (Target) - Sample")
        st.write(y_train[:5])

        st.subheader("Data Test (Target) - Sample")
        st.write(y_test[:5])

        # Membangun Model Neural Network
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer dengan 64 neuron
            Dense(32, activation='relu'),  # Hidden layer
            Dense(1, activation='sigmoid')  # Output layer dengan sigmoid untuk klasifikasi biner
        ])

        model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

        st.subheader("Ringkasan Model")
        st.text(model.summary())

        # Melatih Model dengan menampilkan progres secara dinamis
        st.subheader("Proses Pelatihan Model")

        # Membuat callback untuk progres pelatihan
        class TrainingProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                st.write(f'Epoch {epoch+1}: Akurasi = {logs["accuracy"]}, Loss = {logs["loss"]}')
        
        # Melatih Model
        history = model.fit(
            X_train, y_train, 
            epochs=100, 
            validation_split=0.2, 
            callbacks=[TrainingProgressCallback()]
        )

        # Menampilkan Grafik Akurasi
        st.subheader("Grafik Akurasi Model")
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        st.pyplot()

        # Menampilkan Grafik Loss
        st.subheader("Grafik Loss Model")
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        st.pyplot()

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

        # Tampilkan classification report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
