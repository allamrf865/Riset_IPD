# 1. Import library yang diperlukan
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# 2. Fungsi untuk menghitung metrik evaluasi
def hitung_metrik(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # Asumsi 1 = positif, 0 = negatif
    true_positives = cm[0, 0]
    false_positives = cm[1, 0]
    false_negatives = cm[0, 1]
    true_negatives = cm[1, 1]

    # 1. Sensitivitas (Recall)
    sensitivitas = true_positives / (true_positives + false_negatives)

    # 2. Spesifisitas
    spesifisitas = true_negatives / (true_negatives + false_positives)

    # 3. Nilai Duga Positif (PPV)
    ppv = true_positives / (true_positives + false_positives)

    # 4. Nilai Duga Negatif (NPV)
    npv = true_negatives / (true_negatives + false_negatives)

    # 5. Prevalensi
    prevalensi = (true_positives + false_negatives) / len(y_true)

    # 6. Rasio Kemungkinan Positif (PLR)
    plr = sensitivitas / (1 - spesifisitas)

    # 7. Rasio Kemungkinan Negatif (NLR)
    nlr = (1 - sensitivitas) / spesifisitas

    # 8. Akurasi (Accuracy)
    akurasi = accuracy_score(y_true, y_pred)

    return sensitivitas, spesifisitas, ppv, npv, prevalensi, plr, nlr, akurasi

# 3. Fungsi untuk visualisasi ROC dan menghitung AUC
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_value = roc_auc_score(y_true, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')  # Garis acak
    plt.xlabel('False Positive Rate (1 - Spesifisitas)')
    plt.ylabel('True Positive Rate (Sensitivitas)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    st.pyplot(plt)

    return auc_value

# 4. Fungsi untuk menampilkan interpretasi
def interpretasi(sensitivitas, spesifisitas, ppv, npv, prevalensi, plr, nlr, akurasi, auc_value):
    st.subheader("Interpretasi Hasil")
    st.write(f"Sensitivitas: {sensitivitas:.2f} - Model mendeteksi {sensitivitas*100:.1f}% dari semua kasus positif.")
    st.write(f"Spesifisitas: {spesifisitas:.2f} - Model mendeteksi {spesifisitas*100:.1f}% dari semua kasus negatif.")
    st.write(f"Nilai Duga Positif (PPV): {ppv:.2f} - {ppv*100:.1f}% dari prediksi positif adalah benar.")
    st.write(f"Nilai Duga Negatif (NPV): {npv:.2f} - {npv*100:.1f}% dari prediksi negatif adalah benar.")
    st.write(f"Prevalensi: {prevalensi:.2f} - Prevalensi kasus positif dalam dataset.")
    st.write(f"Rasio Kemungkinan Positif (PLR): {plr:.2f} - Likelihood hasil positif benar-benar positif.")
    st.write(f"Rasio Kemungkinan Negatif (NLR): {nlr:.2f} - Likelihood hasil negatif benar-benar negatif.")
    st.write(f"Akurasi: {akurasi:.2f} - Model memiliki akurasi sebesar {akurasi*100:.1f}%.")
    st.write(f"AUC: {auc_value:.2f} - Area under the curve, menunjukkan kemampuan model membedakan antara kelas positif dan negatif.")

# 5. Fungsi utama untuk menjalankan pipeline analisis
def run_evaluation_pipeline(df, target_col):
    # Pastikan target kolom dipilih dengan benar
    if target_col not in df.columns:
        st.error(f"Kolom target '{target_col}' tidak ditemukan dalam dataset.")
        return

    # Pisahkan target dan fitur prediktor
    X = df.drop(columns=[target_col])  # Semua kolom selain target sebagai fitur
    y = df[target_col]  # Kolom target

    # Membagi dataset menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Melatih model RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # 1-8: Menghitung metrik evaluasi
    sensitivitas, spesifisitas, ppv, npv, prevalensi, plr, nlr, akurasi = hitung_metrik(y_test, y_pred)

    # 9: Plot ROC curve dan hitung AUC
    auc_value = plot_roc_curve(y_test, y_pred_prob)

    # Menampilkan interpretasi hasil
    interpretasi(sensitivitas, spesifisitas, ppv, npv, prevalensi, plr, nlr, akurasi, auc_value)

# 6. Antarmuka Streamlit
st.title("AI by Allam Rafi FKUI 2022_Research Scientist")
st.write("Unggah dataset Anda dan dapatkan analisis metrik evaluasi AI.")

uploaded_file = st.file_uploader("Unggah Dataset (Excel)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Dataset yang diunggah:")
    st.write(df.head())

    # Menampilkan pilihan untuk memilih kolom target
    target_col = st.selectbox("Pilih kolom target (label)", options=df.columns)

    # Menjalankan evaluasi jika pengguna memilih target
    if st.button("Jalankan Analisis"):
        run_evaluation_pipeline(df, target_col)
