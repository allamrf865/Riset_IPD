# Import library yang diperlukan
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# Fungsi untuk menghitung metrik evaluasi
def hitung_metrik(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # Asumsi 1 = positif, 0 = negatif
    true_positives = cm[0, 0]
    false_positives = cm[1, 0]
    false_negatives = cm[0, 1]
    true_negatives = cm[1, 1]

    # 1. Sensitivitas (Recall)
    sensitivitas = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else float('nan')

    # 2. Spesifisitas
    spesifisitas = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else float('nan')

    # 3. Nilai Duga Positif (PPV)
    ppv = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else float('nan')

    # 4. Nilai Duga Negatif (NPV)
    npv = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else float('nan')

    # 5. Prevalensi
    prevalensi = (true_positives + false_negatives) / len(y_true) if len(y_true) > 0 else float('nan')

    # 6. Rasio Kemungkinan Positif (PLR)
    plr = sensitivitas / (1 - spesifisitas) if spesifisitas < 1 else float('nan')

    # 7. Rasio Kemungkinan Negatif (NLR)
    nlr = (1 - sensitivitas) / spesifisitas if spesifisitas > 0 else float('nan')

    # 8. Akurasi (Accuracy)
    akurasi = accuracy_score(y_true, y_pred)

    return sensitivitas, spesifisitas, ppv, npv, prevalensi, plr, nlr, akurasi

# Fungsi untuk visualisasi ROC dan menghitung AUC untuk satu dataset
def plot_roc_curve(y_true, y_pred_prob, pos_label, label):
    # Menghitung ROC curve dengan label positif yang ditentukan user
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob, pos_label=pos_label)
    auc_value = roc_auc_score(y_true, y_pred_prob)

    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_value:.2f})')
    return auc_value

# Fungsi untuk menampilkan interpretasi
def interpretasi(sensitivitas, spesifisitas, ppv, npv, prevalensi, plr, nlr, akurasi, auc_value, label):
    st.subheader(f"Interpretasi Hasil untuk {label}")
    
    # Gunakan try-except atau cek untuk menggantikan NaN dengan informasi yang lebih jelas
    st.write(f"Sensitivitas: {sensitivitas:.2f} - Model mendeteksi {sensitivitas*100:.1f}% dari semua kasus positif.") if not pd.isna(sensitivitas) else st.write("Sensitivitas: Tidak terdefinisi")
    st.write(f"Spesifisitas: {spesifisitas:.2f} - Model mendeteksi {spesifisitas*100:.1f}% dari semua kasus negatif.") if not pd.isna(spesifisitas) else st.write("Spesifisitas: Tidak terdefinisi")
    st.write(f"Nilai Duga Positif (PPV): {ppv:.2f} - {ppv*100:.1f}% dari prediksi positif adalah benar.") if not pd.isna(ppv) else st.write("Nilai Duga Positif: Tidak terdefinisi")
    st.write(f"Nilai Duga Negatif (NPV): {npv:.2f} - {npv*100:.1f}% dari prediksi negatif adalah benar.") if not pd.isna(npv) else st.write("Nilai Duga Negatif: Tidak terdefinisi")
    st.write(f"Prevalensi: {prevalensi:.2f} - Prevalensi kasus positif dalam dataset.") if not pd.isna(prevalensi) else st.write("Prevalensi: Tidak terdefinisi")
    st.write(f"Rasio Kemungkinan Positif (PLR): {plr:.2f} - Likelihood hasil positif benar-benar positif.") if not pd.isna(plr) else st.write("Rasio Kemungkinan Positif: Tidak terdefinisi")
    st.write(f"Rasio Kemungkinan Negatif (NLR): {nlr:.2f} - Likelihood hasil negatif benar-benar negatif.") if not pd.isna(nlr) else st.write("Rasio Kemungkinan Negatif: Tidak terdefinisi")
    st.write(f"Akurasi: {akurasi:.2f} - Model memiliki akurasi sebesar {akurasi*100:.1f}%.")
    if auc_value:
        st.write(f"AUC: {auc_value:.2f} - Area under the curve untuk {label}.")

# Fungsi utama untuk menjalankan pipeline analisis untuk satu dataset
def run_evaluation_pipeline(df, target_col, pos_label, label):
    # **Pemetaan nilai 1 dan 2 menjadi 0 dan 1 di seluruh kolom berdasarkan pilihan kelas positif**
    df_mapped = df.applymap(lambda x: 1 if x == pos_label else (0 if x != pos_label else x))

    # Pisahkan target dan fitur prediktor
    X = df_mapped.drop(columns=[target_col])
    y = df_mapped[target_col]

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

    # 9: Plot ROC curve dan hitung AUC, menggunakan pos_label yang dipilih oleh user
    auc_value = plot_roc_curve(y_test, y_pred_prob, 1, label)  # Positif selalu dianggap sebagai 1 setelah pemetaan

    # Menampilkan interpretasi hasil
    interpretasi(sensitivitas, spesifisitas, ppv, npv, prevalensi, plr, nlr, akurasi, auc_value, label)

    return auc_value

# 6. Antarmuka Streamlit
st.title("AI by Allam Rafi FKUI 2022_Research Scientist")
st.write("Unggah hingga 10 dataset Anda untuk membandingkan ROC dan AUC.")

# Menggunakan layout kiri dan kanan agar tampil lebih rapi
cols = st.columns(2)

# Mengunggah hingga 10 file dataset di dua kolom
uploaded_files = []
for i in range(10):
    col = cols[i % 2]  # Bergantian antara kolom kiri dan kanan
    with col:
        uploaded_file = st.file_uploader(f"Unggah Dataset {i + 1} (Excel)", type=["xlsx"], key=f"file_{i}")
        if uploaded_file:
            uploaded_files.append(uploaded_file)

# Menjalankan analisis jika ada file yang diunggah
if len(uploaded_files) > 0:
    plt.figure(figsize=(10, 6))

    for idx, file in enumerate(uploaded_files):
        df = pd.read_excel(file)
        st.write(f"Dataset {idx + 1}:")
        st.write(df.head())

        # Pilih kolom target untuk setiap dataset
        target_col = st.selectbox(f"Pilih kolom target untuk Dataset {idx + 1}", options=df.columns, key=f"target_{idx + 1}")
        
        # Pilih kelas yang menjadi kelas positif (misalnya 1 atau 2)
        pos_label = st.selectbox(f"Pilih kelas positif untuk Dataset {idx + 1}", options=df[target_col].unique(), key=f"pos_label_{idx + 1}")

        # Jalankan pipeline evaluasi
        auc_value = run_evaluation_pipeline(df, target_col, pos_label, f"Dataset {idx + 1}")

    # Menampilkan kurva ROC untuk semua dataset
    plt.plot([0, 1], [0, 1], 'r--')  # Garis acak
    plt.xlabel('False Positive Rate (1 - Spesifisitas)')
    plt.ylabel('True Positive Rate (Sensitivitas)')
    plt.title('Perbandingan ROC Curve untuk Semua Dataset')
    plt.legend(loc='lower right')
    plt.grid()
    st.pyplot(plt)
