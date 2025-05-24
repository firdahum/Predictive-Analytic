# Laporan Proyek Machine Learning - Firda Humaira
## Domain Proyek
Diabetes melitus (DM) adalah penyakit yang berkaitan dengan gangguan metabolisme, ditandai oleh peningkatan kadar glukosa dalam darah akibat berkurangnya produksi insulin oleh sel beta pankreas dan adanya gangguan atau resistensi terhadap insulin.

Diabetes melitus merupakan salah satu penyakit kronis yang berkontribusi signifikan terhadap angka kematian di Indonesia. Berdasarkan data dari Institute for Health Metrics and Evaluation (IHME), pada tahun 2019 diabetes tercatat sebagai penyebab kematian tertinggi ketiga di Indonesia, dengan tingkat kematian mencapai 57,42 per 100.000 penduduk. Selain itu, laporan dari International Diabetes Federation (IDF) menunjukkan bahwa jumlah penderita diabetes di Indonesia mengalami peningkatan yang signifikan dalam kurun waktu sepuluh tahun terakhir hingga tahun 2021.

Permasalahan ini menjadi sangat penting karena diabetes tidak hanya berdampak pada aspek kesehatan individu, tetapi juga menimbulkan beban ekonomi bagi negara akibat tingginya biaya pengobatan jangka panjang dan menurunnya produktivitas kerja penderita. Oleh karena itu, diperlukan upaya sistematis dalam pencegahan dan pengendalian diabetes, seperti peningkatan kesadaran masyarakat melalui edukasi gaya hidup sehat, deteksi dini melalui pemeriksaan rutin kadar gula darah, serta penguatan sistem layanan kesehatan dalam penanganan penyakit tidak menular.

## Business Understanding
### Problem Statements
1. Seberapa besar pengaruh kadar glukosa darah dan indeks massa tubuh (BMI) terhadap risiko seseorang menderita diabetes?
2. Apakah kombinasi kadar glukosa dan riwayat keluarga (Diabetes Pedigree Function) dapat digunakan untuk memprediksi diabetes secara akurat?
3. Bagaimana interaksi antara usia, kadar glukosa, dan BMI memengaruhi kemungkinan seseorang menderita diabetes?

### Goals
1. Menentukan seberapa besar kontribusi kadar glukosa darah dan BMI dalam memprediksi risiko diabetes.
2. Mengetahui seberapa besar faktor genetik dan kadar gula darah berkontribusi terhadap diagnosis diabetes.
3. Menganalisis bagaimana ketiga faktor tersebut saling berinteraksi dalam mempengaruhi outcome diabetes.

### Solution statements
1. Membangun model machine learning untuk mengukur kontribusi kedua faktor terhadap risiko diabetes, sehingga dapat diidentifikasi seberapa kuat pengaruh masing-masing variabel terhadap kondisi tersebut.
2. Membandingkan performa model dengan tuning optuna untuk menemukan model yang paling akurat dalam memprediksi risiko diabetes, kemudian menggunakan model terbaik tersebut untuk melakukan prediksi.
3. Melakukan analisis interaksi antara usia, kadar glukosa, dan BMI dalam membangun model prediktif, dengan tujuan memahami kontribusi gabungan ketiga variabel tersebut terhadap risiko diabetes dan memilih model terbaik berdasarkan metrik evaluasi yang terukur.

## Data Understanding
Dataset yang saya gunakan diambil dari platform open source Kaggle dan dipublikasikan oleh paultimothymooney, _https://www.kaggle.com/code/paultimothymooney/predict-diabetes-from-medical-records/notebook_. Dataset ini berisi pengukuran yang berkaitan dengan Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.
| **Nama Variabel**            | **Deskripsi**                                                            |
| ---------------------------- | ------------------------------------------------------------------------ |
| **Pregnancies**              | Jumlah kehamilan yang pernah dialami pasien                              |
| **Glucose**                  | Konsentrasi glukosa plasma (mg/dL) dalam tes toleransi glukosa           |
| **BloodPressure**            | Tekanan darah diastolik (mm Hg)                                          |
| **SkinThickness**            | Ketebalan lipatan kulit triceps (mm)                                     |
| **Insulin**                  | Konsentrasi insulin serum dua jam (mu U/ml)                              |
| **BMI**                      | Indeks Massa Tubuh (kg/m²), dihitung dari berat badan dan tinggi badan   |
| **DiabetesPedigreeFunction** | Skor riwayat genetik diabetes (kemungkinan risiko berdasarkan keturunan) |
| **Age**                      | Usia pasien (dalam tahun)                                                |
| **Outcome**                  | Hasil diagnosis (1 = menderita diabetes, 0 = tidak menderita diabetes)   |

Terkait project selanjutnya ada di file .ipynb dan .py

© 2025 Firda Humaira
