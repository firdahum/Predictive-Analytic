# Laporan Proyek Machine Learning - Firda Humaira
## 1. Domain Proyek
Diabetes melitus (DM) adalah penyakit yang berkaitan dengan gangguan metabolisme, ditandai oleh peningkatan kadar glukosa dalam darah akibat berkurangnya produksi insulin oleh sel beta pankreas dan adanya gangguan atau resistensi terhadap insulin.

Diabetes melitus merupakan salah satu penyakit kronis yang berkontribusi signifikan terhadap angka kematian di Indonesia. Berdasarkan data dari Institute for Health Metrics and Evaluation (IHME), pada tahun 2019 diabetes tercatat sebagai penyebab kematian tertinggi ketiga di Indonesia, dengan tingkat kematian mencapai 57,42 per 100.000 penduduk. Selain itu, laporan dari International Diabetes Federation (IDF) menunjukkan bahwa jumlah penderita diabetes di Indonesia mengalami peningkatan yang signifikan dalam kurun waktu sepuluh tahun terakhir hingga tahun 2021.

Permasalahan ini menjadi sangat penting karena diabetes tidak hanya berdampak pada aspek kesehatan individu, tetapi juga menimbulkan beban ekonomi bagi negara akibat tingginya biaya pengobatan jangka panjang dan menurunnya produktivitas kerja penderita. Oleh karena itu, diperlukan upaya sistematis dalam pencegahan dan pengendalian diabetes, seperti peningkatan kesadaran masyarakat melalui edukasi gaya hidup sehat, deteksi dini melalui pemeriksaan rutin kadar gula darah, serta penguatan sistem layanan kesehatan dalam penanganan penyakit tidak menular.

## 2. Business Understanding
### 2.1 Problem Statements
1. Seberapa besar pengaruh kadar glukosa darah dan indeks massa tubuh (BMI) terhadap risiko seseorang menderita diabetes?
2. Apakah kombinasi kadar glukosa dan riwayat keluarga (Diabetes Pedigree Function) dapat digunakan untuk memprediksi diabetes secara akurat?
3. Bagaimana interaksi antara usia, kadar glukosa, dan BMI memengaruhi kemungkinan seseorang menderita diabetes?

### 2.2 Goals
1. Menentukan seberapa besar kontribusi kadar glukosa darah dan BMI dalam memprediksi risiko diabetes.
2. Mengetahui seberapa besar faktor genetik dan kadar gula darah berkontribusi terhadap diagnosis diabetes.
3. Menganalisis bagaimana ketiga faktor tersebut saling berinteraksi dalam mempengaruhi outcome diabetes.

### 2.3 Solution statements
1. Membangun model machine learning untuk mengukur kontribusi kedua faktor terhadap risiko diabetes, sehingga dapat diidentifikasi seberapa kuat pengaruh masing-masing variabel terhadap kondisi tersebut.
2. Membandingkan performa model dengan tuning optuna untuk menemukan model yang paling akurat dalam memprediksi risiko diabetes, kemudian menggunakan model terbaik tersebut untuk melakukan prediksi.
3. Melakukan analisis interaksi antara usia, kadar glukosa, dan BMI dalam membangun model prediktif, dengan tujuan memahami kontribusi gabungan ketiga variabel tersebut terhadap risiko diabetes dan memilih model terbaik berdasarkan metrik evaluasi yang terukur.

## 3. Data Understanding
Dataset yang saya gunakan diambil dari platform open source Kaggle dan dipublikasikan oleh paultimothymooney,_https://www.kaggle.com/code/paultimothymooney/predict-diabetes-from-medical-records/notebook_. Dataset ini berisi pengukuran yang berkaitan dengan Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.
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

### 3.1 Exploratory Data Analysis
#### 3.1.1 Distribusi Fitur pada Dataset Diabetes
<p align="center">
  <img src="assets/histogram_eda.png"width="1000"/>
</p>

Gambar di atas menunjukkan histogram dari setiap fitur dalam dataset, informasi yang dapat dilihat:
- Beberapa fitur seperti `Insulin` dan `SkinThickness` mengandung banyak nilai nol yang mencurigakan.
- `BMI` dan `Glucose` menunjukkan distribusi yang cukup normal, namun mayoritas pasien memiliki BMI tinggi.
- `Outcome` menunjukkan distribusi kelas yang tidak seimbang, dengan lebih banyak pasien non-diabetik.

#### 3.1.2 Pairplot antar Fitur pada Dataset Diabetes
<p align="center">
  <img src="assets/pairplot_eda.png"width="1000"/>
</p>

Gambar di atas menunjukan pairplot hubungan antara fitur-fitur dalam, informasi yang dapat dilihat:
- **Glucose** menunjukkan pemisahan yang cukup jelas antara pasien diabetes (`Outcome=1`) dan non-diabetes (`Outcome=0`). Kadar glukosa yang lebih tinggi cenderung dikaitkan dengan risiko diabetes.
- **BMI** (Body Mass Index) juga memiliki pola distribusi yang serupa, di mana pasien dengan nilai BMI lebih tinggi cenderung positif diabetes.
- **Age** memiliki korelasi moderat dengan `Outcome`, menunjukkan bahwa usia lebih tua cenderung berhubungan dengan peningkatan risiko.
- Korelasi antar fitur terlihat antara:
  - `Glucose` dan `Insulin`
  - `BMI` dan `SkinThickness`
  Korelasi ini ditunjukkan dengan pola menyudut pada scatterplot.
- Beberapa fitur seperti `BloodPressure` dan `DiabetesPedigreeFunction` tampak tidak memiliki hubungan yang kuat terhadap `Outcome`.

#### 3.1.3. Correlation Matrix pada antar Fitur pada Dataset Diabetes
<p align="center">
  <img src="assets/CM_eda.png"width="1000"/>
</p>

**insight:**
1. Korelasi terhadap Outcome

| Fitur                            | Korelasi dengan `Outcome` | Interpretasi                                                                   |
| -------------------------------- | ------------------------- | ------------------------------------------------------------------------------ |
| **Glucose**                      | **0.47**                  | Korelasi tertinggi → Semakin tinggi glukosa, semakin tinggi risiko diabetes |
| **BMI**                          | 0.29                      | Korelasi sedang → Obesitas berkontribusi terhadap risiko diabetes              |
| **Age**                          | 0.24                      | Semakin tua, risiko sedikit meningkat                                          |
| **Pregnancies**                  | 0.22                      | Wanita dengan lebih banyak kehamilan cenderung punya risiko lebih tinggi       |
| **DiabetesPedigreeFunction**     | 0.17                      | Ada pengaruh genetik, meskipun tidak terlalu kuat                              |
| **Insulin**                      | 0.13                      | Korelasi lemah, kemungkinan terpengaruh oleh banyak nilai nol                  |
| **BloodPressure, SkinThickness** | < 0.1                     | Korelasi sangat rendah → pengaruh terhadap diabetes lemah                      |


2. Korelasi antar Fitur

| Fitur 1           | Fitur 2           | Korelasi | Insight                                                                   |
| ----------------- | ----------------- | -------- | ------------------------------------------------------------------------- |
| **SkinThickness** | **Insulin**       | **0.44** | Cukup berkorelasi → menunjukkan hubungan biologis terkait metabolisme     |
| **BMI**           | **SkinThickness** | 0.39     | Individu dengan BMI lebih tinggi cenderung memiliki kulit lebih tebal     |
| **Glucose**       | **Insulin**       | 0.33     | Masuk akal, karena insulin berperan dalam regulasi glukosa                |
| **Pregnancies**   | **Age**           | **0.54** | Semakin tua, semakin banyak kemungkinan kehamilan (logis secara biologis) |

## 4. Data Preparation
### 4.1 Data Cleaning
#### 4.1.1 Missing Values dan Duplicated
<p align="center">
  <img src="assets/Missing_values.png"width="500"/>
</p>
**insight:**

Tidak terdapat missing values di dataset ini.

<p align="center">
  <img src="assets/duplicated.png"width="500"/>
</p>
**insight:**

Tidak terdapat duplikasi data di dataset ini.

#### 4.1.2 Outlier
<p align="center">
  <img src="assets/cek_outlier.png"width="500"/>
</p>

Akan menghasilkan output

<p align="center">
  <img src="assets/cek_outlier.png"width="500"/>
</p>

Terlihat ada variabel yang terdapat oulier, saya akan menghapus oulier tersebut.

<p align="center">
  <img src="assets/hapus_outlier.png"width="500"/>
</p>

<p align="center">
  <img src="assets/visual_after_hapus.png"width="500"/>
</p>

Oulier sudah tidak terdapat pada dataset ini.

### 4.2  Train Test Split
Train Test Split adalah teknik untuk membagi data menjadi data latih dan data uji. Data latih digunakan untuk melatih model, sementara data uji digunakan untuk mengukur performa model pada data baru.

Target pada dataset ini adalah variabel `Outcome` untuk mengetahui akurasi prediksi dari Outcome , maka akan menghapus kolom tersebut dari data dan assign kolom tersebut ke variabel baru. Selanjutnya, melakukan split data dengan skema data training sebesar 80% untuk melatih model dan 20% untuk menguji model.
<p align="center">
  <img src="assets/split.png"width="1000"/>
</p>
Tujuan Train-Test Split
- Memisahkan dataset df_filtered menjadi data latih (train) dan data uji (test).
- Komposisi: 80% untuk melatih model (train) dan 20% untuk menguji performa (test).
- random_state=30 digunakan agar pembagian data konsisten saat dijalankan ulang.

### 4.3  Normalisasi
**Tujuan Normalisasi**
1. Menyamakan skala fitur
  - Supaya fitur dengan skala besar (Insulin atau Glucose) tidak mendominasi fitur lain seperti BMI atau Age dalam proses training model.

2. Meningkatkan kinerja algoritma ML
  - Algoritma seperti K-Nearest Neighbors (KNN), SVM, dan Logistic Regression sensitif terhadap skala data. Tanpa normalisasi, performa model bisa menurun.

3. Mempercepat proses training
  - Karena perhitungan gradien atau jarak antar data menjadi lebih stabil dan efisien.

<p align="center">
  <img src="assets/normalisasi.png"width="1000"/>
</p>
**insight:**

- Tidak ada lagi fitur yang dominasinya terlalu besar (Insulin atau Age yang sebelumnya punya nilai jauh lebih tinggi dari fitur lain).
- Model yang dilatih menggunakan data ini akan lebih stabil, cepat dilatih, dan hasil evaluasi lebih valid.

## 5. Modeling
Hasil Akurasi 4 Model yang saya pakai:
### 5.1 Model Development dengan K-Nearest Neighbor
<p align="center">
  <img src="assets/knn.png"width="1000"/>
</p>

### 5.2 Model Development dengan Random Forest
<p align="center">
  <img src="assets/rf.png"width="1000"/>
</p>

### 5.3 Model Development dengan Logistic Regression
<p align="center">
  <img src="assets/lr.png"width="1000"/>
</p>

### 5.4 Model Development dengan Catboost
<p align="center">
  <img src="assets/cb.png"width="1000"/>
</p>

### Kelebihan dan Kekurangan Setiap Algoritma
| **Model**               | **Kelebihan**                                                                    | **Kekurangan**                                                      |
| ----------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **KNN**                 | - Mudah dipahami dan diimplementasikan                                           | - Prediksi lambat pada dataset besar                                |
|                         | - Non-parametrik, tidak membuat asumsi distribusi data                           | - Sensitif terhadap skala fitur (perlu normalisasi)                 |
|                         | - Adaptif terhadap data baru                                                     | - Kinerja menurun pada data berdimensi tinggi                       |
|                         |                                                                                  | - Rentan terhadap noise dan outlier                                 |
| **Random Forest**       | - Akurasi tinggi melalui ensemble learning                                       | - Waktu training dan prediksi lebih lama dibanding model sederhana  |
|                         | - Mengurangi risiko overfitting dari decision tree                               | - Interpretasi model lebih sulit                                    |
|                         | - Dapat menangani data kategorikal dan numerik                                   | - Konsumsi memori dan komputasi tinggi                              |
|                         | - Memberikan feature importance                                                  |                                                                     |
| **Logistic Regression** | - Cepat dan efisien pada dataset besar                                           | - Mengasumsikan hubungan linear antara fitur dan output             |
|                         | - Mudah diinterpretasi (koefisien fitur)                                         | - Tidak fleksibel untuk pola non-linear                             |
|                         | - Memberikan estimasi probabilitas                                               | - Rentan terhadap multikolinearitas                                 |
| **CatBoost**            | - Performa tinggi dan stabil di berbagai jenis data                              | - Lebih kompleks dan butuh tuning                                   |
|                         | - Tangani fitur kategorikal secara otomatis                                      | - Waktu pelatihan lebih lama dibanding logistic regression atau KNN |
|                         | - Mencegah overfitting dengan teknik regularisasi internal                       | - Interpretasi lebih sulit dibanding model linear                   |
|                         | - Efisien dan cepat dibanding gradient boosting lain (seperti XGBoost, LightGBM) |                                                                     |

### Pemilihan Model Terbaik
<p align="center">
  <img src="assets/rating.png"width="1000"/>
</p>
Model yang terbaik dari urutan akurasi dan f1 score adalah Algoritma K-Nearest Neighbor (KNN). Model KNN akan saya terapkan untuk menjawab masalah yang saya buat.

## 6. Evaluation
Dalam proyek ini, evaluasi model dilakukan dengan menggunakan confusion matrix, akurasi, dan f1 score sebagai metrik penilaian untuk setiap model. Sebelum itu, akan dijelaskan terlebih dahulu cara menghitung akurasi dan f1 score serta cara memanfaatkan confusion matrix.

**Mengukur ketepatan prediksi positif Precision, Recall, F1-Score**
1. Precision

            Precision = TP / (TP + FP)
   - Non-Diabetic: 84 / (84 + 14) = 0.86
   - Diabetic: 20 / (20 + 10) = 0.67
2.  Recall (Sensitivity)
            Recall = TP / (TP + FN)
   - Non-Diabetic: 84 / (84 + 10) = 0.89
   - Diabetic: 20 / (20 + 14) = 0.59
3.   F1-Score

            F1-score = 2 x ((Precision x Recall) / (Precision + Recall))

   - Non-Diabetic: 0.88
   - Diabetic: 0.62

**Mengukur Accuracy**

            Accuracy = (TP + TN) / Total = (84 + 20) / 128 = 0.81

- True Positive (TP): Jumlah data yang benar-benar positif dan diprediksi sebagai positif.
- False Positive (FP): Jumlah data yang sebenarnya negatif tetapi diprediksi sebagai positif (disebut juga Type I Error).
- True Negative (TN): Jumlah data yang benar-benar negatif dan diprediksi sebagai negatif.
- False Negative (FN): Jumlah data yang sebenarnya positif tetapi diprediksi sebagai negatif (Type II Error).

## 7. Menjawab Problem Statements
**1. Seberapa besar pengaruh kadar glukosa darah dan indeks massa tubuh (BMI) terhadap risiko seseorang menderita diabetes?**
<p align="center">
  <img src="assets/no1.png"width="1000"/>
</p>

**Interpretasi:**
1. Pengaruh Glukosa terhadap Prediksi Risiko Diabetes
  - Boxplot Glukosa menunjukkan perbedaan yang jelas antara pasien yang diprediksi menderita diabetes (Predicted Outcome = 1) dan yang tidak (Predicted Outcome = 0). Nilai median glukosa pada kelompok yang diprediksi diabetes berada pada kisaran yang jauh lebih tinggi dibandingkan kelompok non-diabetes. Selain itu, rentang interkuartil (IQR) pasien diabetes juga bergeser ke arah nilai glukosa yang lebih tinggi. Hal ini menunjukkan bahwa kadar glukosa darah sangat berpengaruh dalam menentukan prediksi risiko diabetes oleh model KNN.
 - **kesimpulan :** Perbedaan ini mengindikasikan bahwa semakin tinggi kadar glukosa seseorang, semakin besar kemungkinan ia diprediksi memiliki diabetes oleh model. Dengan kata lain, glukosa merupakan salah satu fitur paling dominan dalam pengambilan keputusan klasifikasi KNN pada dataset ini.


 2. Pengaruh BMI terhadap Prediksi Risiko Diabetes
  - Distribusi BMI berdasarkan prediksi KNN menunjukkan bahwa pasien yang diprediksi diabetes cenderung memiliki BMI yang sedikit lebih tinggi dibandingkan pasien yang diprediksi tidak diabetes. Meskipun demikian, terdapat tumpang tindih yang cukup besar antara kedua kelompok, terutama pada rentang IQR dan sebaran nilai BMI secara keseluruhan.
  - **kesimpulan :** meskipun BMI turut berkontribusi dalam prediksi risiko diabetes, pengaruhnya tidak sekuat glukosa. BMI lebih bersifat sebagai faktor pendukung, bukan penentu utama dalam keputusan model KNN. Dengan demikian, BMI tetap relevan, tetapi kadar glukosa lebih kuat dalam membedakan risiko diabetes.

**2. Bagaimana interaksi antara usia, kadar glukosa, dan BMI memengaruhi kemungkinan seseorang menderita diabetes?**
<p align="center">
  <img src="assets/no2.png"width="1000"/>
</p>

**intepretasi:**

Dari visualisasi 3D di atas, dapat dilihat bahwa individu dengan kadar glukosa tinggi (di atas 120 dalam skala standar) dan BMI yang tinggi (sekitar 35 ke atas) lebih banyak diprediksi menderita diabetes (ditunjukkan oleh titik-titik berwarna merah). Ini menunjukkan bahwa kadar glukosa dan BMI memiliki kontribusi signifikan terhadap peningkatan risiko diabetes.

Selain itu, usia juga tampak berperan, walaupun tidak sekuat glukosa dan BMI. Individu dengan usia lebih tua yang juga memiliki glukosa dan BMI tinggi cenderung lebih banyak terklasifikasi sebagai penderita diabetes. Namun, pada individu dengan usia muda, meskipun BMI tinggi, jika glukosa rendah, kemungkinan besar diprediksi tidak menderita diabetes.

Dengan akurasi model sebesar 81.125%, dapat disimpulkan bahwa model KNN mampu mengenali pola keterkaitan antara kombinasi usia, kadar glukosa, dan BMI terhadap risiko diabetes secara cukup baik. Visualisasi ini menguatkan bahwa kadar glukosa dan BMI adalah dua variabel yang sangat memengaruhi hasil prediksi diabetes, dengan usia berperan sebagai faktor pendukung.

**3. Apakah kombinasi kadar glukosa dan riwayat keluarga (Diabetes Pedigree Function) dapat digunakan untuk memprediksi diabetes secara akurat?**
<p align="center">
  <img src="assets/no3.png"width="1000"/>
</p>

**Interpretasi:**

**Kadar glukosa tinggi berkorelasi kuat dengan prediksi diabetes**
Titik berwarna merah (Predicted Outcome = 1) cenderung terkonsentrasi di sisi kanan grafik, yaitu pada kadar glukosa yang tinggi (sekitar di atas 130). Ini menunjukkan bahwa semakin tinggi kadar glukosa seseorang, semakin besar kemungkinan model memprediksi bahwa orang tersebut menderita diabetes.

**Riwayat keluarga (DPF) tidak terlalu dominan sebagai faktor prediksi tunggal**
Titik merah dan biru tersebar di berbagai nilai DPF, baik rendah maupun tinggi. Ini berarti bahwa DPF sendiri kurang memberikan perbedaan yang jelas dalam prediksi outcome, kecuali bila dikombinasikan dengan glukosa tinggi.

**Interaksi glukosa & DPF memberi gambaran yang lebih lengkap**
Meskipun DPF tidak terlalu dominan sendiri, ketika digabungkan dengan glukosa tinggi, prediksi diabetes menjadi lebih kuat. Contohnya: di area kanan bawah (glukosa tinggi, DPF rendah), tetap banyak prediksi diabetes.

### Kesimpulan

Kombinasi kadar glukosa dan riwayat keluarga (DPF) memang bisa digunakan untuk memprediksi diabetes, namun glukosa memiliki pengaruh yang jauh lebih besar dalam keputusan model KNN.

# **Referensi**

1. Rusdi, M. S. (2020). Hipoglikemia Pada Pasien Diabetes Melitus. Journal Syifa Sciences and Clinical Research (JSSCR), 2(2), 83-90.
2. https://ditpui.ugm.ac.id/diabetes-penyebab-kematian-tertinggi-di-indonesia-batasi-dengan-snack-sehat-rendah-gula/ (diakses pada tanggal 23 Mei 2025)
3. International Diabetes Federation (IDF). (2021). Indonesia - International Diabetes Federation. Diakses dari https://idf.org/our-network/regions-and-members/western-pacific/members/indonesia/

© 2025 Firda Humaira
