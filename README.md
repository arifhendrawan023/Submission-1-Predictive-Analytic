# Laporan Proyek Machine Learning - Arif Hendrawan Priliyanto

## Domain Proyek

Penyakit kardiovaskular (CVD) merupakan salah satu penyebab kematian utama di seluruh dunia, memakan korban sekitar 17,9 juta nyawa setiap tahunnya, yang menyumbang sekitar 31% dari total kematian di seluruh dunia. Empat dari 5 kematian akibat CVD disebabkan oleh serangan jantung dan stroke, dan sepertiga dari kematian ini terjadi sebelum usia 70 tahun. Gagal jantung merupakan kejadian umum yang disebabkan oleh CVD, dan dataset ini berisi 11 fitur yang dapat digunakan untuk memprediksi kemungkinan penyakit jantung.

Orang-orang dengan penyakit kardiovaskular atau yang berisiko tinggi terkena penyakit kardiovaskular (karena adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia, atau penyakit yang sudah ada) memerlukan deteksi dini dan pengelolaan yang tepat. Di sinilah pendekatan machine learning atau deep learning dapat memberikan kontribusi besar.

Pendekatan machine learning memiliki beberapa kontribusi spesifik dalam penanganan penyakit kardiovaskular yang sulit dicapai dengan metode konvensional:

- Pemrosesan Data Kompleks: Metode machine learning mampu mengekstraksi pola dari dataset yang kompleks dan berdimensi tinggi, seperti data medis yang mengandung berbagai fitur klinis. Ini membantu mengidentifikasi hubungan yang rumit antara fitur-fitur tersebut dengan risiko penyakit kardiovaskular.

- Penggabungan Informasi: Ensemble methods seperti Random Forest dan Boosting Algorithm menggabungkan hasil dari beberapa model untuk membuat prediksi yang lebih baik. Ini memungkinkan untuk mengatasi variabilitas dan bias yang mungkin ada dalam satu model.

- Deteksi Dini: Machine learning dapat mengidentifikasi risiko penyakit kardiovaskular pada tahap awal berdasarkan fitur-fitur tertentu. Ini membantu dalam pencegahan dan intervensi dini, yang mungkin tidak dapat dilakukan dengan metode konvensional.

- Adaptabilitas: Model machine learning dapat beradaptasi dengan perubahan dalam data atau kondisi pasien. Mereka dapat diperbarui dengan data baru dan dapat meningkatkan akurasi seiring waktu.

- Pengurangan Biaya dan Waktu: Dengan menerapkan model machine learning untuk melakukan prediksi, proses diagnosis dan deteksi risiko dapat menjadi lebih cepat dan efisien, yang pada akhirnya dapat mengurangi biaya perawatan jangka panjang.

## Business Understanding

Problem Statement 1:
Bagaimana cara pendekatan yang lebih efektif untuk deteksi dini dan manajemen penyakit kardiovaskular untuk mengurangi angka kematian dan dampaknya terhadap kesehatan masyarakat ?

Goal 1:
Dengan cara mengembangkan model prediksi menggunakan teknik machine learning (K-Nearest Neighbor, Random Forest, dan Boosting Algorithm) untuk mengidentifikasi individu dengan risiko penyakit kardiovaskular yang lebih tinggi.

Problem Statement 2:
Bagaimana meningkatkan efisiensi dan akurasi pengelolaan penyakit kardiovaskular dengan pendekatan yang memanfaatkan teknologi machine learning?

Goal 2:
Dengan cara memanfaatkan teknik optimasi hyperparameter, terutama melalui metode Random Search, untuk menemukan kombinasi parameter yang optimal dalam model prediksi penyakit kardiovaskular. Dengan melakukan pencarian parameter secara acak dan terstruktur, tujuan utama adalah meningkatkan akurasi dan performa keseluruhan model dengan mengidentifikasi parameter terbaik yang sesuai untuk algoritma machine learning yang digunakan.

## Solution statements
Solusi untuk permasalahan deteksi dini penyakit kardiovaskular menggunakan metode K-Nearest Neighbor, Random Forest, dan Boosting Algorithm melalui pendekatan machine learning adalah sebagai berikut:

1. K-Nearest Neighbor (K-NN):
K-NN adalah algoritma yang dapat digunakan untuk klasifikasi. Ide dasarnya adalah mencari "k" titik terdekat dari data yang baru (titik uji) dalam ruang fitur dan kemudian mengambil mayoritas kelas dari tetangga terdekat sebagai prediksi. Dalam konteks deteksi penyakit kardiovaskular, Anda dapat menggunakan K-NN untuk mengklasifikasikan apakah seseorang memiliki risiko penyakit kardiovaskular berdasarkan fitur-fitur tertentu yang ada dalam dataset.

2. Random Forest:
Random Forest adalah metode ensemble yang menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dalam Random Forest dilatih dengan subset data dan subset fitur yang acak. Dalam kasus ini, Random Forest dapat digunakan untuk menggabungkan banyak pohon keputusan yang memutuskan apakah seseorang memiliki risiko penyakit kardiovaskular. Ini akan menghasilkan prediksi yang lebih stabil dan akurat.

3. Boosting Algorithm:
Boosting adalah metode ensemble lain yang bekerja dengan melatih sejumlah model yang lemah secara berurutan. Setiap model berfokus pada mengoreksi kesalahan model sebelumnya. Gradient Boosting adalah salah satu algoritma boosting yang populer. Dalam konteks deteksi penyakit kardiovaskular, Boosting Algorithm dapat digunakan untuk mengoptimalkan prediksi dengan mempertimbangkan kesalahan prediksi sebelumnya.

4. Random Search ( Hyperparameter tuning )
Untuk mengatasi tantangan optimasi parameter, solusi yang dapat diambil adalah penerapan metode Random Search. Dengan menerapkan solusi ini, model yang dihasilkan akan memiliki kinerja yang lebih baik karena parameter optimal telah ditemukan melalui Random Search. Dengan demikian, akurasi prediksi penyakit kardiovaskular dapat ditingkatkan secara signifikan, dan model akan lebih siap digunakan dalam praktik klinis atau lingkungan kesehatan.

## Data Understanding
Dataset ini dibuat dengan menggabungkan berbagai dataset yang sudah tersedia secara mandiri namun belum digabungkan sebelumnya. Dalam kumpulan data ini, 5 kumpulan data jantung digabungkan dalam 11 fitur umum yang menjadikannya kumpulan data penyakit jantung terbesar yang tersedia sejauh ini untuk tujuan penelitian. Lima kumpulan data yang digunakan untuk kurasinya adalah:

Cleveland: 303 observasi
Hongaria: 294 observasi
Swiss: 123 observasi
Long Beach VA: 200 observasi
Kumpulan Data Stalog (Hati): 270 observasi

Total: 1190 observasi
Duplicate: 272 observasi

Kaggle (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Age: Umur pasien dalam tahun. Informasi ini memberikan gambaran tentang seberapa tua atau muda pasien.

- Sex: Jenis kelamin pasien. Nilai "M" mewakili laki-laki (Male) dan "F" mewakili perempuan (Female).

- ChestPainType: Jenis nyeri dada yang dirasakan oleh pasien. Nilai "TA" mengindikasikan Typical Angina (nyeri dada khas), "ATA" mengindikasikan Atypical Angina (nyeri dada tidak khas), "NAP" mengindikasikan Non-Anginal Pain (nyeri dada non-angina), dan "ASY" mengindikasikan Asymptomatic (tanpa gejala nyeri dada).

- RestingBP: Tekanan darah istirahat pasien dalam satuan mm Hg (milimeter raksa). Ini mengukur tekanan darah pada saat pasien dalam keadaan istirahat.

- Cholesterol: Kolesterol serum pasien dalam satuan mm/dl (milimeter per deciliter). Informasi ini memberikan gambaran tentang tingkat kolesterol dalam darah pasien.

- FastingBS: Kadar gula darah puasa pasien. Nilai "1" menunjukkan bahwa kadar gula darah puasa lebih dari 120 mg/dl, sementara nilai "0" menunjukkan sebaliknya.

- RestingECG: Hasil elektrokardiogram istirahat pasien. Nilai "Normal" menunjukkan hasil normal, "ST" menunjukkan adanya abnormalitas gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST > 0.05 mV), dan "LVH" menunjukkan kemungkinan atau pasti adanya hipertrofi ventrikel kiri berdasarkan kriteria Estes.

- MaxHR: Detak jantung maksimum yang dicapai oleh pasien, diukur dalam nilai numerik antara 60 hingga 202 denyut per menit. Ini memberikan gambaran tentang kapasitas jantung pasien.

- ExerciseAngina: Apakah pasien mengalami angina yang dipicu oleh latihan fisik. Nilai "Y" menunjukkan bahwa pasien mengalami angina saat berolahraga (Yes), sementara "N" menunjukkan sebaliknya (No).

- Oldpeak: Depresi segmen ST atau oldpeak yang diukur dalam nilai numerik. Ini memberikan informasi tentang perubahan relatif dalam segmen ST selama aktivitas fisik dibandingkan dengan istirahat.

- ST_Slope: Kemiringan segmen ST puncak latihan. Nilai "Up" menunjukkan kemiringan naik (upsloping), "Flat" menunjukkan kemiringan datar (flat), dan "Down" menunjukkan kemiringan turun (downsloping).

- HeartDisease: Kelas output yang menunjukkan apakah pasien memiliki penyakit jantung atau tidak. Nilai "1" mengindikasikan keberadaan penyakit jantung, sedangkan "0" mengindikasikan ketiadaannya (normal).

**Proses eksplorasi dataset
- Mengecek typedata
- Mengecek deskripsi statistik
  
  Univariate Analysis
- Melakukan proses analisis data dengan teknik Univariate EDA dengan membagi fitur pada dataset menjadi dua bagian, yaitu numerical features dan categorical features
- Melakukan analisis terhadap fitur kategori
- Fitur Sex
  
  ![download (5)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/78c0200c-9b1c-49c5-9795-db9c42bca6af)

  Di fitur sex terdapat 2 kategori yaitu male dan female, dari data presentase dapat disimpulkan 76% lebih dalam dataset ini merupakan laki-laki dan sisanya perempuan

- Fitur ChestPainType

  ![download (6)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/522532d3-f878-4e0e-a49b-af4cb52b5363)

  ChestPainType merupakan jenis nyeri dada yang dirasakan oleh pasien. Nilai "TA" mengindikasikan Typical Angina (nyeri dada khas), "ATA" mengindikasikan Atypical Angina (nyeri dada tidak khas), "NAP"
  mengindikasikan Non-Anginal Pain (nyeri dada non-angina), dan "ASY" mengindikasikan Asymptomatic (tanpa gejala nyeri dada). Mayoritas pasien tanpa gejala nyeri dada dan hanya sedikit pasien yang merasakan
  nyeri dada khas

- Fitur RestingECG

  ![download (7)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/8f316a5b-1ca4-4f8f-8c69-5637ceea4530)

  Hasil elektrokardiogram istirahat pasien. Nilai "Normal" menunjukkan hasil normal, "ST" menunjukkan adanya abnormalitas gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST > 0.05 mV), dan
  "LVH" menunjukkan kemungkinan atau pasti adanya hipertrofi ventrikel kiri berdasarkan kriteria Estes. Dari data presentase lebih dari 60% pasien hasil elektrodiagram istirahatnya normal.

- Fitur ExerciseAgina

  ![download (8)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/0cc5c8d3-aef0-441c-b5ee-787471c49389)

  Apakah pasien mengalami angina yang dipicu oleh latihan fisik. Nilai "Y" menunjukkan bahwa pasien mengalami angina saat berolahraga (Yes), sementara "N" menunjukkan sebaliknya (No). Sekitar 62% pasien tidak
  mengalami angina yang dipicu latihan fisik, hanya berkisar 37% pasien yang mengalaminya.

- Fitur ST_Slope

  ![download (9)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/36830e31-3133-4168-974a-ad7de24f24e1)

  Kemiringan segmen ST puncak latihan. Nilai "Up" menunjukkan kemiringan naik (upsloping), "Flat" menunjukkan kemiringan datar (flat), dan "Down" menunjukkan kemiringan turun (downsloping). Upsloping dan Flat
  menghasilkan data yang hampir sama.

- Melakukan analisis terhadap fitur numerik

  ![download (10)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/782d945b-e370-4ae4-8fec-72d328d588c6)

  Berdasarkan pengamatan histogram di atas, khususnya histogram untuk variabel "HeartDisease" yang merupakan fitur target (label) pada data. Dari histogram "HeartDisease", kita bisa memperoleh beberapa
  informasi, antara lain:
  - Data pasien yang tidak mengidap penyakit jantung lebih banyak sedikit dibanding data pasien yang menderita penyakit jantung

  Multivariate Analysis
  Multivariate EDA menunjukkan hubungan antara dua atau lebih variabel pada data. Multivariate EDA yang menunjukkan hubungan antara dua variabel biasa disebut sebagai bivariate EDA.
  - Mengecek rata-rata HeartDisease terhadap masing-masing fitur untuk mengetahui pengaruh fitur kategori terhadap HeartDisease.

    ![download (11)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/5a063b24-a68a-41e8-bbfd-ebda329c76c8)

    Dengan mengamati rata-rata HeartDisease relatif terhadap fitur kategori di atas, kita memperoleh insight sebagai berikut:
    - Pada fitur sex, jumlah penderita penyakit jantung paling banyak di derita Laki-laki bahkan hampir 3x lipatnya daripada perempuan.
    - Pada fitur ChestPainType, hasil pengamatan penderita jantung paling banyak justru pasien yang tidak memiliki gejala nyeri pada dada
    - Pada fitur RestingECG, hasil pengamatan pasien yang menunjukkan kemungkinan atau pasti adanya hipertrofi ventrikel kiri berdasarkan kriteria Estes "LVH" dan menunjukkan adanya abnormalitas gelombang
      ST-T (inversi gelombang T dan/atau elevasi atau depresi ST > 0.05 mV) "ST" memiliki angka yang cukup tinggi dan hampir sama
    - Pada fitur ExerciseAngina, pasien mengalami angina yang dipicu oleh latihan fisik
    - Pada fitur ST_Slope, pasien yang menunjukkan kemiringan datar (flat), dan menunjukkan kemiringan turun (downsloping) memiliki potensi yang menderita yang cukup tinggi.
   
  - Mengamati hubungan antara fitur numerik menggunakan fungsi pairplot().

    ![download (12)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/5dc62e11-3030-427e-9664-3370eb01dac5)

    Melihat relasi antara semua fitur numerik dengan fitur target kita yaitu ‘HeartDisease’.Mendapatkan hasil yang sulit diamati karena label kategorikal hanya memiliki dua nilai, yaitu 0 dan 1, maka titik-
    titik akan terkelompok dalam dua garis lurus, satu untuk nilai 0 dan satu untuk nilai 1. Hal ini menyulitkan pengamatan hubungan antara fitur numerik karena titik-titik yang memiliki nilai kategorikal sama
    akan saling bertumpuk.
    
  - Mengevaluasi skor korelasinya, gunakan fungsi corr().
 
    ![download (13)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/6600fbd3-cd3e-4ef3-bebd-8a197337577c)

    Koefisien korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1
    atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah.  Fitur ‘FastingBS’ memiliki korelasi yang sangat kecil (0.0). Sehingga, fitur tersebut dapat di-drop.

## Data Preparation
Teknik data preparation yang dilakukan : 
- Mencari apakah terdapat data duplikat. Dan hasilnya tidak terdapat data yang berduplikat
- Mengecek missing value pada kolom RestingBP dan Cholesterol, karena kolom ini seharusnya tidak boleh bernilai 0 atau kosong. Mendapatkan hasil :
  Nilai 0 di kolom RestingBP ada:  1
  Nilai 0 di kolom Cholesterol ada:  172
- Mengganti nilai yang hilang atau 0 dengan median
- Mengecek apakah terdapat outliner dengan memvisualisasikan data Heart Failure Prediction dengan boxplot untuk mendeteksi outliers pada beberapa fitur numerik
  
  ![download (1)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/10835e1e-2e8b-422b-920e-836b7547a96c)
  ![download (2)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/56b425c8-dd22-42dc-b4da-3ed9afbabe1f)
  ![download (3)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/2b14381a-9a78-4a08-8fec-9c11f37cd7aa)
  ![download (4)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/2e5214d0-1a4f-4407-8ea3-89a4852804e9)

  Pada beberapa fitur numerik di atas terdapat outliers. Selanjutnya adalah mengatasi outliers tersebut dengan metode yang IQR.
- Setelah dilakukan IQR Dataset sekarang telah bersih dan memiliki 643 sampel.
- Melakukan proses encoding fitur kategori
  Encoding fitur kategori adalah proses mengubah variabel kategori (data kualitatif) menjadi bentuk yang dapat digunakan dalam analisis atau pemodelan. Ini diperlukan  karena algoritma machine learning umumnya
  hanya dapat bekerja dengan data numerik. Menggunakan One-Hot Encoding: Ini mengubah setiap kategori menjadi kolom baru dan mengisinya dengan nilai 0 atau 1, menunjukkan apakah suatu sampel termasuk dalam
  kategori tersebut atau tidak. Ini baik untuk fitur kategori nominal di mana tidak ada urutan yang bermakna. Ini mencegah masalah ordinal yang terjadi pada Label Encoding.
- Train-Test-Split
  Pembagian data menjadi set pelatihan (training set) dan set pengujian (test set) adalah praktik umum dalam machine learning untuk mengukur kinerja model dengan benar dan menghindari masalah seperti
  overfitting. Rasio yang digunakan untuk membaginya yaitu 60:40, 60 data latih dan 40 data uji. Dari total 643 sampel, dibagi menjadi 385 data latih dan 258 data uji.

## Modeling

**Menyiapkan data frame untuk analisis
- import pandas as pd: Ini mengimpor pustaka pandas untuk bekerja dengan data dalam bentuk DataFrame, yang memudahkan manipulasi dan analisis data.
- import numpy as np: Ini mengimpor pustaka NumPy, yang merupakan pustaka fundamental untuk komputasi numerik dalam Python.
- from sklearn.model_selection import RandomizedSearchCV: Ini mengimpor kelas RandomizedSearchCV dari pustaka scikit-learn. Ini adalah alat yang digunakan untuk melakukan pencarian acak parameter yang optimal untuk model Anda. Ini membantu Anda menemukan kombinasi parameter yang memberikan kinerja terbaik tanpa harus mencoba semua kemungkinan kombinasi.
- from sklearn.neighbors import KNeighborsRegressor: Ini mengimpor kelas KNeighborsRegressor dari scikit-learn. Ini adalah algoritma K-Nearest Neighbors untuk tugas regresi (prediksi nilai numerik).
- from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor: Ini mengimpor kelas RandomForestRegressor dan GradientBoostingRegressor dari scikit-learn. Ini adalah algoritma ensemble (kumpulan) yang kuat untuk tugas regresi.
- from sklearn.metrics import mean_squared_error: Ini mengimpor fungsi mean_squared_error dari scikit-learn. Metrik evaluasi ini digunakan untuk menghitung rata-rata dari kuadrat perbedaan antara nilai yang diprediksi dan nilai yang sebenarnya.
- models = pd.DataFrame(index=['train_mse', 'test_mse'], columns=['KNN', 'RandomForest', 'Boosting']): Ini membuat DataFrame kosong dengan dua baris (train_mse dan test_mse) dan tiga kolom (KNN, RandomForest, Boosting) yang akan digunakan untuk menyimpan hasil evaluasi model.

**Membuat model KNN dengan Hyperparameter Random Search
- knn = KNeighborsRegressor(): Baris ini membuat sebuah objek model K-Nearest Neighbors dengan parameter default. Ini akan menjadi model dasar yang akan dioptimalkan menggunakan RandomizedSearchCV.
- param_dist_knn = { ... }: Ini adalah kamus yang berisi himpunan parameter yang akan dioptimalkan untuk model KNN. Parameter yang didefinisikan adalah:
  - n_neighbors: Jumlah tetangga yang akan diperiksa.
  - weights: Metode penimbangan tetangga ("uniform" atau "distance").
  - algorithm: Algoritma yang akan digunakan untuk menghitung tetangga ("auto", "ball_tree", "kd_tree", atau "brute").
  - p: Parameter p untuk metrik jarak (1 untuk manhattan_distance, 2 untuk euclidean_distance).
  - random_search_knn = RandomizedSearchCV(...): Ini adalah inisialisasi objek RandomizedSearchCV yang akan melakukan pencarian acak parameter terbaik untuk model KNN. Parameter yang disebutkan adalah:
    - knn: Objek model KNN yang telah dibuat sebelumnya.
    - param_distributions: Himpunan parameter yang akan dioptimalkan.
    - n_iter: Jumlah iterasi pencarian acak.
    - scoring: Metrik evaluasi yang digunakan untuk membandingkan kombinasi parameter (neg_mean_squared_error karena kita ingin meminimalkan MSE).
    - cv: Jumlah lipatan dalam validasi silang.
    - random_search_knn.fit(X_train, y_train): Baris ini melakukan pencarian parameter acak pada model KNN menggunakan data pelatihan.
- best_knn = random_search_knn.best_estimator_: Setelah pencarian selesai, ini menyimpan model KNN terbaik yang ditemukan selama pencarian acak.
- train_mse_knn = mean_squared_error(...): Menghitung Mean Squared Error pada data pelatihan menggunakan model KNN terbaik.
- test_mse_knn = mean_squared_error(...): Menghitung Mean Squared Error pada data pengujian menggunakan model KNN terbaik.
- models.loc['train_mse', 'KNN'] = train_mse_knn: Memasukkan nilai MSE pada data pelatihan ke dalam DataFrame models.
- models.loc['test_mse', 'KNN'] = test_mse_knn: Memasukkan nilai MSE pada data pengujian ke dalam DataFrame models.


Berikut adalah nilai parameter untuk algoritma K-Nearest Neighbors Regressor (KNN) :
- n_neighbors: Nilai-nilai yang diuji adalah 1 hingga 20.
- n_iter: Jumlah iterasi acak yang dilakukan dalam pencarian parameter acak (dalam kasus ini, 10 iterasi acak).
- scoring: Metrik penilaian yang digunakan dalam pencarian parameter (dalam kasus ini, 'neg_mean_squared_error').
- cv: Jumlah lipatan silang dalam validasi silang (dalam kasus ini, 5 lipatan silang).

**Membuat model Random Forest dengan Hyperparameter Random Search
- rf = RandomForestRegressor(): Pada baris ini, objek model Random Forest Regressor dibuat dengan parameter default. Ini adalah model dasar yang nantinya akan dioptimalkan menggunakan RandomizedSearchCV.
- param_dist_rf = { ... }: Ini adalah kamus yang berisi himpunan parameter yang akan dioptimalkan untuk model Random Forest. Parameter yang didefinisikan adalah:
- n_estimators: Jumlah pohon dalam ensemble (50, 100, 150, 200, 250).
- max_depth: Kedalaman maksimal setiap pohon (None untuk tidak ada batasan, atau rentang 5 hingga 20 dengan interval 5).
- min_samples_split: Jumlah sampel minimum yang diperlukan untuk membagi node dalam pohon.
- min_samples_leaf: Jumlah sampel minimum yang diperlukan dalam setiap daun pohon.
- random_search_rf = RandomizedSearchCV(...): Inisialisasi objek RandomizedSearchCV untuk mencari kombinasi parameter terbaik untuk model Random Forest. Parameter yang disebutkan adalah:
  - rf: Objek model Random Forest yang telah dibuat sebelumnya.
  - param_distributions: Himpunan parameter yang akan dioptimalkan.
  - n_iter: Jumlah iterasi pencarian acak.
  - scoring: Metrik evaluasi yang digunakan untuk membandingkan kombinasi parameter (neg_mean_squared_error karena kita ingin meminimalkan MSE).
  - cv: Jumlah lipatan dalam validasi silang.
  - random_search_rf.fit(X_train, y_train): Proses pencarian acak parameter terbaik untuk model Random Forest menggunakan data pelatihan.
- best_rf = random_search_rf.best_estimator_: Setelah pencarian selesai, ini menyimpan model Random Forest terbaik yang ditemukan selama pencarian acak.
- train_mse_rf = mean_squared_error(...): Menghitung Mean Squared Error pada data pelatihan menggunakan model Random Forest terbaik.
- test_mse_rf = mean_squared_error(...): Menghitung Mean Squared Error pada data pengujian menggunakan model Random Forest terbaik.
- models.loc['train_mse', 'RandomForest'] = train_mse_rf: Memasukkan nilai MSE pada data pelatihan ke dalam DataFrame models.
- models.loc['test_mse', 'RandomForest'] = test_mse_rf: Memasukkan nilai MSE pada data pengujian ke dalam DataFrame models.

Berikut adalah nilai parameter untuk algoritma Random Forest Regressor:
- n_estimators: Nilai-nilai yang diuji adalah 50, 100, 150, 200, dan 250.
- max_depth: Nilai-nilai yang diuji meliputi None (tidak ada batasan kedalaman) serta 5, 10, 15, dan 20.
- min_samples_split: Nilai-nilai yang diuji adalah dari 2 hingga 10.
- min_samples_leaf: Nilai-nilai yang diuji adalah dari 1 hingga 10.
- n_iter: Jumlah iterasi acak yang dilakukan dalam pencarian parameter acak (dalam kasus ini, 10 iterasi acak).
- scoring: Metrik penilaian yang digunakan dalam pencarian parameter (dalam kasus ini, 'neg_mean_squared_error').
- cv: Jumlah lipatan silang dalam validasi silang (dalam kasus ini, 5 lipatan silang).

**Membuat model GradientBoostingRegressor dengan Hyperparameter Random Search
- boosting = GradientBoostingRegressor(): Pada baris ini, objek model Gradient Boosting Regressor dibuat dengan parameter default. Ini adalah model dasar yang nantinya akan dioptimalkan menggunakan RandomizedSearchCV.
- param_dist_boosting = { ... }: Ini adalah kamus yang berisi himpunan parameter yang akan dioptimalkan untuk model Gradient Boosting. Parameter yang didefinisikan adalah:
  - n_estimators: Jumlah pohon dalam ensemble (50, 100, 150, 200, 250).
  - learning_rate: Tingkat pembelajaran untuk setiap pohon dalam ensemble.
  - max_depth: Kedalaman maksimal setiap pohon.
  - min_samples_split: Jumlah sampel minimum yang diperlukan untuk membagi node dalam pohon.
  - min_samples_leaf: Jumlah sampel minimum yang diperlukan dalam setiap daun pohon.
  - random_search_boosting = RandomizedSearchCV(...): Ini adalah inisialisasi objek RandomizedSearchCV untuk mencari kombinasi parameter terbaik untuk model Gradient Boosting. Parameter yang disebutkan adalah
    sama seperti sebelumnya:
    - boosting: Objek model Gradient Boosting yang telah dibuat sebelumnya.
    - param_distributions: Himpunan parameter yang akan dioptimalkan.
    - n_iter: Jumlah iterasi pencarian acak.
    - scoring: Metrik evaluasi yang digunakan untuk membandingkan kombinasi parameter (neg_mean_squared_error karena kita ingin meminimalkan MSE).
    - cv: Jumlah lipatan dalam validasi silang.
    - random_search_boosting.fit(X_train, y_train): Pencarian parameter acak pada model Gradient Boosting dilakukan menggunakan data pelatihan.
- best_boosting = random_search_boosting.best_estimator_: Setelah pencarian selesai, ini menyimpan model Gradient Boosting terbaik yang ditemukan selama pencarian acak.
- train_mse_boosting = mean_squared_error(...): Menghitung Mean Squared Error pada data pelatihan menggunakan model Gradient Boosting terbaik.
- test_mse_boosting = mean_squared_error(...): Menghitung Mean Squared Error pada data pengujian menggunakan model Gradient Boosting terbaik.
- models.loc['train_mse', 'Boosting'] = train_mse_boosting: Memasukkan nilai MSE pada data pelatihan ke dalam DataFrame models.
- models.loc['test_mse', 'Boosting'] = test_mse_boosting: Memasukkan nilai MSE pada data pengujian ke dalam DataFrame models.


Berikut adalah nilai parameter untuk algoritma Gradient Boosting Regressor:
- n_estimators: Nilai-nilai yang diuji adalah 50, 100, 150, 200, dan 250.
- learning_rate: Nilai-nilai yang diuji adalah 0.001, 0.01, 0.1, 0.2, dan 0.3.
- max_depth: Nilai-nilai yang diuji adalah 3, 4, 5, 6, 7, 8, 9, dan 10.
- min_samples_split: Nilai-nilai yang diuji adalah dari 2 hingga 10.
- min_samples_leaf: Nilai-nilai yang diuji adalah dari 1 hingga 10.
- n_iter: Jumlah iterasi acak yang dilakukan dalam pencarian parameter acak (dalam kasus ini, 10 iterasi acak).
- scoring: Metrik penilaian yang digunakan dalam pencarian parameter (dalam kasus ini, 'neg_mean_squared_error').
- cv: Jumlah lipatan silang dalam validasi silang (dalam kasus ini, 5 lipatan silang).
  
## Evaluation
Untuk mengukur evaluasi menggunakan metrik MSE
MSE (Mean Squared Error):
MSE adalah metrik evaluasi yang mengukur rata-rata dari kuadrat perbedaan antara nilai prediksi dan nilai yang sebenarnya. Dalam konteks ini, MSE sangat cocok karena:
- Sensitivitas terhadap Kesalahan: MSE memberikan bobot lebih besar pada kesalahan besar antara prediksi dan nilai yang sebenarnya. Karena penyakit kardiovaskular adalah masalah serius yang memerlukan prediksi risiko yang akurat, mengurangi kesalahan besar sangatlah penting.
- Pentingnya Deteksi Risiko Tinggi: MSE mempertimbangkan setiap prediksi dengan proporsional terhadap kesalahan kuadrat. Ini sangat relevan dalam mengukur performa model dalam mengidentifikasi individu dengan risiko penyakit kardiovaskular yang lebih tinggi, yang menjadi salah satu tujuan dari solusi yang diinginkan.
- Tujuan Optimasi: Tujuan dalam pengembangan model adalah mengoptimasi parameter untuk menghasilkan prediksi yang seminimal mungkin kesalahan. MSE secara langsung mengukur sejauh mana model mendekati nilai yang benar, sehingga konsisten dengan tujuan optimasi.
- Penekanan pada Kesalahan: Kesalahan besar akan berkontribusi secara signifikan pada nilai MSE yang tinggi, mendorong pengembangan model yang lebih baik dan lebih akurat.
Ketika tujuan adalah meningkatkan akurasi prediksi risiko penyakit kardiovaskular dan mengidentifikasi individu dengan risiko yang lebih tinggi, MSE menyediakan indikator yang kuat dan relevan

- Melakukan uji evaluasi model

|           |   train  |   test   |
| :---      |   :---:  |  :---:   |
| KNN       | 0.000082 | 0.000142 |
| RF        | 0.000016 | 0.000133 |
| Boosting  | 0.000033 | 0.000138 |

Dari hasil evaluasi model menggunakan MSE diperoleh hasil bahwa Random Forest memiliki tingkat error yang lebih rendah dibandingkan yang lainnya.

![download (14)](https://github.com/arifhendrawan023/Submission-1-Predictive-Analytic/assets/55530939/2315cbf4-013b-4e1d-8cd0-26990614f021)

Dari gambar di atas, terlihat bahwa, model Random Forest (RF) memberikan nilai eror yang paling kecil. Sedangkan model dengan algoritma KNN memiliki eror yang paling besar. Model inilah yang akan kita pilih sebagai model terbaik untuk melakukan prediksi

- Menguji dengan membuat prediksi

|        |   y_true  | prediksi_KNN | prediksi_RF | prediksi_Boosting |
| :---   |   :---:   |  :---:       |    :---:    |       :---:       |
| 63     |     1     |      0.8     |      1.0    |         0.9       |


Berdasarkan hasil prediksi tampak bahwa setiap model (KNN, Random Forest, dan Boosting) memberikan prediksi yang berbeda untuk satu data uji dengan nilai aktual yaitu 1.

y_true: Ini adalah nilai aktual atau target yang sesungguhnya untuk data uji tersebut. Dalam kasus ini, nilai aktualnya adalah 1.

prediksi_KNN: Hasil prediksi dari model K-Nearest Neighbors (KNN) untuk data uji tersebut. Nilai prediksi yang diberikan oleh model KNN adalah 0.8.

prediksi_RF: Hasil prediksi dari model Random Forest untuk data uji tersebut. Nilai prediksi yang diberikan oleh model Random Forest adalah 1.0.

prediksi_Boosting: Hasil prediksi dari model Boosting (misalnya Gradient Boosting) untuk data uji tersebut. Nilai prediksi yang diberikan oleh model Boosting adalah 0.9.

Dalam konteks ini, setiap model memberikan prediksi yang berbeda untuk data uji yang sama. Model Random Forest memprediksi nilai 1.0 yang sesuai dengan nilai aktual, sementara model KNN dan Boosting memberikan prediksi yang sedikit berbeda.




