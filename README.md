# Machine Learning Terapan - Proyek Pertama

## Domain Proyek
<p align=justify>Pada proyek ini mengambil permasalahan tentang regresi. Regresi adalah salah satu metode untuk menentukan hubungan sebab-akibat antara variabel dengan variabel lainnya. Topik yang diambil yaitu prediksi harga jual mobil bekas. Tingginya harga jual mobil baru membuat banyak orang dengan finansial yang pas-pasan memilih untuk membeli mobil bekas saja. Terlebih disaat sulit seperti pada pandemi sekarang ini ada beberapa orang yang memutuskan untuk membeli mobil bekas dan ada pula yang memutuskan untuk menjual mobilnya. Memperkirakan harga jual mobil bekas ini penting dalam bidang komersial baik untuk yang membeli maupun yang menjual. Jika dilihat dari segi penjual, ketika tidak bisa memperkirakan harga jual dengan baik, dimana semisal malah menerapkan harga yang terlalu tinggi juga akan membuat calon pembeli mundur atau jika harga terlalu rendah maka harus siap untuk mendapat keuntungan yang sedikit. Sedangkan jika dilihat dari segi pembeli maka jika ingin membeli suatu mobil bekas sudah ada perkiraan harga, sehingga bisa mempersiapkan keuangan dengan baik dan juga dapat terhindar dari harga yang terlalu tinggi yang membuat pembeli merasa rugi.
 
[Link jurnal terkait](https://www.researchgate.net/publication/319306871_Predicting_the_Price_of_Used_Cars_using_Machine_Learning_Techniques "Jurnal Referensi")
 
## Business Understanding
### Problem Statements
- Bagaimana cara memprediksi harga jual mobil bekas berdasarkan data yang sudah ada?
- Fitur apa yang paling mempengaruhi harga jual mobil bekas?
### Goals
- Mampu memprediksi harga jual mobil bekas dengan efisien dan akurat.
- Dapat mengetahui fitur-fitur apa saja yang sangat mempengaruhi harga jual pada mobil bekas.
### Solution statements
<p align=justify>Berdasarkan permasalahan yang ada di atas, maka solusi yang akan ditawarkan yaitu membuat sistem yang dapat memprediksi harga jual mobil bekas dengan efisien dan akurat menggunakan 6 pilihan algoritma machine learning yaitu :</p>
 
- K-Nearest Neighbors (KNN)
	<p align=justify>Nearest Neighbors Regression adalah algoritma yang digunakan untuk menyelesaikan masalah regresi berdasarkan k-nearest neighbor. Neighbors-based regression dapat digunakan ketika label data bernilai kontinu. Label yang ditetapkan ke titik query dihitung berdasarkan rata-rata label tetangga terdekatnya. KNeighborsRegressor mengimplementasikan pembelajaran berdasarkan tetangga terdekat dari setiap titik query, yang merupakan nilai integer yang ditentukan oleh pengguna.</p>
- Random Forest
	<p align=justify>Random Forest adalah meta estimator yang menyesuaikan dengan sejumlah klasifikasi pohon keputusan pada berbagai sub-sampel dari dataset dan menggunakan rata-rata untuk meningkatkan akurasi prediksi dan kontrol overfitting. Ukuran sub-sampel dikontrol dengan parameter max_samples jika bootstrap=True (default), jika tidak maka seluruh dataset digunakan untuk membangun setiap pohon.</p>
- Boosting Algorithm 
	<p align=justify>Salah satu algoritma boosting yang populer adalah AdaBoost yang dikenalkan oleh Freund dan Schapire pada tahun 1995. Prinsip inti dari AdaBoost adalah menyesuaikan urutan pembelajaran yang lemah (yaitu, model yang hanya sedikit lebih baik daripada tebakan acak, seperti pohon keputusan sederhana) pada versi data yang dimodifikasi berulang kali. Prediksi dari keseluruhannya kemudian digabungkan melalui suara mayoritas (atau jumlah) untuk menghasilkan prediksi akhir. Pada kasus regresi, AdaBoost yang digunakan adalah AdaBoost Regressor yang merupakan sebuah meta-estimator yang dimulai dengan memasang sebuah regressor pada dataset asli lalu kemudian menyesuaikan salinan tambahan dari regressor pada dataset yang sama tetapi bobot instance disesuaikan dengan prediksi error saat ini. Dengan demikian, regressor berikutnya lebih fokus pada kasus-kasus sulit.</p>
- Decision Tree
	<p align=justify>Decision Trees (DTs) adalah metode supervised learning non-parametrik yang digunakan untuk klasifikasi dan regresi. Tujuannya adalah untuk membuat model yang memprediksi nilai variabel target dengan mempelajari aturan keputusan sederhana yang disimpulkan dari fitur data. Sebuah pohon dapat dilihat sebagai pendekatan konstan sepotong demi sepotong. Semakin panjang pohonnya, semakin kompleks aturan keputusannya dan semakin fit modelnya.</p>
- Linear Regression
	<p align=justify>Linear regression merupakan salah satu teknik regresi yang paling banyak digunakan. Metode ini adalah salah satu metode regresi paling sederhana. Salah satu keunggulan utamanya adalah kemudahan dalam menginterpretasikan hasil. Linear Regression cocok dengan model linier dengan koefisien w = (w1, …, wp) untuk meminimalkan jumlah sisa kuadrat antara target yang diamati dalam kumpulan data dan target yang diprediksi oleh pendekatan linier.</p>
- Support Vector Machine (SVM)
	<p align=justify>SVM adalah metode supervised learning yang digunakan untuk klasifikasi, regresi, dan deteksi outlier.</p>
 
  Keuntungan dari SVM adalah:
  - Efektif dalam ruang dimensi tinggi.
  - Masih efektif dalam kasus di mana jumlah dimensi lebih besar dari jumlah sampel.
  - Menggunakan subset titik pelatihan dalam fungsi keputusan (disebut vektor pendukung), sehingga juga hemat memori.
  - Serbaguna: fungsi Kernel yang berbeda dapat ditentukan untuk fungsi keputusan. Kernel umum disediakan, tetapi juga dimungkinkan untuk menentukan kernel khusus.<br>
 
  Kerugian dari mesin vektor dukungan meliputi:
  - Jika jumlah fitur jauh lebih besar daripada jumlah sampel, hindari kecocokan berlebihan dalam memilih fungsi Kernel dan istilah regularisasi sangat penting.
  - SVM tidak secara langsung memberikan perkiraan probabilitas, ini dihitung menggunakan five-fold cross-validation yang mahal.
 
  <p align=justify>Metode SVM yang digunakan untuk menyelesaikan masalah regresi adalah Support Vector Regression. Secara analog, model yang dihasilkan oleh Support Vector Regression hanya     bergantung pada subset dari data pelatihan, karena cost function mengabaikan sampel yang prediksinya mendekati targetnya.</p>
 
## Data Understanding
<p align=justify>Dataset yang diambil berasal dari Kaggle. Dataset tersebut berisi data-data tentang mobil bekas beserta harga jualnya yang ada di UK. Terdapat banyak merk mobil yang disediakan pada situs kaggle “100,000 UK Used Car Dataset”, tetapi pada proyek ini hanya mengambil dataset dengan merk mobil Audi. Dataset ini terdiri dari 3 fitur kategorikal dan 6 fitur numerik. Kesembilan fitur tersebut yaitu informasi tentang model mobil, tahun registrasi, harga dalam euro, transmisi, jarak yang ditempuh, jenis bahan bakar, pajak jalan, miles per gallon (mpg), dan ukuran mesin dalam liter. Pada proyek ini fitur price (harga) dijadikan sebagai variabel dependen (variabel target atau variabel terikat) dan variabel yang lainnya sebagai variabel independen (variabel bebas).</p>
 
<p align=justify>Secara rinci, variabel-variabel pada Audi dataset adalah sebagai berikut:</p>
	
  - Model adalah data yang berisi model-model dari mobil Audi. Pada dataset terdapat 26 model audi yaitu ' A1', ' A6', ' A4', ' A3', ' Q3', ' Q5', ' A5', ' S4', 'Q2', ' A7', ' TT', ' Q7', ' RS6', ' RS3', ' A8', ' Q8', ' RS4', ' RS5', ' R8', ' SQ5', ' S8', ' SQ7', ' S3', ' S5', ' A2', dan ' RS7'.
  - Year (tahun) adalah kolom yang berisi nilai tentang tahun berapa mobil tersebut teregistrasi.
  - Price (harga) adalah kolom yang berisi nilai harga jual mobil bekas tersebut. Fitur harga ini nantinya yang akan dijadikan target.
  - Transmission (transmisi) adalah kolom yang berisi jenis transmisi apa mobil tersebut. Terdapat 3 jenis transmisi yaitu manual, semi-automatic, dan automatic.
  - Mileage (jarak tempuh) adalah kolom yang berisi jumlah total mil yang ditempuh dalam waktu tertentu atau seberapa jauh mobil tersebut sudah menempuh perjalanan.
  - FuelType (jenis bahan bakar) adalah kolom yang berisi informasi tentang bahan bakar apa yang digunakan oleh mobil tersebut. Terdapat 2 jenis bahan bakar yaitu petrol dan diesel.
  - Tax (pajak) adalah kolom yang berisi informasi tentang seberapa besar pajak yang harus dibayar suatu kendaraan beroda untuk digunakan di jalan umum.
  - Miles per gallon (mpg) adalah kolom yang berisi informasi tentang jarak yang diukur dalam mil, yang dapat ditempuh mobil per galon bahan bakar.
  - EngineSize (ukuran mesin) adalah kolom yang berisi informasi tentang ukuran seberapa luas ruang piston mesin beroperasi. Angka yang lebih besar berarti setiap piston mampu mendorong lebih banyak udara dan bahan bakar melalui mesin mobil setiap kali bergerak.
	
  [Link download dataset](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes?select=audi.csv "Dataset Kaggle")
 
## Data Preparation
Pada tahap data preparation ini dibagi menjadi 3 tahap pengerjaan yaitu :
- Encoding fitur kategori
	<p align=justify>Proses encoding adalah mengubah fitur kategorikal menjadi variabel numerik. Seperti yang sudah dijelaskan sebelumnya bahwa dalam dataset yang digunakan memiliki fitur kategorikal, sedangkan pada proyek ini akan menyelesaikan masalah regresi yang mana semua fitur yang digunakan haruslah dalam bentuk numerik. Sehingga pada dataset ini perlu dilakukan proses encoding. Pada proyek ini terdapat 2 teknik encoding yang digunakan yaitu one hot encoding dan ordinal encoding. One hot encoding adalah proses membuat kolom baru dari variabel kategorikal di mana setiap kategori menjadi kolom baru dengan nilai 0 atau 1 (0 mewakili tidak ada dan 1 mewakili ada). Teknik one hot encoding ini digunakan pada fitur transmission dan fuelType. Sedangkan ordinal encoding adalah proses mengubah fitur kategori sebagai array integer. Input ke transformator ini harus berupa bilangan bulat atau string seperti array, yang menunjukkan nilai yang diambil oleh fitur kategorikal (diskrit). Fitur dikonversi ke bilangan bulat ordinal sehingga menghasilkan satu kolom bilangan bulat (0 hingga n_kategori - 1) per fitur. Teknik ordinal encoding ini digunakan pada fitur model.</p>
- Train test split
	<p align=justify>Proses selanjutnya pada data preparation yaitu train test split. Proses ini dilakukan dengan tujuan membagi data latih dan data uji sesuai dengan porsi yang diinginkan. Hal ini dilakukan agar nantinya tidak mengotori data uji dengan informasi yang kita dapat dari data latih karena data uji akan berperan sebagai data baru. Pada proyek ini train test split dibagi dengan rasio 70:30, dimana 70% dari keseluruhan data berperan sebagai data latih dan 30% sisanya berperan sebagai data uji.</p>
- Standarisasi
	<p align=justify>Proses terakhir yang dilakukan pada tahap data preparation adalah standarisasi. Proses standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Data dengan skala relatif sama atau mendekati distribusi normal akan lebih mudah serta memiliki performa yang lebih baik untuk dimodelkan. Karena pada dataset yang digunakan ada beberapa fitur yang tidak memiliki skala yang sama sehingga untuk lebih meningkatkan performa dilakukanlah proses standarisasi ini. Teknik standarisasi yang digunakan yaitu standard scaler, yang melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.</p>
 
## Modeling
<p align=justify>Untuk menyelesaikan permasalahan yang sudah dijelaskan di atas, pada proyek ini menggunakan 6 jenis algoritma untuk dibandingkan yaitu K-Nearest Neighbor, Random Forest, Boosting Algorithm, Decision Tree Algorithm, Linear Regression Algorithm, dan SVM Algorithm. Keenam algoritma pada awalnya dibuat dengan menggunakan baseline model, selanjutnya 5 dari algoritma tersebut diolah dengan metode hyperparameter tuning untuk membandingkan bagaimana performa menggunakan baseline model dengan hyperparameter tuning. Pada algoritma Linear Regression tidak dibuat hyperparameter tuning karena pada algoritma tersebut tidak terdapat parameter yang bisa dibuat hyperparameter tuning. Hyperparameter tuning dilakukan untuk mendapatkan nilai parameter terbaik yang mana menghasilkan performa model yang baik pula. Teknik hyperparameter yang digunakan pada model ini adalah GridSearch. Dalam metode GridSearch terdapat satu parameter bernama param_grid yang berfungsi untuk memberikan nilai-nilai parameter suatu algoritma yang ingin dicocokkan nantinya. Teknik ini bekerja dengan cara mengombinasikan seluruh nilai parameter yang ditentukan untuk estimator, sehingga tidak ada yang terlewat. Semua kemungkinan kombinasi nilai parameter dievaluasi dan kombinasi terbaik dipertahankan. Nilai MSE yang didapatkan dari keenam algoritma tersebut beserta hyperparameter tuning nya dapat dilihat pada tabel berikut :</p>                
 
Algoritma | Train | Test
------------ | ------------- | -------------
KNN | 3915.44 | 4234.56
Random Forest | 764.382 | 3580.44
Boosting | 9499.99 | 9612.96
Decision Tree | 94.8128 | 6350.96
Linear Regression | 8083.89 | 7604.88
SVM | 51490.6 | 50409.2
KNN Hyperparameter Tuning | 2934.81 | 4049.85
Random Forest Hyperparameter Tuning | 2513.12 | 3494.36
Boosting Hyperparameter Tuning | 8054.92 | 8201.74
Decision Tree Hyperparameter Tuning | 2950.2 | 4374.3
SVM Hyperparameter Tuning | 5360.71 | 5133.72
 
Untuk lebih memudahkan, maka di bawah ini adalah plot yang mengurutkan nilai MSE test dari yang terkecil hingga yang terbesar:
 
[![MSE.png](https://i.postimg.cc/Cdphvxvw/MSE.png)](https://postimg.cc/2LTNyrRt)
 
<p align=justify>Dari gambar di atas dapat diketahui bahwa hasil yang paling bagus didapatkan dengan menggunakan algoritma random forest yang sudah diproses dengan hyperparameter tuning yaitu dengan nilai MSE train sebesar 2513.12 dan nilai MSE test sebesar 3494.36. Lalu dapat disimpulkan pula bahwa algoritma yang menggunakan proses hyperparameter tuning akan menghasilkan model terbaik dibandingkan algoritma yang hanya menggunakan baseline model. Hal ini dapat dilihat dari gambar di atas, dimana algoritma yang menggunakan proses hyperparameter tuning hasilnya selalu lebih bagus daripada baseline modelnya.</p>
 
## Evaluation
<p align=justify>Untuk mengukur kinerja model, maka digunakanlah metrik evaluasi yaitu Mean Squared Error (MSE). Metrik ini digunakan karena proyek yang dikerjakan mengangkat masalah regresi, sehingga akan tepat jika dievaluasi dengan metrik MSE. Cara kerja MSE yaitu dengan menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi. Nilai akhir yang dihasilkan oleh MSE adalah nilai error. Semakin kecil nilai error, maka kinerja model akan semakin bagus. Nilai MSE dihitung dengan menggunakan rumus sebagai berikut :</p>
 
[![rumus-mse.png](https://i.postimg.cc/KjGS43sk/rumus-mse.png)](https://postimg.cc/ZBMDMnm4)
 
Keterangan:
<br>N = jumlah dataset
<br>yi = nilai sebenarnya
<br>y_pred = nilai prediksi
 
Keunggulan MSE :
- Grafik MSE dapat dibedakan, sehingga dapat dengan mudah menggunakannya sebagai fungsi loss.
 
Kekurangan MSE :
- Nilai yang didapatkan setelah menghitung MSE adalah satuan keluaran kuadrat. Misalnya variabel output dalam meter(m) maka setelah menghitung MSE output yang didapatkan adalah dalam bentuk meter kuadrat.
- Jika dataset memiliki outlier maka itu akan menghasilkan penalti outlier paling banyak dan MSE yang dihitung lebih besar. Sehingga tidak Robust terhadap outlier yang merupakan keunggulan di MAE.
 
Cara mengimplementasikan MSE pada program yaitu menggunakan kode program sebagai berikut :
 
```python
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_test,y_pred))
```
 
<p align=justify>Tetapi sebelum menggunakan kode tersebut pastikan terlebih dahulu bahwa data uji sudah dilakukan proses scaling pada fitur numerik sama seperti yang dilakukan pada data latih.</p>
 
***Referensi***
- Dokumentasi Scikit-learn : https://scikit-learn.org/stable/modules/classes.html
