# Submission 2 : MLOps - Sentiment Analysis

Nama: Wendi Kardian

Username Dicoding: wendie13


| Deskripsi | |
|--|--|
| Dataset | [Dataset](https://www.kaggle.com/datasets/dineshpiyasamara/sentiment-analysis-dataset) |
| Masalah | Di era digital saat ini, analisis sentimen menjadi sangat penting karena membantu organisasi dan perusahaan dalam memahami opini publik di media sosial dan platform online. Dengan kemampuan ini, para pemangku kepentingan dapat merespons cepat terhadap perubahan sentimen, menjaga reputasi merek, dan membuat keputusan strategis yang lebih baik untuk memenuhi kebutuhan dan harapan pelanggan dalam lingkungan digital yang dinamis. |
| Solusi Machine Learning | Dibuat untuk membuat sistem machine learning yang dapat menganalisis sentimen dalam teks, menentukan apakah bersifat positif atau negatif. |
| Metode Pengolahan | Dalam proyek ini, metode pengolahan yang digunakan melibatkan tokenisasi teks (teks dari berita) yang diubah menjadi urutan angka untuk merepresentasikan teks tersebut. Pendekatan ini memungkinkan model analisis sentimen untuk lebih efektif memahami dan mengidentifikasi sentimen dalam teks, sehingga meningkatkan kemampuan model dalam menganalisis opini atau perasaan dalam berita. |
| Arsitektur Model | Model ini memanfaatkan layer TextVectorization sebagai langkah awal mengolah input string dan mengubahnya menjadi urutan angka yang dapat dimengerti oleh model untuk analisis sentimen. Layer Embedding digunakan untuk memahami kedekatan kata, memungkinkan model menilai konotasi positif atau negatif setiap kata. Selanjutnya, dua hidden layer menggali informasi kompleks dari representasi teks, diikuti oleh satu output layer untuk memprediksi sentimen. Pendekatan ini dirancang untuk meningkatkan kemampuan model dalam menangkap nuansa sentimen dalam teks input. |
| Metrik Evaluasi | Metrik evaluasi mencakup ExampleCount, AUC, serta detail seperti FalsePositives, TruePositives, FalseNegatives, dan TrueNegatives, memberikan gambaran menyeluruh tentang akurasi dan performa model klasifikasi biner dengan implementasi threshold dan batas parameter. Pendekatan ini memungkinkan analisis mendalam terhadap kesalahan dan perubahan kinerja yang signifikan. |
| Performa Model | Model yang dikembangkan menunjukkan performa yang sangat baik dalam memprediksi sentimen dari teks berita yang diinputkan, dengan keakuratan mencapai 93% pada data pelatihan dan validasi. Hasil ini menunjukkan kemampuan model dalam memahami dan mengklasifikasikan sentimen dalam teks berita dengan akurasi tinggi. |
| Opsi Deployment | Deployment menggunakan Cloud Run di GCP dengan alokasi CPU selama pemrosesan permintaan dan peningkatan CPU saat startup. Konfigurasi: maksimum 80 permintaan bersamaan, timeout 300 detik, maksimum 100 instance dengan auto-scaling. Menggunakan Docker image  CPU 1 core dan memori 512MiB per instance. |
| Web App | [Web App](https://sentiment-546041470502.asia-southeast2.run.app/v1/models/sentiment/metadata) |
| Monitoring | Dashboard monitoring MLOps ini menampilkan beberapa metrik untuk memantau latensi permintaan pada model machine learning. Grafik menunjukkan bahwa latensi tetap stabil sepanjang waktu yang dipantau, tanpa adanya lonjakan yang signifikan. Ini mengindikasikan bahwa model beroperasi dengan performa yang konsisten dan responsif. |
|Link monitoring | [Prometheus](https://monitoring-546041470502.asia-southeast2.run.app/) [Grafana](http://34.128.78.227:3000/) |