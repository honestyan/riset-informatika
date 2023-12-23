# Application of PERN Stack with Product Recommendations and Detection of Skin Problems Using CNN Transfer Learning

## Persoalan Praktis

- Kesulitan konsumen dalam menemukan produk yang sesuai dengan kebutuhan dan preferensi mereka di antara banyaknya variasi produk yang tersedia.

- Lamanya waktu diagnosa yang dibutuhkan untuk mendeteksi masalah kulit akibat pemeriksaan yang bersifat manual.

## Latar Belakang

Kulit Anda adalah organ terbesar Anda—itu sudah cukup untuk memberi tahu Anda bahwa perawatan kulit itu penting. Namun tetap saja, kami tahu banyak orang berasumsi bahwa hal tersebut tidak benar. Yang lain tidak yakin. Oleh karena itu, alasan “mengapa perawatan kulit itu penting” menjadi salah satu penelusuran Google terkait kulit teratas.

Tujuannya untuk menjaga isi perut kita tetap dalam mungkin tampak sederhana, namun jika dicermati, hal ini memainkan sejumlah peran yang mengejutkan dalam kehidupan kita.

Antara kita dan dunia luar terdapat sebuah antarmuka yang membentuk hampir 16% dari berat badan kita, yaitu KULIT kita – organ terbesar dalam tubuh kita.

Laporan terbaru dari Organisasi Kesehatan Dunia (WHO) menunjukkan bahwa penyakit kulit merupakan penyakit yang paling umum diderita manusia dan menyerang hampir 900 juta orang di dunia setiap saat.

Ada banyak dokter, ahli dermatologi, dan peneliti yang melakukan yang terbaik dalam bidang ilmu kedokteran ini. Bahkan terdapat model Pembelajaran Mesin dengan akurasi tinggi untuk membantu para peneliti di bidang ini dalam membuat penelitian mereka lebih efektif, namun semua kemajuan teknologi ini masih jauh dari kemampuan masyarakat umum. Kita sebagai masyarakat belum memanfaatkan kemajuan teknologi ini.

Mengingat terbatasnya sumber daya teknologi, masih terdapat cakupan luas yang dapat mencakup kehidupan masyarakat yang kurang atau tidak memiliki pengetahuan tentang penyakit kulit tersebut. Masih ada revolusi yang menunggu untuk mengubah semua kehidupan ini.

Di dunia di mana penyakit kulit memiliki stigma dan dianggap tabu, kami ingin membuat aplikasi yang dapat membuat perbedaan. Kami ingin membuat sesuatu yang mudah dan segera tersedia bagi pengguna, sesuatu yang dapat membantu orang dalam berbagai cara. Oleh karena itu, kami memutuskan untuk mengembangkan aplikasi ini, yang merupakan prediktor penyakit kulit berbasis web. WHO mengatakan, "Pendidikan kesehatan dasar diperlukan untuk mengidentifikasi gangguan ini sehingga dapat membantu kita mengurangi angka kesakitan secara substansial", dan aplikasi ini adalah sebuah langkah kecil ke arah ini.

Berbicara tentang India, laporan menunjukkan bahwa prevalensi penyakit kulit sekitar 60%. Dengan kata lain, sekitar 60% populasi pada kelompok usia 21 dan 55 tahun menderita kelainan kulit. Data ini membuat kelainan kulit semakin mengkhawatirkan, namun masyarakat memilih untuk mendiagnosis diri sendiri sehingga dapat memperburuk kelainan tersebut. Dengan aplikasi ini, kami bertujuan untuk memberikan diagnosis yang tepat kepada orang-orang di seluruh dunia.

Menggunakan aplikasi ini tidak memerlukan pengetahuan sebelumnya. Semudah menggunakan aplikasi Gmail Anda.

Aplikasi ini dapat digunakan sebagai aplikasi asli oleh pengguna untuk memprediksi kelainan kulit hanya dengan mengunggah gambar kulitnya. Aplikasi ini menggunakan model pembelajaran mesin yang dilatih untuk mengidentifikasi 10 penyakit kulit yang berbeda. Dukungan deteksi real-time di aplikasi ini memungkinkan pengguna bahkan menggunakan kamera mereka dan mendapatkan hasil instan tepat di depan mereka.

Lebih lanjut, aplikasi ini juga merekomendasikan dokter terdekat secara otomatis dengan menggunakan lokasi mereka saat ini di Google Map.

Kami juga menyertakan fitur yang memungkinkan pengguna menjadwalkan janji temu dengan dokter kulit untuk membantu pengguna mendapatkan diagnosis dari dokter sungguhan, dan mengatasi keraguan mereka. Dengan cara ini, pengguna bahkan dapat meminta bantuan dari dokter sungguhan melalui fitur ini tanpa perlu keluar rumah.

## Research Questions

- Bagaimana penerapan content-based filtering recommendation system dapat meningkatkan relevansi rekomendasi produk pada aplikasi e-commerce?

- Seberapa tinggi akurasi deteksi masalah kulit yang dapat dicapai dengan menerapkan CNN transfer learning pada aplikasi telemedicine?

## Tinjauan Pustaka

- Gupta, R. et al. (2019). A content-based recommendations algorithm for ecommerce purchases. Information Technology, 65(3), 1-9.

- Wang, F. et al. (2020). Accurate skin disease classification using single CNN transfer learning. Nature Medicine, 26(6), 864–871.

- Mahbod, A. et al. (2020). Skin disease detection using deep convolutional neural networks. Computers in Biology and Medicine, 124, 103929.

- Oktavian, A. (2019). Developing a PERN-stack based enterprise application. Journal of Software Engineering, 15(2), 104-114.

## ⁠Metode yang Digunakan

Kami akan menggunakan Mobilenet karena arsitekturnya ringan. Ia menggunakan konvolusi yang dapat dipisahkan secara mendalam yang pada dasarnya berarti ia melakukan konvolusi tunggal pada setiap saluran warna daripada menggabungkan ketiganya dan meratakannya. Hal ini mempunyai efek memfilter saluran masukan. Atau seperti yang dijelaskan dengan jelas oleh penulis makalah: “ Untuk MobileNets, konvolusi mendalam menerapkan filter tunggal ke setiap saluran masukan. Konvolusi searah kemudian menerapkan konvolusi 1×1 untuk menggabungkan keluaran konvolusi mendalam. Konvolusi standar memfilter dan menggabungkan masukan menjadi serangkaian keluaran baru dalam satu langkah. Konvolusi yang dapat dipisahkan secara mendalam membaginya menjadi dua lapisan, lapisan terpisah untuk pemfilteran dan lapisan terpisah untuk penggabungan. Faktorisasi ini berdampak pada pengurangan komputasi dan ukuran model secara drastis. ”

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*L97mX8J7dBNPtRwb5VwqUw.png"></p>

Jadi arsitektur Mobilenet secara keseluruhan adalah sebagai berikut, memiliki 30 lapisan

1. lapisan konvolusional dengan langkah 2
2. lapisan yang mendalam
3. lapisan runcing yang menggandakan jumlah saluran
4. lapisan mendalam dengan langkah 2
5. lapisan runcing yang menggandakan jumlah saluran
   dll.

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*lrxsPkbVrrIPVmr7jy-noA.png"></p>

Perawatannya juga sangat rendah sehingga bekerja cukup baik dengan kecepatan tinggi. Ada juga banyak jenis model terlatih dengan ukuran jaringan di memori dan disk sebanding dengan jumlah parameter yang digunakan. Kecepatan dan konsumsi daya jaringan sebanding dengan jumlah MAC (Multiply-Accumulates) yang merupakan ukuran jumlah operasi Perkalian dan Penjumlahan yang digabungkan.

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*XeJGMg7siqgjI6kQ3gke9A.png"></p>

## Dataset

- Notebook for skin cancer problem : https://www.kaggle.com/naim99/skin-cancer-detection-using-pytorch
- Dataset for segmented skin cancer images: https://www.kaggle.com/naim99/segmented-images-of-the-skin-cancer-dataset
- Diverse Dermatology Images: https://ddi-dataset.github.io/
- Dermnet NZ: https://dermnetnz.org
- DermIS: https://dermis.net
- Dermatology Atlas: http://www.atlasdermatologico.com.br/browse.jsf

## Kode Program

- [predict.py](https://github.com/honestyan/riset-informatika/blob/main/predict.py)
  git

## Contoh hasil survey

<p align="center"><img src="https://raw.githubusercontent.com/ishubham21/infinity-skncure-angular/master/readme-assets/feedback1.png"></p>

<p align="center"><img src="https://raw.githubusercontent.com/ishubham21/infinity-skncure-angular/master/readme-assets/feedback2.png"></p>

<p align="center"><img src="https://raw.githubusercontent.com/ishubham21/infinity-skncure-angular/master/readme-assets/orgs.png"></p>

# Aplikasi ini dibangun berdasarkan penelitian sebelumnya yang dilakukan di bidang ini

- [World Health Organization](https://apps.who.int/iris/bitstream/handle/10665/69229/WHO_FCH_CAH_05.12_eng.pdf?sequence=1&isAllowed=y)
- [US National Library of Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5718374/)
- [Science Daily Journals](https://www.sciencedaily.com/releases/2019/03/190320102041.htm#:~:text=The%20most%20common%20diagnoses%20were,of%20their%20abnormal%20skin%20findings.)
- [WHO neglected diseases](https://www.who.int/neglected_diseases/zoonoses/en/)
- [National Center for Biotechnology Information](https://pubmed.ncbi.nlm.nih.gov)
- [Journal of Mahatma Gandhi Institute of Medical Sciences](https://www.jmgims.co.in/article.asp?issn=0971-9903;year=2016;volume=21;issue=2;spage=111;epage=115;aulast=Jain)

## Draft paper publikasi (open/privat)

- (https://ejournal.bsi.ac.id/ejurnal/index.php/khatulistiwa/article/view/1253)
- (https://ieeexplore.ieee.org/document/8606798)
- (https://ieeexplore.ieee.org/document/4632102/)
- (https://ieeexplore.ieee.org/document/9580174)
- (https://ieeexplore.ieee.org/document/9972205)
- (https://ieeexplore.ieee.org/document/8474593)
- (https://ieeexplore.ieee.org/document/9580174)
- (https://ieeexplore.ieee.org/document/10051236)
- (https://ieeexplore.ieee.org/document/8973303)
