\# 🎨 Almanca Çizim Oyunu



Google'ın Quick Draw oyunundan esinlenen interaktif Almanca öğrenme oyunu. Çocuklar çizerek Almanca kelimeler öğrenir!



\## 🎮 Oyun Nasıl Çalışır?



1\. \*\*🤖 Robot 1\*\*: Size bir Almanca kelime verir

2\. \*\*👦 Çocuk\*\*: Bu kelimeyi canvas üzerine çizer  

3\. \*\*🤖 Robot 2\*\*: Çizimi AI ile tahmin etmeye çalışır

4\. \*\*🏆 Puan\*\*: Doğru tahmin edilirse puan kazanırsınız!



\## 🛠️ Kurulum



\### Gereksinimler

\- Python 3.7+ (mevcut: Python 3.7.3)

\- pip package manager

\- `almanca\_cizim\_modeli.h5` model dosyası



\### Adımlar



1\. \*\*Proje klasörüne gidin:\*\*

```bash

cd C:\\projects\\draw4german\_game

```



2\. \*\*Kurulum scriptini çalıştırın:\*\*

```bash

python setup.py

```



3\. \*\*Uygulamayı başlatın:\*\*

```bash

streamlit run main.py

```



\### Manuel Kurulum



Eğer kurulum scripti çalışmazsa:



```bash

pip install --upgrade pip

pip install -r requirements.txt

streamlit run main.py

```



\## 📁 Proje Yapısı



```

draw4german\_game/

├── main.py                    # Ana Streamlit uygulaması

├── requirements.txt           # Python paket gereksinimleri

├── setup.py                  # Kurulum scripti

├── README.md                 # Bu dosya

├── almanca\_cizim\_modeli.h5   # TensorFlow modeli (egitilmis)

├── data.zip                  # Eğitim verisi (opsiyonel)

└── filelist.txt              # Model sınıf listesi

```



\## 🎯 Özellikler



\- \*\*116 farklı nesne\*\* çizimi ve tahmini

\- \*\*Interaktif çizim canvas'ı\*\*

\- \*\*Real-time AI tahmini\*\*

\- \*\*Almanca-İngilizce çeviri\*\*

\- \*\*Skor sistemi\*\*

\- \*\*Çocuk dostu arayüz\*\*



\## 🧠 Desteklenen Objeler



Oyun 116 farklı nesneyi destekler:

\- Hayvanlar: Kedi (die Katze), Köpek (der Hund), Balık (der Fisch)...

\- Yiyecekler: Elma (der Apfel), Muz (die Banane), Pizza (die Pizza)...

\- Nesneler: Araba (das Auto), Ev (das Haus), Sandalye (der Stuhl)...

\- Ve daha fazlası!



\## 🔧 Teknik Detaylar



\### Kullanılan Teknolojiler

\- \*\*Streamlit\*\*: Web arayüzü

\- \*\*TensorFlow\*\*: AI model (v2.13.1)

\- \*\*OpenCV\*\*: Görüntü işleme

\- \*\*PIL\*\*: Görüntü manipülasyonu

\- \*\*NumPy\*\*: Numerical işlemler



\### Model Bilgileri

\- \*\*Girdi\*\*: 28x28 grayscale görüntü

\- \*\*Çıktı\*\*: 116 sınıf tahmini

\- \*\*Eğitim\*\*: TensorFlow 2.19 ile Google Colab'da eğitildi

\- \*\*Format\*\*: Keras H5 modeli



\## 🐛 Sorun Giderme



\### Model Yüklenmiyor

```

❌ Model yüklenirken hata: ...

```

\*\*Çözüm\*\*: `almanca\_cizim\_modeli.h5` dosyasının proje klasöründe olduğundan emin olun.



\### Paket Yükleme Hatası

```

❌ Paket yükleme hatası: ...

```

\*\*Çözüm\*\*: 

```bash

pip install --upgrade pip

pip install tensorflow==2.13.1 streamlit==1.28.1

```



\### Canvas Çalışmiyor

\*\*Çözüm\*\*: Tarayıcınızı yenileyin veya farklı bir tarayıcı deneyin.



\## 📱 Deployment



\### Streamlit Cloud'a Deploy

1\. GitHub'a push edin

2\. \[share.streamlit.io](https://share.streamlit.io) 'ya gidin  

3\. Repository'nizi bağlayın

4\. `main.py` dosyasını seçin



\### Lokal Deployment

```bash

streamlit run main.py --server.port 8501

```



\## 🤝 Katkıda Bulunun



1\. Fork edin

2\. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)

3\. Commit edin (`git commit -m 'Add amazing feature'`)

4\. Push edin (`git push origin feature/amazing-feature`)

5\. Pull Request açın



\## 📄 Lisans



Bu proje eğitim amaçlıdır ve açık kaynak olarak paylaşılmıştır.



\## 👨‍💻 Geliştirici Notları



\### Python Sürüm Uyumluluğu

\- \*\*Minimum\*\*: Python 3.7

\- \*\*Test edildi\*\*: Python 3.7.3

\- \*\*Önerilen\*\*: Python 3.8+



\### Versiyon Notları

\- TensorFlow 2.13.1: Python 3.7 ile uyumlu son sürüm

\- Streamlit 1.28.1: Kararlı sürüm

\- NumPy 1.21.6: Python 3.7 ile uyumlu



---



\*\*🎓 Bu oyunla Almanca öğrenmek artık daha eğlenceli!\*\*

