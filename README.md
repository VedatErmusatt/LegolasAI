# 🧝‍♂️ LegolasAI

LegolasAI, PDF dokümanları üzerinden anlamlı sohbetler yapabilen, web'de arama yapabilen, hava durumu ve hesaplama araçlarına sahip çok yönlü bir yapay zeka asistanıdır. LangChain ve Google Gemini LLM teknolojisiyle geliştirilmiştir.

---

## 🚀 Özellikler

- 📄 Yüklenen PDF içeriğinden anlamlı sohbet yapabilir
- 🌐 Web arama (DuckDuckGo üzerinden)
- 🌦️ Anlık hava durumu sorguları
- ➗ Basit matematiksel hesaplamalar
- 🔁 Otomatik Türkçe ⇄ İngilizce çeviri
- 🧠 Sohbet geçmişi ve bellek destekli yanıtlar

---

## 🛠️ Kullanılan Teknolojiler

- Python 3.12+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- Google Gemini (via `langchain-google-genai`)
- FAISS vektör veri tabanı
- DuckDuckGo Search API
- OpenWeatherMap API
- Deep Translator (Google Çeviri)

---

## 📦 Kurulum

1. Bu repoyu klonla:

```bash
git clone https://github.com/VedatErmusatt/LegolasAI.git
cd LegolasAI
```

2. Sanal ortam oluştur ve etkinleştir:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Gerekli paketleri yükle:

```bash
pip install -r requirements.txt
```

4. `.env` dosyası oluştur:

`.env` dosyasının içeriği şu şekilde olmalıdır:

```env
GOOGLE_API_KEY=your_google_api_key
WEATHER_API_KEY=your_openweathermap_api_key
```

---

## ▶️ Uygulamayı Başlat

```bash
streamlit run app.py
```

> `app.py` dosyasının adı sende farklıysa, kendi dosya adını kullan.

---

## 📌 Notlar

- Hava durumu özelliği için [OpenWeatherMap](https://openweathermap.org/api)'ten API key alınmalıdır.
- PDF yüklemek isteğe bağlıdır. Bot, PDF olmadan da genel soruları yanıtlar.
- Çeviri özelliği varsayılan olarak açıktır, sidebar'dan kapatılabilir.

---

## 📄 Lisans

MIT Lisansı © 2025 Vedat Ermusatt
