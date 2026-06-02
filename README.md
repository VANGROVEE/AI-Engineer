# VANGROVE AI - Plant Disease Classification API

## Deskripsi Singkat Proyek

VANGROVE AI merupakan sistem klasifikasi penyakit tanaman berbasis Artificial Intelligence menggunakan teknologi Deep Learning dan Transfer Learning. Sistem ini mampu mendeteksi berbagai penyakit daun pada tanaman jagung, mangga, kentang, dan tomat berdasarkan gambar daun yang diberikan pengguna.

Model dikembangkan menggunakan TensorFlow dengan arsitektur EfficientNetB0 serta teknik Fine Tuning untuk meningkatkan performa klasifikasi multi-class. Sistem juga dilengkapi integrasi Generative AI menggunakan Gemini API untuk memberikan penjelasan penyakit, penyebab, penanganan, pencegahan, dan kemungkinan pemulihan tanaman secara otomatis.

Fitur utama:
- Klasifikasi penyakit tanaman berbasis gambar daun
- Multi-class disease classification
- Confidence score prediction
- AI-generated disease explanation
- FastAPI REST API
- Docker deployment ready
- TensorBoard monitoring
- Explainable AI support
- Custom augmentation layer
- Unknown prediction handling

---

# Teknologi yang Digunakan

- Python
- TensorFlow
- EfficientNetB0
- FastAPI
- Docker
- TensorBoard
- Gemini API
- NumPy
- Pillow
- Railway Deployment

---

# Struktur Proyek

```bash
project/
│
├── api.py
├── inference.py
├── best_model_finetune.keras
├── class_names.pkl
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── .env.example
├── logs/
└── test_image/
```

---

# Setup Environment

## 1. Clone Repository

```bash
git clone https://github.com/VANGROVEE/AI-Engineer.git
```

```bash
cd <NAMA_FOLDER_PROJECT>
```

---

## 2. Membuat Virtual Environment

### Windows

```bash
python -m venv venv
```

Aktifkan environment:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependency

```bash
pip install -r requirements.txt
```

---

## 4. Setup Environment Variable

Buat file `.env`

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

---

# Template Environment

## File `.env.example`

```env
GEMINI_API_KEY=
```

---

# Dependensi

Dependency utama yang digunakan:

```txt
tensorflow==2.16.1
numpy==1.26.4
protobuf==4.25.3
pillow==10.3.0
fastapi
uvicorn
google-generativeai
python-multipart
python-dotenv
requests
pydantic
```

---

# Konfigurasi Pendukung

## `.gitignore`

```gitignore
venv/
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## `.dockerignore`

```dockerignore
venv
__pycache__
*.pyc
.ipynb_checkpoints
```

---

# Menjalankan Aplikasi

## Menjalankan Secara Lokal

```bash
python api.py
```

atau

```bash
uvicorn api:app --reload
```

API berjalan di:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Menjalankan Dengan Docker

## Build Docker Image

```bash
docker build -t vangrove-ai .
```

---

## Run Docker Container

```bash
docker run -p 8000:8000 --env-file .env vangrove-ai
```

---

# Endpoint API

## POST `/predict`

Digunakan untuk melakukan prediksi penyakit tanaman berdasarkan URL gambar.

### Request Body

```json
{
  "image_url": "IMAGE_URL",
  "top_k": 3,
  "explain": true
}
```

---

# Contoh Response

```json
{
  "nama_tanaman": "Tomat",
  "nama_penyakit": "Tomato Late Blight",
  "confidence": 0.91,

  "predictions": [
    {
      "nama_tanaman": "Tomat",
      "nama_penyakit": "Tomato Late Blight",
      "confidence": 0.91
    }
  ],

  "ai_explanation": "Penjelasan penyakit..."
}
```

---

# Monitoring TensorBoard

Jalankan:

```bash
tensorboard --logdir logs
```

Buka browser:

```text
http://localhost:6006
```

TensorBoard digunakan untuk memonitor:
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

---

# Model Machine Learning

Model Deep Learning dikembangkan menggunakan:
- EfficientNetB0
- Transfer Learning
- Fine Tuning
- Custom Layer Augmentation
- TensorBoard Logging
- Custom Training Loop menggunakan `tf.GradientTape`

---

# Komponen AI Lanjutan

Project ini mengimplementasikan beberapa advanced AI components:
- Custom Layer (`RandomBackground`)
- Custom Training Loop (`tf.GradientTape`)
- TensorBoard Monitoring
- Fine Tuning
- Transfer Learning
- Explainable AI
- Gemini AI Integration

---

# Deployment

Backend API dideploy menggunakan:
- Docker
- Railway

---

# Tautan Model ML

```txt
https://drive.google.com/drive/folders/1JHwyOMxRCy6K65Hm31MbgUautedTg_9i?usp=sharing
```

---

# Tautan Repository

```txt
https://github.com/VANGROVEE/AI-Engineer
```

---

# Catatan Tambahan

- Pastikan file `best_model_finetune.keras` dan `class_names.pkl` tersedia sebelum menjalankan API.
- Jika tidak menggunakan Gemini API, fitur AI explanation dapat dinonaktifkan.
- Unknown prediction handling digunakan ketika confidence model berada di bawah threshold.
- Sistem mendukung deployment berbasis Docker dan cloud deployment menggunakan Railway.

---

# Author

VANGROVE AI Team
