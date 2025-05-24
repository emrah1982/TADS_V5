# Multi-Model YOLO Detection System

Bu proje, **YOLO11** (genel nesneler) ve **Farm Model** (tarım analizi) olmak üzere iki farklı modeli ayrı ayrı veya aynı anda kullanarak çoklu video kaynaklarından gerçek zamanlı nesne tespiti yapan kapsamlı bir sistemdir.

## 🚀 Özellikler

### 🤖 Dual Model Architecture
- **YOLO11 (Genel Model)**: İnsan, araç, hayvan, günlük nesneler (80+ sınıf)
- **Farm Model**: Mineral eksikliği, hastalık, zararlı, olgunluk durumu vb. (özel tarım sınıfları)
- **Simultaneous Processing**: Her iki model aynı anda çalışır
- **Independent Control**: Modeller ayrı ayrı aktif/pasif edilebilir

### 📹 Video Kaynakları
- **Local Kamera**: USB/webcam desteği, çoklu kamera
- **DJI Drone**: UDP/RTMP stream, komut gönderme  
- **Parrot Anafi**: HTTP API, uçuş kontrolleri

### 🎯 Gelişmiş Tespit Özellikleri
- **Multi-Model Results**: Her kaynak için iki ayrı model sonucu
- **Real-time Processing**: Paralel model inference
- **Performance Tracking**: Model bazlı istatistikler
- **Configurable Thresholds**: Dinamik güven eşiği ayarı

### 🌐 Enhanced Web Interface
- **Model Control Panel**: Modelleri ayrı ayrı kontrol
- **Dual Detection Display**: Genel ve Farm tespitleri ayrı panellerde
- **Performance Monitoring**: Model bazlı FPS ve süre takibi
- **Interactive Controls**: Confidence slider, model toggle switches

## 📋 Model Dosyaları

Proje iki farklı model dosyası gerektirir:

```bash
# Genel YOLO modeli (otomatik indirilir)
yolo11.pt

# Farm modeli (manuel olarak eklenmelidir)
farm_best.pt
```

## 🛠️ Kurulum

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/your-repo/multi-model-yolo-detection.git
cd multi-model-yolo-detection
```

### 2. Python Ortamını Hazırlayın
```bash
# Virtual environment oluştur
python -m venv venv

# Activate et
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Gereksinimleri yükle
pip install -r requirements.txt
```

### 3. Model Dosyalarını Yerleştirin
```bash
# YOLO11 otomatik indirilecek
# farm_best.pt dosyasını proje kök dizinine koyun
cp /path/to/your/farm_best.pt ./
```

### 4. Dizin Yapısını Oluşturun
```bash
mkdir -p models logs recordings static/uploads
```

## 🚀 Çalıştırma

### Geliştirme Ortamı
```bash
# Direkt çalıştırma
python main.py

# Model dosya yollarını belirtme
MODEL_GENERAL=yolo11.pt MODEL_FARM=farm_best.pt python main.py
```

### Docker ile Çalıştırma
```bash
# Build (model dosyalarını kopyala)
docker build -t multi-model-yolo .

# Run
docker run -p 8000:8000 \
  --device /dev/video0:/dev/video0 \
  -v ./farm_best.pt:/app/farm_best.pt \
  multi-model-yolo
```

## 📱 Kullanım

### Web Arayüzü
1. Tarayıcıda `http://localhost:8000` adresine gidin
2. **Model Kontrolü**:
   - Genel Model ve Farm Model'i ayrı ayrı aktif/pasif edebilirsiniz
   - Güven eşiğini slider ile ayarlayabilirsiniz
3. **Video Kaynakları**:
   - Local Kamera, DJI Drone, Parrot Anafi'yi başlatın
4. **Tespit Sonuçları**:
   - Genel model tespitleri (mavi panel)
   - Farm model tespitleri (turuncu panel)
   - Gerçek zamanlı performance metrikleri

### API Kullanımı

#### Model Management
```bash
# Model bilgilerini görüntüle
curl "http://localhost:8000/api/models/info"

# Model istatistikleri
curl "http://localhost:8000/api/models/statistics"

# Genel modeli pasif et
curl -X POST "http://localhost:8000/api/models/general/toggle?enable=false"

# Farm modelini aktif et
curl -X POST "http://localhost:8000/api/models/farm/toggle?enable=true"

# Güven eşiğini ayarla
curl -X POST "http://localhost:8000/api/models/confidence" \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.7}'
```

#### Video Sources (Eski API'ler aynı)
```bash
# Local kamera başlat
curl -X POST "http://localhost:8000/api/local-camera/start" \
  -H "Content-Type: application/json" \
  -d '{"camera_id": 0}'
```

### WebSocket Multi-Model Data
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/local_camera');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'multi_model_detection') {
        console.log('General detections:', data.data.general_detections);
        console.log('Farm detections:', data.data.farm_detections);
        console.log('Processing times:', data.data.processing_times);
    }
};
```

## ⚙️ Konfigürasyon

### Environment Variables
```bash
# .env dosyası
MODEL_GENERAL=yolo11.pt
MODEL_FARM=farm_best.pt
CONFIDENCE_THRESHOLD=0.5
ENABLE_GENERAL=true
ENABLE_FARM=true
LOG_LEVEL=INFO
```

### Model Settings
```python
# Her iki modeli aktif et
multi_detector = MultiModelYOLODetector(
    general_model_path="yolo11.pt",
    farm_model_path="farm_best.pt",
    confidence_threshold=0.5,
    enable_general=True,
    enable_farm=True
)

# Sadece farm modeli
multi_detector = MultiModelYOLODetector(
    farm_model_path="farm_best.pt",
    enable_general=False,
    enable_farm=True
)
```

## 📊 Multi-Model Results

### Detection Result Structure
```json
{
  "frame_id": 123,
  "timestamp": 1640995200.123,
  "general_detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.85,
      "bbox": [100, 100, 200, 300],
      "center": [150, 200],
      "area": 20000,
      "model_type": "general"
    }
  ],
  "farm_detections": [
    {
      "class_id": 2,
      "class_name": "mineral_deficiency",
      "confidence": 0.92,
      "bbox": [150, 150, 250, 250],
      "center": [200, 200],
      "area": 10000,
      "model_type": "farm"
    }
  ],
  "processing_times": {
    "general": 0.025,
    "farm": 0.030,
    "total": 0.055
  },
  "detection_counts": {
    "general": 1,
    "farm": 1,
    "total": 2
  }
}
```

## 🎨 Visual Features

### Color Coding
- **Genel Model**: Mavi renk teması (#667eea)
- **Farm Model**: Turuncu renk teması (#ed8936)
- **Combined Stats**: Yeşil renk teması (#48bb78)

### Web Interface Sections
1. **Model Status Panel**: Her iki modelin durumu ve kontrolleri
2. **Confidence Control**: Tek slider ile her iki model için eşik ayarı
3. **Performance Overview**: Model bazlı FPS ve tespit sayıları
4. **Dual Detection Panels**: Ayrı panellerde genel ve farm tespitleri
5. **Video Streams**: Multi-model sonuçlarını gösteren video akışları

## 🔧 Geliştirme

### Proje Yapısı
```
├── main.py                           # Ana FastAPI uygulaması
├── models/
│   ├── multi_model_detector.py       # Multi-model YOLO wrapper
│   └── yolo_detector.py             # Legacy single model (opsiyonel)
├── video_sources/
│   ├── local_camera.py              # Local kamera (multi-model)
│   ├── dji_drone.py                 # DJI drone (multi-model)
│   └── parrot_anafi.py              # Parrot Anafi (multi-model)
├── utils/
│   ├── video_manager.py             # Multi-model video manager
│   └── websocket_manager.py         # WebSocket yöneticisi
├── static/
│   └── index.html                   # Enhanced web interface
├── yolo11.pt                        # Genel YOLO modeli
├── farm_best.pt                     # Farm modeli
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### Yeni Model Ekleme
```python
# models/multi_model_detector.py dosyasını güncelleyin

class ModelType(Enum):
    GENERAL = "general"
    FARM = "farm"
    NEW_MODEL = "new_model"  # Yeni model ekle

# Constructor'a yeni model parametresi ekleyin
def __init__(self, ..., new_model_path: str = None, enable_new_model: bool = True):
    # Yeni model yükleme kodu
```

## 📈 Performance Karşılaştırması

| Özellik | Tek Model | Multi-Model |
|---------|-----------|-------------|
| **Tespit Çeşitliliği** | Sınırlı | Çok Geniş |
| **İşleme Süresi** | ~25ms | ~55ms |
| **Bellek Kullanımı** | ~2GB | ~4GB |
| **Accuracy** | Model bazlı | Her alan için optimize |
| **Esneklik** | Düşük | Çok Yüksek |

## 🚨 Sorun Giderme

### Model Yükleme Sorunları
```bash
# Model dosyalarını kontrol et
ls -la *.pt

# Model yollarını kontrol et
python -c "
from models.multi_model_detector import MultiModelYOLODetector
detector = MultiModelYOLODetector()
print(detector.get_model_info())
"
```

### Memory Issues
```bash
# GPU memory temizle
python -c "import torch; torch.cuda.empty_cache()"

# Sadece bir model ile test et
ENABLE_GENERAL=true ENABLE_FARM=false python main.py
```

### Performance Optimization
```bash
# CPU-only mode
CUDA_VISIBLE_DEVICES="" python main.py

# Reduce confidence threshold for speed
CONFIDENCE_THRESHOLD=0.7 python main.py
```

## 🆕 Yenilikler

### v2.0 - Multi-Model Support
- ✅ Dual model architecture
- ✅ Independent model control
- ✅ Enhanced web interface
- ✅ Model-specific color coding
- ✅ Performance monitoring per model
- ✅ Confidence threshold control

### Gelecek Özellikler
- [ ] Model cascading (önce genel, sonra farm)
- [ ] Custom model training interface
- [ ] Model ensemble techniques
- [ ] Auto-model selection based on content
- [ ] Real-time model switching
- [ ] A/B testing between models

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🤝 Katkıda Bulunma

1. Bu projeyi fork'layın
2. Yeni model desteği ekleyin
3. Pull request gönderin

**Örnek katkılar:**
- Yeni domain-specific modeller
- Performance optimizasyonları
- UI/UX iyileştirmeleri
- Docker optimization

---

⭐ **Bu multi-model sistemi beğendiyseniz star vermeyi unutmayın!**

🔗 **Demo**: [Live Demo](http://your-demo-url.com)  
📧 **İletişim**: your-email@example.com  
📱 **Discord**: [Join our community](http://discord-link)