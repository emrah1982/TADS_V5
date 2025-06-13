from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import io
from PIL import Image
import time
import logging
import os
import sys

# YOLOv8 modelini import et
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics kütüphanesi bulunamadı. Yükleniyor...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# Logging ayarları
logger = logging.getLogger(__name__)

# Model yolu
MODEL_PATH = "models/plant_disease_model.pt"

# Sınıf isimleri
CLASS_NAMES = {
    0: "yaprak_lekesi",
    1: "mildiyö",
    2: "külleme",
    3: "pas",
    4: "sağlıklı"
}

# API için router
router = APIRouter()

# Model yükleme
model = None

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model dosyası bulunamadı: {MODEL_PATH}. Varsayılan YOLOv8n modeli kullanılacak.")
            model = YOLO("yolov8n.pt")
        else:
            model = YOLO(MODEL_PATH)
        logger.info("YOLOv8 modeli başarıyla yüklendi")
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        model = None

# Modeli yükle
load_model()

# Request modeli
class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image

# Görüntü işleme fonksiyonu
def process_image(base64_image):
    try:
        # Base64'ten görüntüyü decode et
        image_data = base64.b64decode(base64_image)
        
        # Numpy array'e dönüştür
        nparr = np.frombuffer(image_data, np.uint8)
        
        # OpenCV ile görüntüyü oku
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Görüntü decode edilemedi")
            
        # BGR'den RGB'ye dönüştür (YOLOv8 RGB bekler)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb, image.shape[1], image.shape[0]  # width, height
        
    except Exception as e:
        logger.error(f"Görüntü işleme hatası: {e}")
        raise HTTPException(status_code=400, detail=f"Görüntü işleme hatası: {str(e)}")

# Tespit endpoint'i
@router.post("/detect")
async def detect_plant_diseases(request: DetectionRequest):
    try:
        # Model kontrolü
        if model is None:
            load_model()
            if model is None:
                raise HTTPException(status_code=503, detail="Model yüklenemedi")
        
        # Görüntüyü işle
        image, original_width, original_height = process_image(request.image)
        
        # Başlangıç zamanı
        start_time = time.time()
        
        # YOLOv8 ile tespit
        results = model(image, conf=0.25)  # Confidence threshold
        
        # İşlem süresi
        process_time = time.time() - start_time
        
        # Sonuçları formatla
        predictions = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Koordinatlar
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Sınıf ve güven
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Sınıf adı
                class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                
                # Kutu genişliği ve yüksekliği
                width = x2 - x1
                height = y2 - y1
                
                predictions.append({
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(width),
                    "height": float(height),
                    "class": class_name,
                    "confidence": conf
                })
        
        return JSONResponse({
            "status": "success",
            "process_time_ms": round(process_time * 1000, 2),
            "predictions": predictions,
            "original_width": original_width,
            "original_height": original_height
        })
        
    except Exception as e:
        logger.error(f"Tespit hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Tespit hatası: {str(e)}")
