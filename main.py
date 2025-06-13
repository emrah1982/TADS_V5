import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List
import uvicorn

from models.multi_model_detector import MultiModelYOLODetector
from video_sources.video_manager import VideoManager
from utils.websocket_manager import WebSocketManager
from utils.config_manager import ConfigManager

# Logging ayarları
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global instances
detector = None
video_manager = None
websocket_manager = None
config_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    global detector, video_manager, websocket_manager, config_manager
    
    logger.info("FastAPI uygulaması başlatılıyor...")
    
    try:
        # Config manager'ı başlat
        config_manager = ConfigManager()
        
        # WebSocket manager'ı başlat
        websocket_manager = WebSocketManager()
        
        # YOLO detector'ı başlat (mevcut yapıya uygun)
        detector = MultiModelYOLODetector(
            general_model_path="models/yolov8n.pt",
            farm_model_path="models/farm_best.pt",
            zararli_model_path="models/zararliTespiti_best.pt",             
            domatesMineral_model_path="models/DomatesModels/domatesMineralTespiti_best.pt",
            domatesHastalik_model_path="models/DomatesModels/domatesHastalikTespiti_best.pt",
            domatesOlgunluk_model_path="models/DomatesModels/domatesOlgunlukTespiti_best.pt"
        )
        
        # Varsayılan olarak sadece genel ve çiftlik modellerini aktif et
        detector.enable_general = True
        detector.enable_farm = True
        detector.enable_zararli = True
        detector.enable_domatesMineral = False
        detector.enable_domatesHastalik = False
        detector.enable_domatesOlgunluk = False
        
        # initialize metodu yok, constructor'da yükleniyor
        # Model sınıflarını logla
        logger.info(f"Genel model sınıfları: {list(detector.general_class_names.values())}...")
        logger.info(f"Farm model sınıfları: {list(detector.farm_class_names.values())}...")
        logger.info(f"Zararlı model sınıfları: {list(detector.zararli_class_names.values()) if hasattr(detector, 'zararli_class_names') else 'Yüklenemedi'}...")
        
        logger.info("YOLO detector başarıyla yüklendi - Varsayılan modeller aktif")
               
        
        # Video manager'ı başlat
        video_manager = VideoManager(detector, websocket_manager)
        
        logger.info("Tüm bileşenler başarıyla başlatıldı")
        
        yield
        
    except Exception as e:
        logger.error(f"Başlatma hatası: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("Uygulama kapatılıyor...")
        
        if video_manager:
            await video_manager.cleanup()
        
        # detector cleanup metodu yok
        pass
        
        if websocket_manager:
            await websocket_manager.shutdown()
        
        logger.info("Uygulama başarıyla kapatıldı")

# FastAPI app
app = FastAPI(
    title="TADS V5 - Multi-Model Detection System",
    description="Tarımsal Anomali Tespit Sistemi",
    version="5.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/")
async def root():
    """Ana sayfa"""
    return {"message": "TADS V5 API", "status": "running"}

# Health check
@app.get("/health")
async def health_check():
    """Sistem sağlık kontrolü"""
    return {
        "status": "healthy",
        "components": {
            "detector": detector is not None,
            "video_manager": video_manager is not None,
            "websocket_manager": websocket_manager is not None
        }
    }

# Model info endpoint
@app.get("/api/models/info")
async def get_models_info():
    """Model bilgilerini al"""
    try:
        if not detector:
            raise HTTPException(status_code=503, detail="Detector hazır değil")
        
        model_info = detector.get_model_info()
        
        return {
            "status": "success",
            "models": model_info
        }
        
    except Exception as e:
        logger.error(f"Model bilgisi hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Güven eşiği endpoint'i
@app.post("/api/models/confidence")
async def update_confidence(confidence: float):
    """Güven eşiği değerini güncelle"""
    try:
        if not detector:
            raise HTTPException(status_code=503, detail="Detector hazır değil")
            
        if confidence < 0 or confidence > 1:
            raise HTTPException(status_code=400, detail="Güven eşiği 0-1 arasında olmalıdır")
        
        # Güven eşiğini güncelle
        detector.update_confidence_threshold(confidence)
        
        return {
            "status": "success",
            "message": f"Güven eşiği {confidence} olarak güncellendi"
        }
        
    except Exception as e:
        logger.error(f"Güven eşiği güncelleme hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model toggle endpoint
@app.post("/api/models/{model_name}/toggle")
async def toggle_model(model_name: str, enabled: bool):
    """Model durumunu değiştir"""
    try:
        if not detector:
            raise HTTPException(status_code=503, detail="Detector hazır değil")
        
        # Model durumunu değiştir
        success = detector.set_model_enabled(model_name, enabled)
        
        if success:
            return {
                "status": "success",
                "message": f"{model_name} modeli {'etkinleştirildi' if enabled else 'devre dışı bırakıldı'}"
            }
        else:
            raise HTTPException(status_code=400, detail="Model durumu değiştirilemedi")
            
    except Exception as e:
        logger.error(f"Model toggle hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Video source endpoints
@app.post("/api/sources/local-camera/start")
async def start_local_camera(camera_id: int = 0):
    """Local kamerayı başlat"""
    try:
        if not video_manager:
            raise HTTPException(status_code=503, detail="Video manager hazır değil")
        
        success = await video_manager.start_local_camera(camera_id)
        
        if success:
            # WebSocket'e bildirim gönder
            await websocket_manager.notify_source_started(f"local_{camera_id}")
            
            return {
                "status": "success",
                "message": f"Kamera {camera_id} başlatıldı",
                "source_id": f"local_{camera_id}"
            }
        else:
            raise HTTPException(status_code=500, detail="Kamera başlatılamadı")
            
    except Exception as e:
        logger.error(f"Kamera başlatma hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sources/{source_id}/stream")
async def stream_source(source_id: str):
    """Video stream endpoint"""
    stream = video_manager.get_stream(source_id)
    if not stream:
        return {"status": "error", "message": f"Kaynak {source_id} bulunamadı veya aktif değil"}
    
    return StreamingResponse(
        stream,
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/sources/{source_id}/stop")
async def stop_source(source_id: str):
    """Video kaynağını durdur"""
    try:
        if not video_manager:
            raise HTTPException(status_code=503, detail="Video manager hazır değil")
        
        success = await video_manager.stop_source(source_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Kaynak {source_id} durduruldu"
            }
        else:
            raise HTTPException(status_code=404, detail="Kaynak bulunamadı")
            
    except Exception as e:
        logger.error(f"Kaynak durdurma hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sources")
async def get_sources():
    """Aktif kaynakları listele"""
    try:
        if not video_manager:
            raise HTTPException(status_code=503, detail="Video manager hazır değil")
        
        sources = video_manager.get_all_sources_info()
        return {
            "status": "success",
            "sources": sources,
            "active_count": video_manager.get_source_count()
        }
        
    except Exception as e:
        logger.error(f"Kaynak listeleme hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Duplicate endpoint kaldırıldı - yukarıdaki stream_source endpoint'i kullanılacak

@app.get("/api/sources/{source_id}/detections")
async def get_detections(source_id: str):
    """Son tespit sonuçlarını al"""
    try:
        if not video_manager:
            raise HTTPException(status_code=503, detail="Video manager hazır değil")
        
        detections = video_manager.get_latest_detections(source_id)
        
        if detections:
            return {
                "status": "success",
                "data": detections
            }
        else:
            return {
                "status": "success",
                "data": None,
                "message": "Henüz tespit yok"
            }
            
    except Exception as e:
        logger.error(f"Detection alma hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detections/with-gps/{source_id}")
async def get_detections_with_gps(source_id: str):
    """Son tespit sonuçlarını GPS koordinatları ve kategorileri ile birlikte al"""
    try:
        if not video_manager:
            raise HTTPException(status_code=503, detail="Video manager hazır değil")
            
        if not config_manager:
            raise HTTPException(status_code=503, detail="Config manager hazır değil")
        
        detections = video_manager.get_latest_detections(source_id)
        
        if not detections:
            return {
                "status": "success",
                "data": None,
                "message": "Henüz tespit yok"
            }

        # Tüm kategorileri config'den al
        categories = config_manager.get_detection_categories()
        
        # Tespit edilen sınıfların kategori eşleştirmelerini yap
        class_to_category = {}
        for category_name, items in categories.items():
            for item in items:
                class_to_category[item["name"]] = {
                    "main_category": category_name,
                    "display_name": item["display_name"]
                }

        # Tespit kategorilerine göre GPS koordinatları oluştur
        gps_detections = []
        for detection in detections:
            class_name = detection.get("class", "").lower()
            category_info = class_to_category.get(class_name)
            
            if category_info:
                # Türkiye sınırları içinde random GPS koordinatları
                gps_data = {
                    "latitude": random.uniform(36.0, 42.0),  # Türkiye'nin enlem aralığı
                    "longitude": random.uniform(26.0, 45.0),  # Türkiye'nin boylam aralığı
                    "detection": detection,
                    "main_category": category_info["main_category"],
                    "display_name": category_info["display_name"],
                    "severity": detection.get("confidence", 0) * 100  # Tespit şiddeti yüzdesi
                }
                
                # Olgunluk durumu için ek bilgiler
                if category_info["main_category"] == "ripeness":
                    # Model konfigürasyonundan olgunluk bilgilerini al
                    class_config = config_manager.get_class_config("farm", int(detection.get("class_id", 0)))
                    if class_config and hasattr(class_config, "ripeness_info"):
                        ripeness_info = class_config.ripeness_info
                        gps_data.update({
                            "harvest_info": {
                                "days_until_harvest": ripeness_info.get("harvest_time", 0),
                                "estimated_harvest_date": (datetime.now() + timedelta(days=ripeness_info.get("harvest_time", 0))).strftime("%Y-%m-%d"),
                            },
                            "color_analysis": {
                                "ranges": ripeness_info.get("color_ranges", {}),
                                "ripeness_percentage": ripeness_info.get("ripeness_percentage", 0)
                            }
                        })
                
                gps_detections.append(gps_data)
        
        return {
            "status": "success",
            "data": gps_detections
        }
            
    except Exception as e:
        logger.error(f"GPS ile detection alma hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """Sistem istatistiklerini al"""
    try:
        stats = {
            "video_manager": video_manager.get_global_statistics() if video_manager else {},
            "websocket": websocket_manager.get_statistics() if websocket_manager else {},
            "detector": detector.get_statistics() if detector and hasattr(detector, 'get_statistics') else {}
        }
        
        return {
            "status": "success",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"İstatistik hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{source_type}")
async def websocket_endpoint(websocket: WebSocket, source_type: str):
    """WebSocket bağlantı endpoint'i"""
    connection = None
    
    try:
        # Bağlantıyı kabul et
        connection = await websocket_manager.connect(websocket, source_type)
        logger.info(f"WebSocket bağlantısı kuruldu: {connection.client_id}")
        
        # Mesajları dinle
        while True:
            try:
                data = await websocket.receive_text()
                
                # Ping-pong mesajları
                if data == "ping":
                    await connection.send("pong")
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket bağlantısı koptu: {connection.client_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket mesaj hatası: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket hatası: {e}")
    
    finally:
        # Bağlantıyı temizle
        if connection and websocket_manager:
            websocket_manager.disconnect(websocket, source_type)

if __name__ == "__main__":
    # Log dizinini oluştur
    os.makedirs("logs", exist_ok=True)
    
    # Uvicorn ile çalıştır
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )