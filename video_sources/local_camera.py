import cv2
import asyncio
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np
from queue import Queue, Empty

from models.multi_model_detector import MultiModelYOLODetector, MultiModelDetectionResult

logger = logging.getLogger(__name__)

class LocalCameraSource:
    """Local kamera video kaynağı sınıfı"""
    
    def __init__(self, detector: MultiModelYOLODetector, camera_id: int = 0):
        """
        Local kamera kaynağını başlat
        
        Args:
            detector: YOLO detector instance
            camera_id: Kamera ID (genellikle 0)
        """
        self.detector = detector
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.thread = None
        
        # Frame queue
        self.frame_queue = Queue(maxsize=10)
        self.latest_frame = None
        self.latest_detections = None
        
        # Callbacks
        self.frame_callback = None
        self.detection_callback = None
        
        # Configuration
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
        self.frame_id = 0
        
        # Statistics
        self.frames_captured = 0
        self.frames_processed = 0
        self.start_time = time.time()
        
        # Threading
        self._lock = threading.Lock()
    
    async def initialize(self) -> bool:
        """Kamerayı başlat ve ayarla"""
        try:
            logger.info(f"Local kamera başlatılıyor (ID: {self.camera_id})")
            
            # Kamerayı aç
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Kamera açılamadı (ID: {self.camera_id})")
                return False
            
            # Kamera ayarları
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Buffer size'ı küçült (düşük latency için)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Gerçek kamera özelliklerini al
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Kamera başlatıldı: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
            
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            return False
    
    def start_capture(self):
        """Video yakalama işlemini başlat"""
        if self.is_running:
            logger.warning("Kamera zaten çalışıyor")
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_id = 0
        
        # Thread'i başlat
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info("Kamera yakalama başlatıldı")
    
    def stop_capture(self):
        """Video yakalama işlemini durdur"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Thread'in bitmesini bekle
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        # Kamerayı kapat
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Kamera yakalama durduruldu")
    
    def _capture_loop(self):
        """Ana yakalama döngüsü"""
        logger.info("Kamera yakalama döngüsü başladı")
        
        while self.is_running:
            try:
                # Frame yakala
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Frame yakalanamadı")
                    break
                
                with self._lock:
                    self.frames_captured += 1
                    current_frame_id = self.frame_id
                    self.frame_id += 1
                
                # Frame'i queue'ya ekle
                try:
                    self.frame_queue.put_nowait((current_frame_id, frame.copy()))
                except:
                    # Queue dolu, eski frame'i at
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((current_frame_id, frame.copy()))
                    except Empty:
                        pass
                
                # Latest frame'i güncelle
                self.latest_frame = frame.copy()
                
                # Detection işlemi (sync olarak thread içinde)
                self._process_frame(current_frame_id, frame)
                
                # FPS kontrolü
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                logger.error(f"Frame generator hatası: {e}")
                continue
    
    def set_frame_callback(self, callback: Callable):
        """Frame callback fonksiyonunu ayarla"""
        self.frame_callback = callback
    
    def set_detection_callback(self, callback: Callable):
        """Detection callback fonksiyonunu ayarla"""
        self.detection_callback = callback
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Kamera bilgilerini döndür"""
        if not self.cap:
            return {}
        
        return {
            "camera_id": self.camera_id,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "backend": self.cap.getBackendName() if hasattr(self.cap, 'getBackendName') else "Unknown"
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        elapsed_time = time.time() - self.start_time
        
        with self._lock:
            capture_fps = self.frames_captured / elapsed_time if elapsed_time > 0 else 0
            process_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
            
            return {
                "source_type": "local_camera",
                "camera_id": self.camera_id,
                "is_running": self.is_running,
                "frames_captured": self.frames_captured,
                "frames_processed": self.frames_processed,
                "capture_fps": capture_fps,
                "process_fps": process_fps,
                "elapsed_time": elapsed_time
            }
    
    def set_resolution(self, width: int, height: int):
        """Çözünürlük ayarla"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Çözünürlük ayarlandı: {self.frame_width}x{self.frame_height}")
    
    def set_fps(self, fps: int):
        """FPS ayarla"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.fps = fps
            logger.info(f"FPS ayarlandı: {fps}")
    
    def is_camera_available(self) -> bool:
        """Kameranın kullanılabilir olup olmadığını kontrol et"""
        return self.cap is not None and self.cap.isOpened()
    
    async def cleanup(self):
        """Temizlik işlemleri"""
        logger.info("Local kamera temizleniyor")
        self.stop_capture()
        
        # Queue'yu temizle
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        logger.info("Local kamera temizlendi")
    
    def _process_frame(self, frame_id: int, frame: np.ndarray):
        """Frame'i işle ve tespit yap"""
        try:
            # YOLO detection
            detection_result = self.detector.detect(frame, frame_id)
            
            with self._lock:
                self.frames_processed += 1
                self.latest_detections = detection_result
            
            # Callback çağır
            if self.detection_callback:
                self.detection_callback(detection_result)
            
            # Annotated frame oluştur
            if detection_result.all_detections:
                annotated_frame = self.detector.draw_detections(frame, detection_result)
            else:
                annotated_frame = frame
            
            # Frame callback çağır
            if self.frame_callback:
                self.frame_callback(annotated_frame, detection_result)
                
        except Exception as e:
            logger.error(f"Frame işleme hatası: {e}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """En son frame'i döndür"""
        return self.latest_frame
    
    def get_latest_detections(self) -> Optional[MultiModelDetectionResult]:
        """En son tespit sonuçlarını döndür"""
        return self.latest_detections
    
    def get_frame_generator(self):
        """Frame generator (streaming için)"""
        while self.is_running:
            try:
                frame_id, frame = self.frame_queue.get(timeout=1.0)
                
                # Detection yap
                detection_result = self.detector.detect(frame, frame_id)
                
                # Annotated frame oluştur
                if detection_result.all_detections:
                    annotated_frame = self.detector.draw_detections(frame, detection_result)
                else:
                    annotated_frame = frame
                
                # JPEG encode
                _, buffer = cv2.imencode('.jpg', annotated_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Frame generator hatası: {e}")
                break

class LocalCameraManager:
    """Multiple local camera yönetimi için"""
    
    def __init__(self, detector: MultiModelYOLODetector):
        self.detector = detector
        self.cameras: Dict[int, LocalCameraSource] = {}
        self._lock = threading.Lock()
    
    async def add_camera(self, camera_id: int) -> bool:
        """Yeni kamera ekle"""
        with self._lock:
            if camera_id in self.cameras:
                logger.warning(f"Kamera {camera_id} zaten mevcut")
                return False
            
            camera = LocalCameraSource(self.detector, camera_id)
            success = await camera.initialize()
            
            if success:
                self.cameras[camera_id] = camera
                logger.info(f"Kamera {camera_id} başarıyla eklendi")
                return True
            else:
                logger.error(f"Kamera {camera_id} eklenemedi")
                return False
    
    async def remove_camera(self, camera_id: int) -> bool:
        """Kamerayı kaldır"""
        with self._lock:
            if camera_id not in self.cameras:
                logger.warning(f"Kamera {camera_id} bulunamadı")
                return False
            
            camera = self.cameras[camera_id]
            await camera.cleanup()
            del self.cameras[camera_id]
            
            logger.info(f"Kamera {camera_id} kaldırıldı")
            return True
    
    def get_camera(self, camera_id: int) -> Optional[LocalCameraSource]:
        """Kamera instance'ını döndür"""
        return self.cameras.get(camera_id)
    
    def get_all_cameras(self) -> Dict[int, LocalCameraSource]:
        """Tüm kameraları döndür"""
        return self.cameras.copy()
    
    async def start_all_cameras(self):
        """Tüm kameraları başlat"""
        for camera_id, camera in self.cameras.items():
            try:
                camera.start_capture()
                logger.info(f"Kamera {camera_id} başlatıldı")
            except Exception as e:
                logger.error(f"Kamera {camera_id} başlatma hatası: {e}")
    
    async def stop_all_cameras(self):
        """Tüm kameraları durdur"""
        for camera_id, camera in self.cameras.items():
            try:
                camera.stop_capture()
                logger.info(f"Kamera {camera_id} durduruldu")
            except Exception as e:
                logger.error(f"Kamera {camera_id} durdurma hatası: {e}")
    
    async def cleanup_all(self):
        """Tüm kameraları temizle"""
        camera_ids = list(self.cameras.keys())
        for camera_id in camera_ids:
            await self.remove_camera(camera_id)
    
    def get_available_cameras(self) -> List[int]:
        """Kullanılabilir kamera ID'lerini döndür"""
        available = []
        
        try:
            for i in range(10):  # 0-9 arası kamera ID'lerini test et
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
            
            return available
        except Exception as e:
            logger.error(f"Kamera tespit hatası: {e}")
            return []  # Hata durumunda boş liste döndür