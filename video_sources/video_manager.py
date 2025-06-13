import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
import json

from models.multi_model_detector import MultiModelYOLODetector, MultiModelDetectionResult
from video_sources.local_camera import LocalCameraSource
from video_sources.dji_drone import DJIDroneSource
from video_sources.parrot_anafi import ParrotAnafiSource
from utils.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

class VideoManager:
    """Video kaynaklarını yöneten ana sınıf"""
    
    def __init__(self, detector: MultiModelYOLODetector, websocket_manager: WebSocketManager):
        
        """
        Video Manager'ı başlat
        
        Args:
            detector: Multi-model YOLO detector instance
            websocket_manager: WebSocket bağlantı yöneticisi
        """
        
        self.detector = detector
        self.websocket_manager = websocket_manager
        
        # Video kaynakları
        self.sources: Dict[str, Any] = {}
        self.active_sources: Dict[str, bool] = {}
        
        # Detection storage
        self.latest_detections: Dict[str, MultiModelDetectionResult] = {}
        self.detection_history: Dict[str, List[MultiModelDetectionResult]] = {}
        
        # Threading
        self._lock = threading.Lock()
        
        # Statistics
        self.total_frames_processed = 0
        self.start_time = time.time()
        
        # Configuration
        self.max_history_length = 100
        self.auto_cleanup_interval = 300  # 5 dakika
        
        # Background tasks
        self.cleanup_task = None
        self.stats_task = None
        
        # Main event loop'u sakla
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = None
            logger.warning("Main event loop bulunamadı - WebSocket bildirimleri devre dışı olabilir")
    
    async def start_local_camera(self, camera_id: int = 0) -> bool:
        """Local kamerayı başlat"""
        source_id = f"local_{camera_id}"
        
        try:
            with self._lock:
                if source_id in self.sources and self.active_sources.get(source_id, False):
                    logger.warning(f"Kamera {camera_id} zaten aktif")
                    return True
            
            logger.info(f"Kamera başlatılıyor (ID: {camera_id})")
            
            # Source oluştur
            camera_source = LocalCameraSource(self.detector, camera_id)
            
            # Initialize et
            success = await camera_source.initialize()
            if not success:
                raise Exception(f"Kamera {camera_id} başlatılamadı")
            
            # Callbacks ayarla - sync wrapper kullan
            def detection_callback(detection):
                try:
                    # Sync olarak detection'ı işle
                    self._on_detection_received_sync(source_id, detection)
                except Exception as e:
                    logger.error(f"Detection callback hatası: {e}")
            
            def frame_callback(frame, detection):
                try:
                    # Sync olarak frame'i işle
                    self._on_frame_received_sync(source_id, frame, detection)
                except Exception as e:
                    logger.error(f"Frame callback hatası: {e}")
            
            camera_source.set_detection_callback(detection_callback)
            camera_source.set_frame_callback(frame_callback)
            
            # Source'u kaydet
            with self._lock:
                self.sources[source_id] = camera_source
                self.active_sources[source_id] = True
                self.detection_history[source_id] = []
                self.latest_detections[source_id] = None
            
            # Capture'u başlat
            camera_source.start_capture()
            
            logger.info(f"Kamera {camera_id} başarıyla başlatıldı")
            
            # Background tasks'i başlat
            await self._start_background_tasks()
            
            return True
            
        except Exception as e:
            logger.error(f"Kamera {camera_id} başlatma hatası: {e}")
            return False
    
    async def start_dji_drone(self, connection_string: str = "udp:11111") -> str:
        """DJI drone'u başlat"""
        source_id = "dji_drone"
        
        try:
            with self._lock:
                if source_id in self.sources and self.active_sources.get(source_id, False):
                    logger.warning("DJI drone zaten aktif")
                    return source_id
            
            logger.info(f"DJI drone başlatılıyor: {connection_string}")
            
            # Source oluştur
            drone_source = DJIDroneSource(self.detector)
            
            # Initialize et
            success = await drone_source.initialize(connection_string)
            if not success:
                raise Exception("DJI drone başlatılamadı")
            
            # Callbacks ayarla - sync wrapper kullan
            def detection_callback(detection):
                try:
                    self._on_detection_received_sync(source_id, detection)
                except Exception as e:
                    logger.error(f"DJI detection callback hatası: {e}")
            
            def frame_callback(frame, detection):
                try:
                    self._on_frame_received_sync(source_id, frame, detection)
                except Exception as e:
                    logger.error(f"DJI frame callback hatası: {e}")
            
            drone_source.set_detection_callback(detection_callback)
            drone_source.set_frame_callback(frame_callback)
            
            # Source'u kaydet
            with self._lock:
                self.sources[source_id] = drone_source
                self.active_sources[source_id] = True
                self.detection_history[source_id] = []
            
            # Capture'u başlat
            drone_source.start_capture()
            
            logger.info("DJI drone başarıyla başlatıldı")
            
            # Background tasks'i başlat
            await self._start_background_tasks()
            
            return source_id
            
        except Exception as e:
            logger.error(f"DJI drone başlatma hatası: {e}")
            raise
    
    async def start_parrot_anafi(self, drone_ip: str = "192.168.42.1") -> str:
        """Parrot Anafi'yi başlat"""
        source_id = "parrot_anafi"
        
        try:
            with self._lock:
                if source_id in self.sources and self.active_sources.get(source_id, False):
                    logger.warning("Parrot Anafi zaten aktif")
                    return source_id
            
            logger.info(f"Parrot Anafi başlatılıyor: {drone_ip}")
            
            # Source oluştur
            anafi_source = ParrotAnafiSource(self.detector)
            
            # Initialize et
            success = await anafi_source.initialize(drone_ip)
            if not success:
                raise Exception("Parrot Anafi başlatılamadı")
            
            # Callbacks ayarla - sync wrapper kullan
            def detection_callback(detection):
                try:
                    self._on_detection_received_sync(source_id, detection)
                except Exception as e:
                    logger.error(f"Anafi detection callback hatası: {e}")
            
            def frame_callback(frame, detection):
                try:
                    self._on_frame_received_sync(source_id, frame, detection)
                except Exception as e:
                    logger.error(f"Anafi frame callback hatası: {e}")
            
            anafi_source.set_detection_callback(detection_callback)
            anafi_source.set_frame_callback(frame_callback)
            
            # Source'u kaydet
            with self._lock:
                self.sources[source_id] = anafi_source
                self.active_sources[source_id] = True
                self.detection_history[source_id] = []
            
            # Capture'u başlat
            anafi_source.start_capture()
            
            logger.info("Parrot Anafi başarıyla başlatıldı")
            
            # Background tasks'i başlat
            await self._start_background_tasks()
            
            return source_id
            
        except Exception as e:
            logger.error(f"Parrot Anafi başlatma hatası: {e}")
            raise
    
    async def stop_source(self, source_id: str) -> bool:
        """Belirtilen kaynağı durdur"""
        try:
            with self._lock:
                if source_id not in self.sources:
                    logger.warning(f"Kaynak bulunamadı: {source_id}")
                    return False
                
                if not self.active_sources.get(source_id, False):
                    logger.warning(f"Kaynak zaten durmuş: {source_id}")
                    return True
            
            logger.info(f"Kaynak durduruluyor: {source_id}")
            
            source = self.sources[source_id]
            
            # Source'u durdur
            if hasattr(source, 'stop_capture'):
                source.stop_capture()
            
            # Cleanup
            if hasattr(source, 'cleanup'):
                await source.cleanup()
            
            # Source'u kaldır
            with self._lock:
                self.active_sources[source_id] = False
                if source_id in self.detection_history:
                    del self.detection_history[source_id]
                if source_id in self.latest_detections:
                    del self.latest_detections[source_id]
                if source_id in self.sources:
                    del self.sources[source_id]
            
            logger.info(f"Kaynak başarıyla durduruldu: {source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Kaynak durdurma hatası ({source_id}): {e}")
            return False
    
    def get_stream(self, source_id: str):
        """Belirtilen kaynağın stream'ini döndür"""
        with self._lock:
            if source_id not in self.sources or not self.active_sources.get(source_id, False):
                return None
            
            source = self.sources[source_id]
            
            if hasattr(source, 'get_frame_generator'):
                return source.get_frame_generator()
            
            return None
            
    def get_latest_frame(self, source_id: str) -> Optional[Any]:
        """En son frame'i döndür"""
        try:
            with self._lock:
                if source_id not in self.sources or not self.active_sources.get(source_id, False):
                    return None
                    
                source = self.sources[source_id]
                if hasattr(source, 'get_latest_frame'):
                    frame = source.get_latest_frame()
                    if frame is not None:
                        # Son tespitleri al ve frame üzerine çiz
                        detections = self.latest_detections.get(source_id)
                        if detections and detections.all_detections:
                            frame = self.detector.draw_detections(frame.copy(), detections.all_detections)
                        return frame
            return None
        except Exception as e:
            logger.error(f"Frame alma hatası ({source_id}): {e}")
            return None
    
    def get_latest_detections(self, source_id: str) -> Optional[Dict]:
        """En son tespit sonuçlarını döndür"""
        with self._lock:
            if source_id not in self.latest_detections:
                return None
            
            detection_result = self.latest_detections[source_id]
            
            if detection_result is None:
                return None
            
            # Detection'ları serialize et
            detections = []
            for detection in detection_result.all_detections:
                detections.append({
                    'class_id': int(detection.class_id),
                    'class_name': str(detection.class_name),
                    'confidence': float(detection.confidence),
                    'bbox': [float(x) for x in detection.bbox],
                    'center': [float(x) for x in detection.center],
                    'area': float(detection.area)
                })
            
            return {
                'frame_id': int(detection_result.frame_id),
                'timestamp': float(detection_result.timestamp),
                'detections': detections,
                'processing_time': float(detection_result.processing_time) if hasattr(detection_result, 'processing_time') else 0.0,
                'frame_size': [int(x) for x in detection_result.frame_size] if detection_result.frame_size else [0, 0]
            }
    
    def get_active_sources(self) -> List[str]:
        """Aktif kaynakların listesini döndür"""
        with self._lock:
            return [source_id for source_id, active in self.active_sources.items() if active]
    
    def get_source_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Kaynak bilgilerini döndür"""
        with self._lock:
            if source_id not in self.sources:
                return None
            
            source = self.sources[source_id]
            
            # Temel bilgiler
            info = {
                'source_id': source_id,
                'is_active': self.active_sources.get(source_id, False),
                'type': type(source).__name__
            }
            
            # Source-specific bilgiler
            if hasattr(source, 'get_statistics'):
                info['statistics'] = source.get_statistics()
            
            if hasattr(source, 'get_drone_info'):
                info['drone_info'] = source.get_drone_info()
            elif hasattr(source, 'get_camera_info'):
                info['camera_info'] = source.get_camera_info()
            
            return info
    
    def get_all_sources_info(self) -> Dict[str, Dict]:
        """Tüm kaynakların bilgilerini döndür"""
        sources_info = {}
        
        for source_id in self.sources.keys():
            sources_info[source_id] = self.get_source_info(source_id)
        
        return sources_info
    
    async def _on_detection_received(self, source_id: str, detection_result: MultiModelDetectionResult) -> None:
        """Detection alındığında çağrılır"""
        try:
            with self._lock:
                # Latest detection'ı güncelle
                self.latest_detections[source_id] = detection_result
                
                # History'e ekle
                if source_id not in self.detection_history:
                    self.detection_history[source_id] = []
                
                self.detection_history[source_id].append(detection_result)
                
                # History limitini kontrol et
                if len(self.detection_history[source_id]) > self.max_history_length:
                    self.detection_history[source_id] = self.detection_history[source_id][-self.max_history_length:]
                
                # Statistics güncelle
                self.total_frames_processed += 1
            
            # WebSocket üzerinden gönder (sadece detection varsa)
            if detection_result and detection_result.detections:
                await self._send_detection_to_websocket(source_id, detection_result)
            
        except Exception as e:
            logger.error(f"Detection işleme hatası ({source_id}): {e}")
    
    async def _on_frame_received(self, source_id: str, frame: Any, detection_result: MultiModelDetectionResult) -> None:
        """Frame alındığında çağrılır"""
        try:
            # Frame'i WebSocket üzerinden gönder (isteğe bağlı)
            # Bu işlem bandwidth yoğun olduğu için dikkatli kullanılmalı
            pass
            
        except Exception as e:
            logger.error(f"Frame işleme hatası ({source_id}): {e}")
    
    def _on_detection_received_sync(self, source_id: str, detection_result: MultiModelDetectionResult) -> None:
        """Detection alındığında çağrılır (sync version)"""
        try:
            with self._lock:
                # Latest detection'ı güncelle
                self.latest_detections[source_id] = detection_result
                
                # History'e ekle
                if source_id not in self.detection_history:
                    self.detection_history[source_id] = []
                
                self.detection_history[source_id].append(detection_result)
                
                # History limitini kontrol et
                if len(self.detection_history[source_id]) > self.max_history_length:
                    self.detection_history[source_id] = self.detection_history[source_id][-self.max_history_length:]
                
                # Statistics güncelle
                self.total_frames_processed += 1
            
            # WebSocket üzerinden gönder (detection varsa veya yoksa da gönder)
            try:
                # Saklanan main loop'u kullan
                if self._main_loop and self._main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._send_detection_to_websocket(source_id, detection_result),
                        self._main_loop
                    )
                else:
                    logger.debug(f"WebSocket gönderimi atlandı - main event loop aktif değil")
            except Exception as e:
                # Event loop yoksa veya başka bir hata varsa, sadece logla
                logger.debug(f"WebSocket gönderimi atlandı - {e}")
            
        except Exception as e:
            logger.error(f"Sync detection işleme hatası ({source_id}): {e}")
    
    def _on_frame_received_sync(self, source_id: str, frame: Any, detection_result: MultiModelDetectionResult) -> None:
        """Frame alındığında çağrılır (sync version)"""
        try:
            # Frame'i WebSocket üzerinden gönder (isteğe bağlı)
            # Bu işlem bandwidth yoğun olduğu için dikkatli kullanılmalı
            pass
            
        except Exception as e:
            logger.error(f"Sync frame işleme hatası ({source_id}): {e}")
    
    async def _send_detection_to_websocket(self, source_id: str, detection_result: MultiModelDetectionResult) -> None:
        """Detection sonuçlarını WebSocket üzerinden gönder"""
        try:
            # Aktif modellerin detection'larını serialize et
            general_detections = []
            if hasattr(detection_result, 'general_detections') and detection_result.general_detections:
                for detection in detection_result.general_detections:
                    general_detections.append({
                        'class_id': int(detection.class_id),
                        'class_name': str(detection.class_name),
                        'confidence': float(detection.confidence),
                        'bbox': [float(x) for x in detection.bbox],
                        'center': [float(x) for x in detection.center],
                        'area': float(detection.area),
                        'model_type': detection.model_type.value
                    })
            
            farm_detections = []
            if hasattr(detection_result, 'farm_detections') and detection_result.farm_detections:
                for detection in detection_result.farm_detections:
                    farm_detections.append({
                        'class_id': int(detection.class_id),
                        'class_name': str(detection.class_name),
                        'confidence': float(detection.confidence),
                        'bbox': [float(x) for x in detection.bbox],
                        'center': [float(x) for x in detection.center],
                        'area': float(detection.area),
                        'model_type': detection.model_type.value
                    })
            
            zararli_detections = []
            if hasattr(detection_result, 'zararli_detections') and detection_result.zararli_detections:
                for detection in detection_result.zararli_detections:
                    zararli_detections.append({
                        'class_id': int(detection.class_id),
                        'class_name': str(detection.class_name),
                        'confidence': float(detection.confidence),
                        'bbox': [float(x) for x in detection.bbox],
                        'center': [float(x) for x in detection.center],
                        'area': float(detection.area),
                        'model_type': detection.model_type.value
                    })
            
            # Sadece aktif modellerin detection'larını serialize et
            all_detections = []
            for detection in detection_result.all_detections:
                all_detections.append({
                    'class_id': int(detection.class_id),
                    'class_name': str(detection.class_name),
                    'confidence': float(detection.confidence),
                    'bbox': [float(x) for x in detection.bbox],
                    'center': [float(x) for x in detection.center],
                    'area': float(detection.area),
                    'model_type': detection.model_type.value
                })
            
            # Processing times'ı da dönüştür
            processing_times = {}
            if hasattr(detection_result, 'processing_times') and detection_result.processing_times:
                for key, value in detection_result.processing_times.items():
                    processing_times[key] = float(value) if value is not None else 0.0
            
            message = {
                'type': 'multi_model_detection',
                'source_id': source_id,
                'data': {
                    'frame_id': int(detection_result.frame_id),
                    'timestamp': float(detection_result.timestamp),
                    'general_detections': general_detections,
                    'farm_detections': farm_detections,
                    'zararli_detections': zararli_detections,
                    'all_detections': all_detections,
                    'processing_times': processing_times,
                    'frame_size': [int(x) for x in detection_result.frame_size] if detection_result.frame_size else [0, 0],
                    'detection_counts': {
                        'general': len(detection_result.general_detections),
                        'farm': len(detection_result.farm_detections),
                        'zararli': len(detection_result.zararli_detections),
                        'total': len(detection_result.all_detections)
                    }
                }
            }
            
            # WebSocket'e gönder
            await self.websocket_manager.broadcast_to_source(source_id, json.dumps(message))
            
        except Exception as e:
            logger.error(f"WebSocket gönderme hatası: {e}")
            # Daha detaylı hata bilgisi için
            import traceback
            logger.error(f"Hata detayı: {traceback.format_exc()}")
    
    async def _start_background_tasks(self) -> None:
        """Background task'leri başlat"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.stats_task is None or self.stats_task.done():
            self.stats_task = asyncio.create_task(self._stats_loop())
    
    async def _cleanup_loop(self) -> None:
        """Periyodik temizlik döngüsü"""
        while True:
            try:
                await asyncio.sleep(self.auto_cleanup_interval)
                await self._perform_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop hatası: {e}")
    
    async def _stats_loop(self) -> None:
        """İstatistik güncelleme döngüsü"""
        while True:
            try:
                await asyncio.sleep(30)  # 30 saniyede bir
                await self._update_and_send_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stats loop hatası: {e}")
    
    async def _perform_cleanup(self) -> None:
        """Temizlik işlemlerini gerçekleştir"""
        try:
            current_time = time.time()
            
            with self._lock:
                # Eski detection history'lerini temizle
                for source_id in list(self.detection_history.keys()):
                    if source_id not in self.active_sources or not self.active_sources[source_id]:
                        del self.detection_history[source_id]
                        continue
                    
                    history = self.detection_history[source_id]
                    filtered_history = [
                        detection for detection in history
                        if current_time - detection.timestamp <= 300  # 5 dakikadan eski olanları temizle
                    ]
                    
                    self.detection_history[source_id] = filtered_history
            
            logger.info("Periyodik temizlik tamamlandı")
            
        except Exception as e:
            logger.error(f"Cleanup hatası: {e}")
    
    async def _update_and_send_stats(self) -> None:
        """İstatistikleri güncelle ve WebSocket'e gönder"""
        try:
            stats = self.get_global_statistics()
            
            message = {
                'type': 'statistics',
                'data': stats
            }
            
            # Tüm WebSocket bağlantılarına gönder
            await self.websocket_manager.broadcast_to_all(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Stats gönderme hatası: {e}")
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Global sistem istatistiklerini döndür"""
        elapsed_time = time.time() - self.start_time
        
        with self._lock:
            # Source istatistikleri
            source_stats = {}
            total_frames_received = 0
            total_fps = 0
            
            for source_id, source in self.sources.items():
                if hasattr(source, 'get_statistics'):
                    source_stats[source_id] = source.get_statistics()
                    total_frames_received += source_stats[source_id].get('frames_received', 0)
                    total_fps += source_stats[source_id].get('current_fps', 0)
            
            # Detector istatistikleri
            detector_stats = self.detector.get_statistics() if hasattr(self.detector, 'get_statistics') else {}
            
            return {
                'global_stats': {
                    'total_frames_processed': self.total_frames_processed,
                    'total_frames_received': total_frames_received,
                    'total_fps': total_fps,
                    'uptime': elapsed_time,
                    'active_sources': len([s for s in self.active_sources.values() if s])
                },
                'sources': source_stats,
                'detector': detector_stats,
                'timestamp': time.time()
            }
    
    async def cleanup(self) -> None:
        """Tüm kaynakları temizle"""
        logger.info("Video Manager temizleniyor...")
        
        # Background task'leri iptal et
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
        
        if self.stats_task and not self.stats_task.done():
            self.stats_task.cancel()
        
        # Tüm kaynakları durdur
        source_ids = list(self.sources.keys())
        for source_id in source_ids:
            try:
                await self.stop_source(source_id)
            except Exception as e:
                logger.error(f"Source cleanup hatası ({source_id}): {e}")
        
        # Data temizle
        with self._lock:
            self.sources.clear()
            self.active_sources.clear()
            self.latest_detections.clear()
            self.detection_history.clear()
        
        logger.info("Video Manager temizlendi")
    
    def is_source_active(self, source_id: str) -> bool:
        """Kaynağın aktif olup olmadığını kontrol et"""
        with self._lock:
            return self.active_sources.get(source_id, False)
    
    def get_source_count(self) -> int:
        """Aktif kaynak sayısını döndür"""
        with self._lock:
            return len([s for s in self.active_sources.values() if s])
    
    async def restart_source(self, source_id: str) -> bool:
        """Kaynağı yeniden başlat"""
        try:
            logger.info(f"Kaynak yeniden başlatılıyor: {source_id}")
            
            # Mevcut konfigurasyonu sakla
            source_info = self.get_source_info(source_id)
            
            # Kaynağı durdur
            await self.stop_source(source_id)
            
            # Kısa bekleme
            await asyncio.sleep(2)
            
            # Kaynağı yeniden başlat
            if source_id == 'local_camera':
                camera_id = source_info.get('camera_info', {}).get('camera_id', 0)
                await self.start_local_camera(camera_id)
            elif source_id == 'dji_drone':
                connection_string = source_info.get('drone_info', {}).get('connection_string', 'udp:11111')
                await self.start_dji_drone(connection_string)
            elif source_id == 'parrot_anafi':
                drone_ip = source_info.get('drone_info', {}).get('drone_ip', '192.168.42.1')
                await self.start_parrot_anafi(drone_ip)
            else:
                return False
            
            logger.info(f"Kaynak başarıyla yeniden başlatıldı: {source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Kaynak yeniden başlatma hatası ({source_id}): {e}")
            return False