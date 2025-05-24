import cv2
import asyncio
import threading
import time
import logging
import requests
import socket
import json
import numpy as np
from typing import Optional, Callable, Dict, Any, List
from queue import Queue, Empty
import xml.etree.ElementTree as ET

from models.multi_model_detector import MultiModelYOLODetector, MultiModelDetectionResult

logger = logging.getLogger(__name__)

class ParrotAnafiSource:
    """Parrot Anafi drone video kaynağı sınıfı"""
    
    def __init__(self, detector: MultiModelYOLODetector):
        """
        Parrot Anafi kaynağını başlat
        
        Args:
            detector: YOLO detector instance
        """
        self.detector = detector
        self.drone_ip = "192.168.42.1"  # Parrot Anafi default IP
        self.cap = None
        self.is_running = False
        self.thread = None
        
        # Connection settings
        self.http_port = 80
        self.video_stream_url = None
        self.control_url = None
        
        # Frame queue
        self.frame_queue = Queue(maxsize=10)
        self.latest_frame = None
        self.latest_detections = None
        
        # Callbacks
        self.frame_callback = None
        self.detection_callback = None
        
        # Configuration
        self.fps = 30
        self.frame_width = 1920
        self.frame_height = 1080
        self.frame_id = 0
        
        # Statistics
        self.frames_received = 0
        self.frames_processed = 0
        self.connection_lost_count = 0
        self.start_time = time.time()
        
        # Threading
        self._lock = threading.Lock()
        
        # Connection status
        self.is_connected = False
        self.drone_info = {}
        self.last_heartbeat = time.time()
        
        # Parrot Anafi specific
        self.session = None
        self.stream_session_id = None
        self.battery_level = 0
        self.gps_location = None
        self.drone_state = {}
    
    async def initialize(self, drone_ip: str = "192.168.42.1") -> bool:
        """Parrot Anafi bağlantısını başlat"""
        try:
            logger.info(f"Parrot Anafi bağlantısı başlatılıyor: {drone_ip}")
            
            self.drone_ip = drone_ip
            self.control_url = f"http://{drone_ip}"
            
            # HTTP session oluştur
            self.session = requests.Session()
            self.session.timeout = 10
            
            # Drone bağlantısını test et
            success = await self._test_connection()
            if not success:
                logger.error("Parrot Anafi bağlantı testi başarısız")
                return False
            
            # Drone bilgilerini al
            await self._get_drone_info()
            
            # Video stream URL'ini belirle
            await self._setup_video_stream()
            
            if self.video_stream_url:
                logger.info(f"Parrot Anafi bağlantısı başarıyla kuruldu: {self.video_stream_url}")
                self.is_connected = True
                return True
            else:
                logger.error("Video stream URL'i alınamadı")
                return False
                
        except Exception as e:
            logger.error(f"Parrot Anafi başlatma hatası: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Drone bağlantısını test et"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.session.get(f"{self.control_url}/api/v1/info")
            )
            
            if response.status_code == 200:
                logger.info("Parrot Anafi bağlantı testi başarılı")
                return True
            else:
                logger.error(f"Bağlantı testi başarısız: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Bağlantı testi hatası: {e}")
            return False
    
    async def _get_drone_info(self):
        """Drone bilgilerini al"""
        try:
            # Temel bilgiler
            info_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(f"{self.control_url}/api/v1/info")
            )
            
            if info_response.status_code == 200:
                self.drone_info = info_response.json()
                logger.info(f"Drone bilgileri alındı: {self.drone_info.get('name', 'Unknown')}")
            
            # Durum bilgileri
            await self._update_drone_status()
            
        except Exception as e:
            logger.error(f"Drone bilgileri alma hatası: {e}")
    
    async def _update_drone_status(self):
        """Drone durum bilgilerini güncelle"""
        try:
            status_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(f"{self.control_url}/api/v1/status")
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                self.drone_state = status_data
                
                # Batarya seviyesi
                if 'battery' in status_data:
                    self.battery_level = status_data['battery'].get('level', 0)
                
                # GPS konumu
                if 'gps' in status_data:
                    self.gps_location = {
                        'latitude': status_data['gps'].get('latitude'),
                        'longitude': status_data['gps'].get('longitude'),
                        'altitude': status_data['gps'].get('altitude')
                    }
                
                logger.debug(f"Drone durumu güncellendi: Batarya {self.battery_level}%")
            
        except Exception as e:
            logger.error(f"Durum güncelleme hatası: {e}")
    
    async def _setup_video_stream(self):
        """Video stream ayarlarını yap"""
        try:
            # Parrot Anafi için video stream endpoint'lerini test et
            possible_streams = [
                f"rtsp://{self.drone_ip}/live",
                f"http://{self.drone_ip}/video/live.sdp",
                f"udp://@{self.drone_ip}:55004",
                f"http://{self.drone_ip}:80/video.mjpeg"
            ]
            
            for stream_url in possible_streams:
                try:
                    logger.info(f"Video stream test ediliyor: {stream_url}")
                    
                    # Stream'i test et
                    test_cap = cv2.VideoCapture(stream_url)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        test_cap.release()
                        
                        if ret and frame is not None:
                            self.video_stream_url = stream_url
                            logger.info(f"Video stream bulundu: {stream_url}")
                            break
                    else:
                        test_cap.release()
                        
                except Exception as e:
                    logger.debug(f"Stream test hatası {stream_url}: {e}")
                    continue
            
            if not self.video_stream_url:
                # Manuel stream başlatma komutu gönder
                await self._start_video_stream()
            
        except Exception as e:
            logger.error(f"Video stream ayarlama hatası: {e}")
    
    async def _start_video_stream(self):
        """Video stream'i manuel olarak başlat"""
        try:
            # Parrot Anafi'ye stream başlatma komutu gönder
            stream_command = {
                "type": "start_video_stream",
                "parameters": {
                    "resolution": "1080p",
                    "framerate": 30
                }
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.post(
                    f"{self.control_url}/api/v1/commands",
                    json=stream_command
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'stream_url' in result:
                    self.video_stream_url = result['stream_url']
                    logger.info(f"Video stream başlatıldı: {self.video_stream_url}")
                else:
                    # Varsayılan URL'i dene
                    self.video_stream_url = f"rtsp://{self.drone_ip}/live"
            
        except Exception as e:
            logger.error(f"Video stream başlatma hatası: {e}")
    
    def start_capture(self):
        """Video yakalama işlemini başlat"""
        if self.is_running:
            logger.warning("Parrot Anafi zaten çalışıyor")
            return
        
        if not self.video_stream_url:
            logger.error("Video stream URL'i bulunamadı")
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_id = 0
        
        # Thread'i başlat
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # Status update thread'i başlat
        self.status_thread = threading.Thread(target=self._status_update_loop, daemon=True)
        self.status_thread.start()
        
        logger.info("Parrot Anafi yakalama başlatıldı")
    
    def stop_capture(self):
        """Video yakalama işlemini durdur"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.is_connected = False
        
        # Thread'lerin bitmesini bekle
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        if hasattr(self, 'status_thread') and self.status_thread.is_alive():
            self.status_thread.join(timeout=3)
        
        # Bağlantıları kapat
        self._close_connections()
        
        logger.info("Parrot Anafi yakalama durduruldu")
    
    def _close_connections(self):
        """Tüm bağlantıları kapat"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            
            if self.session:
                # Stream'i durdur
                try:
                    stop_command = {"type": "stop_video_stream"}
                    self.session.post(
                        f"{self.control_url}/api/v1/commands",
                        json=stop_command,
                        timeout=5
                    )
                except:
                    pass
                
                self.session.close()
                self.session = None
                
        except Exception as e:
            logger.error(f"Bağlantı kapatma hatası: {e}")
    
    def _capture_loop(self):
        """Ana yakalama döngüsü"""
        logger.info("Parrot Anafi yakalama döngüsü başladı")
        
        try:
            # Video stream'i aç
            self.cap = cv2.VideoCapture(self.video_stream_url)
            
            if not self.cap.isOpened():
                logger.error("Parrot Anafi video stream açılamadı")
                return
            
            # Stream ayarları
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Düşük latency
            
            logger.info("Parrot Anafi video stream açıldı")
            
            consecutive_failures = 0
            max_failures = 30
            
            while self.is_running:
                try:
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        consecutive_failures += 1
                        logger.warning(f"Frame alınamadı (#{consecutive_failures})")
                        
                        if consecutive_failures >= max_failures:
                            logger.error("Çok fazla başarısız frame, yeniden bağlanılacak")
                            asyncio.create_task(self._reconnect())
                            consecutive_failures = 0
                        
                        time.sleep(0.1)
                        continue
                    
                    # Başarılı frame
                    consecutive_failures = 0
                    self._process_received_frame(frame)
                    self.last_heartbeat = time.time()
                    
                except Exception as e:
                    logger.error(f"Capture loop hatası: {e}")
                    consecutive_failures += 1
                    time.sleep(0.1)
                    
                    if consecutive_failures >= max_failures:
                        break
        
        except Exception as e:
            logger.error(f"Capture loop kritik hatası: {e}")
        
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
        
        logger.info("Parrot Anafi yakalama döngüsü sona erdi")
    
    def _status_update_loop(self):
        """Drone durum güncelleme döngüsü"""
        while self.is_running:
            try:
                asyncio.create_task(self._update_drone_status())
                time.sleep(5)  # 5 saniyede bir güncelle
            except Exception as e:
                logger.error(f"Status update hatası: {e}")
                time.sleep(10)
    
    def _process_received_frame(self, frame: np.ndarray):
        """Alınan frame'i işle"""
        try:
            with self._lock:
                self.frames_received += 1
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
            
            # Detection işlemi (async)
            asyncio.create_task(self._process_frame(current_frame_id, frame))
            
        except Exception as e:
            logger.error(f"Frame işleme hatası: {e}")
    
    async def _process_frame(self, frame_id: int, frame: np.ndarray):
        """Frame'i işle ve tespit yap"""
        try:
            # YOLO detection
            detection_result = self.detector.detect(frame, frame_id)
            
            with self._lock:
                self.frames_processed += 1
                self.latest_detections = detection_result
            
            # Callback çağır
            if self.detection_callback:
                await self.detection_callback(detection_result)
            
            # Annotated frame oluştur
            if detection_result.detections:
                annotated_frame = self.detector.draw_detections(frame, detection_result.detections)
            else:
                annotated_frame = frame
            
            # Frame callback çağır
            if self.frame_callback:
                await self.frame_callback(annotated_frame, detection_result)
                
        except Exception as e:
            logger.error(f"Parrot frame işleme hatası: {e}")
    
    async def _reconnect(self):
        """Yeniden bağlanmayı dene"""
        logger.info("Parrot Anafi yeniden bağlanma deneniyor...")
        
        try:
            self._close_connections()
            await asyncio.sleep(3)
            
            success = await self.initialize(self.drone_ip)
            if success:
                logger.info("Parrot Anafi yeniden bağlandı")
                self.is_connected = True
            else:
                logger.error("Parrot Anafi yeniden bağlanamadı")
                
        except Exception as e:
            logger.error(f"Yeniden bağlanma hatası: {e}")
    
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
                logger.error(f"Parrot frame generator hatası: {e}")
                continue
    
    def set_frame_callback(self, callback: Callable):
        """Frame callback fonksiyonunu ayarla"""
        self.frame_callback = callback
    
    def set_detection_callback(self, callback: Callable):
        """Detection callback fonksiyonunu ayarla"""
        self.detection_callback = callback
    
    def get_drone_info(self) -> Dict[str, Any]:
        """Drone bilgilerini döndür"""
        return {
            "drone_type": "Parrot_Anafi",
            "drone_ip": self.drone_ip,
            "video_stream_url": self.video_stream_url,
            "is_connected": self.is_connected,
            "battery_level": self.battery_level,
            "gps_location": self.gps_location,
            "drone_info": self.drone_info,
            "drone_state": self.drone_state
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        elapsed_time = time.time() - self.start_time
        
        with self._lock:
            receive_fps = self.frames_received / elapsed_time if elapsed_time > 0 else 0
            process_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
            
            return {
                "source_type": "parrot_anafi",
                "is_running": self.is_running,
                "is_connected": self.is_connected,
                "frames_received": self.frames_received,
                "frames_processed": self.frames_processed,
                "connection_lost_count": self.connection_lost_count,
                "receive_fps": receive_fps,
                "process_fps": process_fps,
                "elapsed_time": elapsed_time,
                "battery_level": self.battery_level
            }
    
    async def send_command(self, command_type: str, parameters: Dict = None) -> Dict:
        """Drone'a komut gönder"""
        try:
            if not self.session or not self.is_connected:
                return {"success": False, "error": "Drone bağlı değil"}
            
            command = {
                "type": command_type,
                "parameters": parameters or {}
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.post(
                    f"{self.control_url}/api/v1/commands",
                    json=command
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Komut başarılı: {command_type}")
                return {"success": True, "result": result}
            else:
                logger.error(f"Komut hatası: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Komut gönderme hatası: {e}")
            return {"success": False, "error": str(e)}
    
    async def takeoff(self):
        """Drone'u kaldır"""
        return await self.send_command("takeoff")
    
    async def land(self):
        """Drone'u indir"""
        return await self.send_command("land")
    
    async def move(self, direction: str, distance: float):
        """Drone'u hareket ettir"""
        return await self.send_command("move", {
            "direction": direction,
            "distance": distance
        })
    
    async def rotate(self, angle: float):
        """Drone'u döndür"""
        return await self.send_command("rotate", {"angle": angle})
    
    async def set_camera_tilt(self, angle: float):
        """Kamera açısını ayarla"""
        return await self.send_command("camera_tilt", {"angle": angle})
    
    async def start_recording(self):
        """Video kaydını başlat"""
        return await self.send_command("start_recording")
    
    async def stop_recording(self):
        """Video kaydını durdur"""
        return await self.send_command("stop_recording")
    
    async def take_photo(self):
        """Fotoğraf çek"""
        return await self.send_command("take_photo")
    
    def get_flight_status(self) -> Dict[str, Any]:
        """Uçuş durumunu döndür"""
        return {
            "battery_level": self.battery_level,
            "gps_location": self.gps_location,
            "connection_quality": self._get_connection_quality(),
            "flight_time": time.time() - self.start_time if self.is_running else 0,
            "drone_state": self.drone_state
        }
    
    def _get_connection_quality(self) -> str:
        """Bağlantı kalitesini değerlendir"""
        if not self.is_connected:
            return "disconnected"
        
        elapsed_since_heartbeat = time.time() - self.last_heartbeat
        
        if elapsed_since_heartbeat < 2:
            return "excellent"
        elif elapsed_since_heartbeat < 5:
            return "good"
        elif elapsed_since_heartbeat < 10:
            return "poor"
        else:
            return "critical"
    
    async def cleanup(self):
        """Temizlik işlemleri"""
        logger.info("Parrot Anafi temizleniyor")
        self.stop_capture()
        
        # Queue'yu temizle
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        logger.info("Parrot Anafi temizlendi")