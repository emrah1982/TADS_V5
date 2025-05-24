import cv2
import asyncio
import threading
import time
import logging
import socket
import numpy as np
from typing import Optional, Callable, Dict, Any
from queue import Queue, Empty
import struct

from models.multi_model_detector import MultiModelYOLODetector, MultiModelDetectionResult

logger = logging.getLogger(__name__)

class DJIDroneSource:
    """DJI Drone video kaynağı sınıfı"""
    
    def __init__(self, detector: MultiModelYOLODetector):
        """
        DJI Drone kaynağını başlat
        
        Args:
            detector: YOLO detector instance
        """
        self.detector = detector
        self.connection_string = None
        self.cap = None
        self.is_running = False
        self.thread = None
        
        # Connection settings
        self.drone_ip = "192.168.1.1"
        self.video_port = 11111
        self.control_port = 8889
        
        # Frame queue
        self.frame_queue = Queue(maxsize=10)
        self.latest_frame = None
        self.latest_detections = None
        
        # Callbacks
        self.frame_callback = None
        self.detection_callback = None
        
        # Configuration
        self.fps = 30
        self.frame_width = 1280
        self.frame_height = 720
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
        self.last_heartbeat = time.time()
        
        # UDP socket for video stream
        self.video_socket = None
        self.control_socket = None
    
    async def initialize(self, connection_string: str = "udp://:11111") -> bool:
        """DJI Drone bağlantısını başlat"""
        try:
            logger.info(f"DJI Drone bağlantısı başlatılıyor: {connection_string}")
            
            self.connection_string = connection_string
            
            # Connection string'i parse et
            if connection_string.startswith("udp://"):
                # UDP bağlantısı
                parts = connection_string.replace("udp://", "").split(":")
                if len(parts) == 2 and parts[0]:
                    self.drone_ip = parts[0]
                    self.video_port = int(parts[1])
                elif len(parts) == 2 and not parts[0]:
                    # :port formatı - herhangi IP'den dinle
                    self.video_port = int(parts[1])
                
                success = await self._initialize_udp_connection()
            else:
                # RTMP veya HTTP stream
                success = await self._initialize_stream_connection()
            
            if success:
                logger.info("DJI Drone bağlantısı başarıyla kuruldu")
                self.is_connected = True
                return True
            else:
                logger.error("DJI Drone bağlantısı kurulamadı")
                return False
                
        except Exception as e:
            logger.error(f"DJI Drone başlatma hatası: {e}")
            return False
    
    async def _initialize_udp_connection(self) -> bool:
        """UDP video stream bağlantısını başlat"""
        try:
            # Video stream için UDP socket
            self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.video_socket.bind(('', self.video_port))
            self.video_socket.settimeout(5.0)
            
            # Control socket (komut gönderimi için)
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            logger.info(f"UDP bağlantısı kuruldu: Port {self.video_port}")
            
            # Drone'a bağlantı komutu gönder
            await self._send_connection_command()
            
            return True
            
        except Exception as e:
            logger.error(f"UDP bağlantı hatası: {e}")
            return False
    
    async def _initialize_stream_connection(self) -> bool:
        """Stream URL bağlantısını başlat"""
        try:
            # OpenCV ile stream bağlantısı
            self.cap = cv2.VideoCapture(self.connection_string)
            
            if not self.cap.isOpened():
                logger.error("Stream bağlantısı açılamadı")
                return False
            
            # Stream ayarları
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Düşük latency
            
            # Stream bilgilerini al
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Stream bağlantısı: {width}x{height} @ {fps} FPS")
            
            return True
            
        except Exception as e:
            logger.error(f"Stream bağlantı hatası: {e}")
            return False
    
    async def _send_connection_command(self):
        """Drone'a bağlantı komutları gönder"""
        try:
            if self.control_socket:
                # DJI Tello benzeri komutlar
                commands = [
                    b"command",      # Komut moduna geç
                    b"streamon",     # Video stream'i aç
                ]
                
                for cmd in commands:
                    self.control_socket.sendto(cmd, (self.drone_ip, self.control_port))
                    await asyncio.sleep(0.5)
                    
                logger.info("Drone komutları gönderildi")
                
        except Exception as e:
            logger.error(f"Komut gönderme hatası: {e}")
    
    def start_capture(self):
        """Video yakalama işlemini başlat"""
        if self.is_running:
            logger.warning("DJI Drone zaten çalışıyor")
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_id = 0
        
        # Thread'i başlat
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info("DJI Drone yakalama başlatıldı")
    
    def stop_capture(self):
        """Video yakalama işlemini durdur"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.is_connected = False
        
        # Thread'in bitmesini bekle
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        # Bağlantıları kapat
        self._close_connections()
        
        logger.info("DJI Drone yakalama durduruldu")
    
    def _close_connections(self):
        """Tüm bağlantıları kapat"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            
            if self.video_socket:
                self.video_socket.close()
                self.video_socket = None
            
            if self.control_socket:
                # Stream'i kapat
                try:
                    self.control_socket.sendto(b"streamoff", (self.drone_ip, self.control_port))
                except:
                    pass
                self.control_socket.close()
                self.control_socket = None
                
        except Exception as e:
            logger.error(f"Bağlantı kapatma hatası: {e}")
    
    def _capture_loop(self):
        """Ana yakalama döngüsü"""
        logger.info("DJI Drone yakalama döngüsü başladı")
        
        if self.video_socket:
            self._udp_capture_loop()
        else:
            self._stream_capture_loop()
    
    def _udp_capture_loop(self):
        """UDP video stream yakalama döngüsü"""
        buffer = b''
        
        while self.is_running:
            try:
                # UDP paketi al
                data, addr = self.video_socket.recvfrom(65536)
                
                if not data:
                    continue
                
                # H.264 stream'i decode et
                buffer += data
                
                # Frame'leri ayıkla
                while len(buffer) > 4:
                    # H.264 start code ara (0x00000001)
                    start_idx = buffer.find(b'\x00\x00\x00\x01')
                    if start_idx == -1:
                        break
                    
                    # Bir sonraki start code'u ara
                    next_start = buffer.find(b'\x00\x00\x00\x01', start_idx + 4)
                    
                    if next_start == -1:
                        break
                    
                    # Frame verilerini al
                    frame_data = buffer[start_idx:next_start]
                    buffer = buffer[next_start:]
                    
                    # Frame'i decode et
                    frame = self._decode_h264_frame(frame_data)
                    
                    if frame is not None:
                        self._process_received_frame(frame)
                
                self.last_heartbeat = time.time()
                
            except socket.timeout:
                # Timeout - bağlantı durumunu kontrol et
                if time.time() - self.last_heartbeat > 10:
                    logger.warning("DJI Drone bağlantısı koptu")
                    self.connection_lost_count += 1
                    self.is_connected = False
                    
                    # Yeniden bağlanmayı dene
                    asyncio.create_task(self._reconnect())
                    
                continue
                
            except Exception as e:
                logger.error(f"UDP yakalama hatası: {e}")
                break
        
        logger.info("DJI Drone UDP yakalama döngüsü sona erdi")
    
    def _stream_capture_loop(self):
        """Stream yakalama döngüsü"""
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.error("Stream bağlantısı koptu")
                    break
                
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Frame alınamadı")
                    self.connection_lost_count += 1
                    
                    # Yeniden bağlanmayı dene
                    if self.connection_lost_count > 10:
                        asyncio.create_task(self._reconnect())
                        self.connection_lost_count = 0
                    
                    time.sleep(0.1)
                    continue
                
                self._process_received_frame(frame)
                self.last_heartbeat = time.time()
                
            except Exception as e:
                logger.error(f"Stream yakalama hatası: {e}")
                break
        
        logger.info("DJI Drone stream yakalama döngüsü sona erdi")
    
    def _decode_h264_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        """H.264 frame'ini decode et"""
        try:
            # OpenCV ile H.264 decode (basit implementasyon)
            # Gerçek uygulamada FFmpeg veya specialized decoder kullanılmalı
            
            # Geçici dosya oluştur ve decode et
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.h264', delete=False) as f:
                f.write(frame_data)
                temp_file = f.name
            
            try:
                cap = cv2.VideoCapture(temp_file)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    return frame
                
            finally:
                os.unlink(temp_file)
            
            return None
            
        except Exception as e:
            logger.error(f"H.264 decode hatası: {e}")
            return None
    
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
            logger.error(f"DJI frame işleme hatası: {e}")
    
    async def _reconnect(self):
        """Yeniden bağlanmayı dene"""
        logger.info("DJI Drone yeniden bağlanma deneniyor...")
        
        try:
            self._close_connections()
            await asyncio.sleep(2)
            
            success = await self.initialize(self.connection_string)
            if success:
                logger.info("DJI Drone yeniden bağlandı")
                self.is_connected = True
            else:
                logger.error("DJI Drone yeniden bağlanamadı")
                
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
                logger.error(f"DJI frame generator hatası: {e}")
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
            "drone_type": "DJI",
            "connection_string": self.connection_string,
            "drone_ip": self.drone_ip,
            "video_port": self.video_port,
            "is_connected": self.is_connected,
            "connection_method": "UDP" if self.video_socket else "Stream"
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        elapsed_time = time.time() - self.start_time
        
        with self._lock:
            receive_fps = self.frames_received / elapsed_time if elapsed_time > 0 else 0
            process_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
            
            return {
                "source_type": "dji_drone",
                "is_running": self.is_running,
                "is_connected": self.is_connected,
                "frames_received": self.frames_received,
                "frames_processed": self.frames_processed,
                "connection_lost_count": self.connection_lost_count,
                "receive_fps": receive_fps,
                "process_fps": process_fps,
                "elapsed_time": elapsed_time
            }
    
    async def send_command(self, command: str) -> bool:
        """Drone'a komut gönder"""
        try:
            if self.control_socket and self.is_connected:
                self.control_socket.sendto(command.encode(), (self.drone_ip, self.control_port))
                logger.info(f"Komut gönderildi: {command}")
                return True
            else:
                logger.warning("Drone bağlı değil, komut gönderilemedi")
                return False
                
        except Exception as e:
            logger.error(f"Komut gönderme hatası: {e}")
            return False
    
    async def cleanup(self):
        """Temizlik işlemleri"""
        logger.info("DJI Drone temizleniyor")
        self.stop_capture()
        
        # Queue'yu temizle
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        logger.info("DJI Drone temizlendi")