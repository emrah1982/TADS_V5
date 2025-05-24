import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
import threading

logger = logging.getLogger(__name__)

class WebSocketConnection:
    """WebSocket bağlantısı wrapper sınıfı"""
    
    def __init__(self, websocket: WebSocket, source_type: str, client_id: str = None):
        self.websocket = websocket
        self.source_type = source_type
        self.client_id = client_id or f"{source_type}_{int(time.time() * 1000)}"
        self.connected_at = time.time()
        self.last_ping = time.time()
        self.message_count = 0
        self.is_alive = True
    
    async def send(self, message: str):
        """Mesaj gönder"""
        try:
            if self.is_alive:
                await self.websocket.send_text(message)
                self.message_count += 1
                return True
        except Exception as e:
            logger.error(f"WebSocket mesaj gönderme hatası ({self.client_id}): {e}")
            self.is_alive = False
            return False
        return False
    
    async def ping(self):
        """Ping gönder"""
        try:
            if self.is_alive:
                ping_message = json.dumps({
                    "type": "ping",
                    "timestamp": time.time()
                })
                success = await self.send(ping_message)
                if success:
                    self.last_ping = time.time()
                return success
        except Exception as e:
            logger.error(f"WebSocket ping hatası ({self.client_id}): {e}")
            self.is_alive = False
            return False
        return False
    
    def get_info(self) -> Dict:
        """Bağlantı bilgilerini döndür"""
        return {
            "client_id": self.client_id,
            "source_type": self.source_type,
            "connected_at": self.connected_at,
            "uptime": time.time() - self.connected_at,
            "last_ping": self.last_ping,
            "message_count": self.message_count,
            "is_alive": self.is_alive
        }

class WebSocketManager:
    """WebSocket bağlantılarını yöneten sınıf"""
    
    def __init__(self):
        self.connections: Dict[str, List[WebSocketConnection]] = {}
        self.all_connections: List[WebSocketConnection] = []
        self._lock = threading.Lock()
        
        # Configuration
        self.ping_interval = 30  # 30 saniye
        self.connection_timeout = 300  # 5 dakika
        self.max_connections_per_source = 10
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        self.start_time = time.time()
        
        # Background tasks
        self.ping_task = None
        self.cleanup_task = None
    
    async def notify_source_started(self, source_type: str):
        """Kaynak başlatıldığında tüm bağlantılara bildirim gönder"""
        try:
            message = json.dumps({
                "type": "source_started",
                "source": source_type,
                "timestamp": time.time()
            })
            await self.broadcast_to_all(message)
            logger.info(f"{source_type} kaynağı başlatıldı bildirimi gönderildi")
        except Exception as e:
            logger.error(f"Source started notification hatası: {e}")
    
    async def connect(self, websocket: WebSocket, source_type: str, client_id: str = None):
        """Yeni WebSocket bağlantısı ekle"""
        try:
            await websocket.accept()
            
            # Connection object oluştur
            connection = WebSocketConnection(websocket, source_type, client_id)
            
            with self._lock:
                # Source type için liste yoksa oluştur
                if source_type not in self.connections:
                    self.connections[source_type] = []
                
                # Maksimum bağlantı kontrolü
                if len(self.connections[source_type]) >= self.max_connections_per_source:
                    # En eski bağlantıyı kapat
                    oldest_connection = min(
                        self.connections[source_type],
                        key=lambda c: c.connected_at
                    )
                    await self._remove_connection(oldest_connection)
                
                # Yeni bağlantıyı ekle
                self.connections[source_type].append(connection)
                self.all_connections.append(connection)
                self.total_connections += 1
            
            logger.info(f"WebSocket bağlantısı eklendi: {connection.client_id} ({source_type})")
            
            # Hoş geldin mesajı gönder
            welcome_message = {
                "type": "welcome",
                "client_id": connection.client_id,
                "source_type": source_type,
                "server_time": time.time()
            }
            
            await connection.send(json.dumps(welcome_message))
            
            return connection
            
        except Exception as e:
            logger.error(f"WebSocket bağlantı hatası: {e}")
            raise
    
    def disconnect(self, websocket: WebSocket, source_type: str):
        """WebSocket bağlantısını kaldır"""
        try:
            with self._lock:
                # Bağlantıyı bul ve kaldır
                if source_type in self.connections:
                    connections_to_remove = [
                        conn for conn in self.connections[source_type]
                        if conn.websocket == websocket
                    ]
                    
                    for connection in connections_to_remove:
                        self._remove_connection_sync(connection)
                        logger.info(f"WebSocket bağlantısı kaldırıldı: {connection.client_id}")
            
        except Exception as e:
            logger.error(f"WebSocket disconnect hatası: {e}")
    
    async def _remove_connection(self, connection: WebSocketConnection):
        """Bağlantıyı kaldır (async)"""
        try:
            connection.is_alive = False
            
            # WebSocket'i kapat
            try:
                await connection.websocket.close()
            except:
                pass
            
            with self._lock:
                self._remove_connection_sync(connection)
                
        except Exception as e:
            logger.error(f"Connection kaldırma hatası: {e}")
    
    def _remove_connection_sync(self, connection: WebSocketConnection):
        """Bağlantıyı kaldır (sync)"""
        try:
            # Source type listesinden kaldır
            if connection.source_type in self.connections:
                if connection in self.connections[connection.source_type]:
                    self.connections[connection.source_type].remove(connection)
                
                # Liste boşsa sil
                if not self.connections[connection.source_type]:
                    del self.connections[connection.source_type]
            
            # Global listeden kaldır
            if connection in self.all_connections:
                self.all_connections.remove(connection)
            
        except Exception as e:
            logger.error(f"Sync connection kaldırma hatası: {e}")
    
    async def broadcast_to_source(self, source_type: str, message: str):
        """Belirli kaynak tipindeki tüm bağlantılara mesaj gönder"""
        try:
            with self._lock:
                connections = self.connections.get(source_type, []).copy()
            
            if not connections:
                return
            
            # Mesajları paralel gönder
            tasks = []
            for connection in connections:
                if connection.is_alive:
                    tasks.append(connection.send(message))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Başarısız bağlantıları temizle
                failed_connections = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception) or result is False:
                        failed_connections.append(connections[i])
                
                # Başarısız bağlantıları kaldır
                for connection in failed_connections:
                    await self._remove_connection(connection)
                
                # İstatistik güncelle
                successful_sends = sum(1 for r in results if r is True)
                self.total_messages_sent += successful_sends
            
        except Exception as e:
            logger.error(f"Broadcast hatası ({source_type}): {e}")
    
    async def broadcast_to_all(self, message: str):
        """Tüm bağlantılara mesaj gönder"""
        try:
            with self._lock:
                connections = self.all_connections.copy()
            
            if not connections:
                return
            
            # Mesajları paralel gönder
            tasks = []
            for connection in connections:
                if connection.is_alive:
                    tasks.append(connection.send(message))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Başarısız bağlantıları temizle
                failed_connections = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception) or result is False:
                        failed_connections.append(connections[i])
                
                # Başarısız bağlantıları kaldır
                for connection in failed_connections:
                    await self._remove_connection(connection)
                
                # İstatistik güncelle
                successful_sends = sum(1 for r in results if r is True)
                self.total_messages_sent += successful_sends
                
        except Exception as e:
            logger.error(f"Global broadcast hatası: {e}")
    
    async def send_to_client(self, client_id: str, message: str) -> bool:
        """Belirli client'a mesaj gönder"""
        try:
            with self._lock:
                target_connection = None
                for connection in self.all_connections:
                    if connection.client_id == client_id and connection.is_alive:
                        target_connection = connection
                        break
            
            if target_connection:
                success = await target_connection.send(message)
                if success:
                    self.total_messages_sent += 1
                return success
            
            return False
                
        except Exception as e:
            logger.error(f"Client mesaj gönderme hatası ({client_id}): {e}")
            return False
    
    def get_connection_count(self, source_type: str = None) -> int:
        """Bağlantı sayısını döndür"""
        with self._lock:
            if source_type:
                return len(self.connections.get(source_type, []))
            else:
                return len(self.all_connections)
    
    def get_active_sources(self) -> List[str]:
        """Aktif kaynak tiplerini döndür"""
        with self._lock:
            return list(self.connections.keys())
    
    def get_connections_info(self) -> Dict:
        """Tüm bağlantı bilgilerini döndür"""
        with self._lock:
            connections_by_source = {}
            
            for source_type, connections in self.connections.items():
                connections_by_source[source_type] = [
                    conn.get_info() for conn in connections if conn.is_alive
                ]
            
            return {
                "total_connections": len(self.all_connections),
                "active_connections": len([c for c in self.all_connections if c.is_alive]),
                "connections_by_source": connections_by_source,
                "total_messages_sent": self.total_messages_sent,
                "uptime": time.time() - self.start_time
            }
    
    def get_statistics(self) -> Dict:
        """WebSocket istatistiklerini döndür"""
        return self.get_connections_info()
    
    async def _ping_loop(self):
        """Periyodik ping döngüsü"""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                await self._ping_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping loop hatası: {e}")
    
    async def _cleanup_loop(self):
        """Ölü bağlantıları temizleme döngüsü"""
        while True:
            try:
                await asyncio.sleep(60)  # Her dakika
                await self._cleanup_dead_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop hatası: {e}")
    
    async def _ping_all_connections(self):
        """Tüm bağlantılara ping gönder"""
        try:
            with self._lock:
                connections = self.all_connections.copy()
            
            tasks = []
            for connection in connections:
                if connection.is_alive:
                    tasks.append(connection.ping())
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Ping all hatası: {e}")
    
    async def _cleanup_dead_connections(self):
        """Ölü ve timeout olmuş bağlantıları temizle"""
        try:
            current_time = time.time()
            connections_to_remove = []
            
            with self._lock:
                # Timeout olmuş bağlantıları bul
                for connection in self.all_connections:
                    if current_time - connection.last_ping > self.connection_timeout:
                        connection.is_alive = False
                        connections_to_remove.append(connection)
                
                # Ölü bağlantıları temizle
                for connection in connections_to_remove:
                    await self._remove_connection(connection)
                
                if connections_to_remove:
                    logger.info(f"{len(connections_to_remove)} ölü bağlantı temizlendi")
                    
        except Exception as e:
            logger.error(f"Ölü bağlantı temizleme hatası: {e}")
    
    async def shutdown(self):
        """WebSocket manager'ı kapat"""
        try:
            logger.info("WebSocket Manager kapatılıyor...")
            
            # Background task'leri iptal et
            if self.ping_task and not self.ping_task.done():
                self.ping_task.cancel()
            
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
            
            # Tüm bağlantıları kapat
            with self._lock:
                connections = self.all_connections.copy()
            
            for connection in connections:
                try:
                    await connection.websocket.close()
                except Exception as e:
                    logger.error(f"Bağlantı kapatma hatası: {e}")
            
            # Listeleri temizle
            with self._lock:
                self.connections.clear()
                self.all_connections.clear()
            
            logger.info("WebSocket Manager başarıyla kapatıldı")
            
        except Exception as e:
            logger.error(f"WebSocket manager kapatma hatası: {e}")