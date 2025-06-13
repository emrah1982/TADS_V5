import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import logging
from typing import List, Dict, Tuple, Optional, Union
import threading
from dataclasses import dataclass
from enum import Enum

# Config manager import
from utils.config_manager import get_config_manager, ConfigManager, ModelConfig, ClassConfig

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model türleri"""
    GENERAL = "general"  # yolo11.pt - genel nesneler
    FARM = "farm"        # farm_best.pt - tarım analizi
    ZARARLI = "zararli"  # zararliTespiti_best.pt - tarım zararlılar

@dataclass
class Detection:
    """Tespit sonucu sınıfı"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    model_type: ModelType  # Hangi model tarafından tespit edildi

@dataclass
class MultiModelDetectionResult:
    """Multi-model tespit sonuçları konteyner sınıfı"""
    frame_id: int
    timestamp: float
    general_detections: List[Detection]  # yolo11.pt sonuçları
    farm_detections: List[Detection]     # farm_best.pt sonuçları
    zararli_detections: List[Detection]  # zararliTespiti_best.pt sonuçları
    domatesMineral_detections: List[Detection]  # domatesMineralTespiti_best.pt sonuçları
    domatesHastalik_detections: List[Detection]  # domatesHastalikTespiti_best.pt sonuçları
    domatesOlgunluk_detections: List[Detection]  # domatesOlgunlukTespiti_best.pt sonuçları
    all_detections: List[Detection]      # Tüm sonuçlar birleşik
    processing_times: Dict[str, float]   # Model bazlı işleme süreleri
    frame_size: Tuple[int, int]

class MultiModelYOLODetector:
    """Multi-model YOLO detector sınıfı"""
    
    def __init__(self, farm_model_path: str, general_model_path: str, zararli_model_path: str, domatesMineral_model_path: str, domatesHastalik_model_path: str, domatesOlgunluk_model_path: str):
        """
        Config-driven Multi-model YOLO detector'ı başlat
        
        Args:
            farm_model_path: Farm model dosya yolu
            general_model_path: Genel model dosya yolu
        """
        # Model paths
        self.farm_model_path = farm_model_path
        self.general_model_path = general_model_path


        self.zararli_model_path = zararli_model_path
        self.domatesMineral_model_path = domatesMineral_model_path
        self.domatesHastalik_model_path = domatesHastalik_model_path
        self.domatesOlgunluk_model_path = domatesOlgunluk_model_path

        
        # Model instances
        self.farm_model = None
        self.general_model = None
        self.zararli_model = None
        self.domatesMineral_model = None
        self.domatesHastalik_model = None
        self.domatesOlgunluk_model = None
        # Model enable flags
        self.enable_general = True
        self.enable_farm = True
        self.enable_zararli = True
        self.enable_domatesMineral = True
        self.enable_domatesHastalik = True
        self.enable_domatesOlgunluk = True

        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Threading
        self._lock = threading.Lock()
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        self.model_stats = {
            'farm': {'frames': 0, 'total_time': 0.0, 'last_detection_count': 0},
            'general': {'frames': 0, 'total_time': 0.0, 'last_detection_count': 0},            
            'zararli': {'frames': 0, 'total_time': 0.0, 'last_detection_count': 0},
            'domatesMineral': {'frames': 0, 'total_time': 0.0, 'last_detection_count': 0},
            'domatesHastalik': {'frames': 0, 'total_time': 0.0, 'last_detection_count': 0},
            'domatesOlgunluk': {'frames': 0, 'total_time': 0.0, 'last_detection_count': 0}
        }
        
        # Class names
        self.general_class_names = {}
        self.farm_class_names = {}
        self.zararli_class_names = {}
        
        # Config manager
        self.config_manager = get_config_manager()
        
        # Models dictionary
        self.models = {}
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Modelleri yükle"""
        try:
            # Device seçimi
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Kullanılan cihaz: {self.device}")
            
            # General model yükleme
            if self.enable_general and self.general_model_path:
                logger.info(f"Genel YOLO modeli yükleniyor: {self.general_model_path}")
                self.general_model = YOLO(self.general_model_path)
                self.general_model.to(self.device)
                
                # Config'den sınıf isimlerini al
                general_config = self.config_manager.get_model_config('general')
                if general_config and hasattr(general_config, 'classes'):
                    self.general_class_names = {int(k): v.display_name
                                              for k, v in general_config.classes.items()}
                else:
                    self.general_class_names = self.general_model.names
                
                logger.info(f"Genel model yüklendi. Sınıf sayısı: {len(self.general_class_names)}")
                
                # Models dictionary'e ekle
                self.models['general'] = {
                    'model': self.general_model,
                    'config': general_config
                }
            
            # Farm model yükleme
            if self.enable_farm and self.farm_model_path:
                logger.info(f"Farm modeli yükleniyor: {self.farm_model_path}")
                self.farm_model = YOLO(self.farm_model_path)
                self.farm_model.to(self.device)
                
                # Config'den farm model sınıflarını al
                farm_config = self.config_manager.get_model_config('farm')
                if farm_config and hasattr(farm_config, 'classes'):
                    self.farm_class_names = {int(k): v.display_name
                                           for k, v in farm_config.classes.items()}
                else:
                    self.farm_class_names = self.farm_model.names
                
                logger.info(f"Farm modeli yüklendi. Sınıf sayısı: {len(self.farm_class_names)}")
                
                # Models dictionary'e ekle
                self.models['farm'] = {
                    'model': self.farm_model,
                    'config': farm_config
                }
            
            # Zararlı model yükleme
            if self.enable_zararli and self.zararli_model_path:
                logger.info(f"Zararlı modeli yükleniyor: {self.zararli_model_path}")
                self.zararli_model = YOLO(self.zararli_model_path)
                self.zararli_model.to(self.device)

                 # Config'den sınıf isimlerini al
                zararli_config = self.config_manager.get_model_config('zararli')
                if zararli_config and hasattr(zararli_config, 'classes'):
                    self.zararli_class_names = {int(k): v.display_name
                                              for k, v in zararli_config.classes.items()}
                else:
                    self.zararli_class_names = self.zararli_model.names
                
                # Zararlı model sınıf isimlerini logla
            if self.zararli_class_names:
                logger.info(f"Zararlı modeli yüklendi. Sınıf sayısı: {len(self.zararli_class_names)}")
                logger.info(f"Zararlı model sınıfları: {list(self.zararli_class_names.values())}")
            else:
                logger.info(f"Zararlı modeli yüklendi fakat sınıf isimleri yüklenemedi.")
            
            # Models dictionary'e ekle
            self.models['zararli'] = {
                'model': self.zararli_model,
                'config': zararli_config
            }
            
            # Domates Mineral model yükleme
            if self.enable_domatesMineral and self.domatesMineral_model_path:
                logger.info(f"Domates Mineral modeli yükleniyor: {self.domatesMineral_model_path}")
                self.domatesMineral_model = YOLO(self.domatesMineral_model_path)
                self.domatesMineral_model.to(self.device)
                
                logger.info(f"Domates Mineral modeli yüklendi. Sınıf sayısı: {len(self.domatesMineral_model.names)}")
                
                # Models dictionary'e ekle
                self.models['domatesMineral'] = {
                    'model': self.domatesMineral_model,
                    'config': None
                }
            
            # Domates Hastalık model yükleme
            if self.enable_domatesHastalik and self.domatesHastalik_model_path:
                logger.info(f"Domates Hastalık modeli yükleniyor: {self.domatesHastalik_model_path}")
                self.domatesHastalik_model = YOLO(self.domatesHastalik_model_path)
                self.domatesHastalik_model.to(self.device)
                
                logger.info(f"Domates Hastalık modeli yüklendi. Sınıf sayısı: {len(self.domatesHastalik_model.names)}")
                
                # Models dictionary'e ekle
                self.models['domatesHastalik'] = {
                    'model': self.domatesHastalik_model,
                    'config': None
                }
            
            # Domates Olgunluk model yükleme
            if self.enable_domatesOlgunluk and hasattr(self, 'domatesOlgunluk_model_path') and self.domatesOlgunluk_model_path:
                logger.info(f"Domates Olgunluk modeli yükleniyor: {self.domatesOlgunluk_model_path}")
                self.domatesOlgunluk_model = YOLO(self.domatesOlgunluk_model_path)
                self.domatesOlgunluk_model.to(self.device)
                
                logger.info(f"Domates Olgunluk modeli yüklendi. Sınıf sayısı: {len(self.domatesOlgunluk_model.names)}")
                
                # Models dictionary'e ekle
                self.models['domatesOlgunluk'] = {
                    'model': self.domatesOlgunluk_model,
                    'config': None
                }
            
            # Debug: Class names'leri logla
            if self.general_class_names:
                logger.info(f"Genel model sınıfları: {list(self.general_class_names.values())[:10]}...")
            if self.farm_class_names:
                logger.info(f"Farm model sınıfları: {list(self.farm_class_names.values())}")
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            raise
    
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> MultiModelDetectionResult:
        """
        Frame üzerinde multi-model nesne tespiti yap
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame ID
            
        Returns:
            MultiModelDetectionResult: Tespit sonuçları
        """
        start_time = time.time()
        
        all_detections = []
        model_detections = {}
        processing_times = {}
        
        try:
            with self._lock:
                # Debug: Mevcut modelleri logla
                logger.info(f"Mevcut modeller: {list(self.models.keys())}")
                logger.info(f"Enable flags - General: {self.enable_general}, Farm: {self.enable_farm}")
                
                # Her aktif model için inference çalıştır
                for model_name, model_data in self.models.items():
                    logger.info(f"Model {model_name} işleniyor...")
                    
                    # Model aktif mi kontrolü
                    is_active = False
                    if model_name == 'general':
                        is_active = self.enable_general
                    elif model_name == 'farm':
                        is_active = self.enable_farm
                    elif model_name == 'zararli':
                        is_active = self.enable_zararli
                    elif model_name == 'domatesMineral':
                        is_active = self.enable_domatesMineral
                    elif model_name == 'domatesHastalik':
                        is_active = self.enable_domatesHastalik
                    elif model_name == 'domatesOlgunluk':
                        is_active = self.enable_domatesOlgunluk
                    
                    if not is_active:
                        logger.warning(f"Model {model_name} devre dışı")
                        # Devre dışı modeller için boş liste ata
                        model_detections[model_name] = []
                        continue
                    
                    logger.info(f"Model {model_name} için inference çalıştırılıyor")
                    
                    model_start = time.time()
                    
                    # Model inference
                    detections = self._run_inference(
                        model_data['model'],
                        frame,
                        model_name,
                        model_data.get('config')
                    )
                    
                    model_time = time.time() - model_start
                    processing_times[model_name] = model_time
                    
                    # Model detections'ları sakla
                    model_detections[model_name] = detections
                    all_detections.extend(detections)
                    
                    logger.info(f"Model {model_name}: {len(detections)} tespit bulundu, süre: {model_time:.3f}s")
                    
                    # Stats güncelle
                    if model_name in self.model_stats:
                        self.model_stats[model_name]['frames'] += 1
                        self.model_stats[model_name]['total_time'] += model_time
                        self.model_stats[model_name]['last_detection_count'] = len(detections)
                
                total_processing_time = time.time() - start_time
                processing_times['total'] = total_processing_time
                
                # Performance tracking
                self.frame_count += 1
                self.total_inference_time += total_processing_time
                
                # Legacy format için backward compatibility
                general_detections = model_detections.get('general', [])
                farm_detections = model_detections.get('farm', [])
                zararli_detections = model_detections.get('zararli', [])
                domatesMineral_detections = model_detections.get('domatesMineral', [])
                domatesHastalik_detections = model_detections.get('domatesHastalik', [])
                domatesOlgunluk_detections = model_detections.get('domatesOlgunluk', [])
                
                return MultiModelDetectionResult(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    general_detections=general_detections,
                    farm_detections=farm_detections,
                    zararli_detections=zararli_detections,
                    domatesMineral_detections=domatesMineral_detections,
                    domatesHastalik_detections=domatesHastalik_detections,
                    domatesOlgunluk_detections=domatesOlgunluk_detections,
                    all_detections=all_detections,
                    processing_times=processing_times,
                    frame_size=(frame.shape[1], frame.shape[0])
                )
                
        except Exception as e:
            logger.error(f"Multi-model detection hatası: {e}")
            return MultiModelDetectionResult(
                frame_id=frame_id,
                timestamp=time.time(),
                general_detections=[],
                farm_detections=[],
                zararli_detections=[],
                domatesMineral_detections=[],
                domatesHastalik_detections=[],
                domatesOlgunluk_detections=[],
                all_detections=[],
                processing_times={'total': time.time() - start_time},
                frame_size=(frame.shape[1], frame.shape[0])
            )
    
    def _run_inference(self, model, frame: np.ndarray, model_name: str, model_config=None):
        """Tek model inference"""
        detections = []
        
        try:
            # Model config'den confidence threshold al
            conf_threshold = model_config.confidence_threshold if model_config else 0.5
            
            # Inference yap
            results = model(frame, verbose=False)[0]  # conf parametresini kaldırdık
            
            # Debug: Model inference sonuçlarını göster
            logger.info(f"Model {model_name} inference sonuçları:")
            logger.info(f"- Tespit sayısı: {len(results.boxes)}")
            logger.info(f"- Ham sonuçlar: {results.boxes.data.tolist()}")
            
            # Tespitleri listeye dönüştür
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = result
                
                # Debug: Her tespit için detayları göster
                logger.info(f"Tespit detayları ({model_name}):")
                logger.info(f"- Koordinatlar: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                logger.info(f"- Güven: {confidence:.2f}, Sınıf ID: {class_id}")
                
                # Güven eşiği kontrolü
                if confidence < conf_threshold:
                    logger.info(f"- Güven eşiği altında ({confidence:.2f} < {conf_threshold}), atlanıyor")
                    continue
                
                # Koordinatları normalize et ve geçerlilik kontrolü yap
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(frame.shape[1], int(x2))
                y2 = min(frame.shape[0], int(y2))
                
                # Debug: Normalize edilmiş koordinatlar
                logger.info(f"- Normalize edilmiş koordinatlar: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                
                # Geçersiz bounding box kontrolü
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Geçersiz bounding box atlandı: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                # Minimum boyut kontrolü (çok küçük bounding box'ları filtrele)
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    logger.info(f"- Çok küçük bounding box atlandı: {x2-x1}x{y2-y1} < 10x10")
                    continue
                
                logger.info(f"- Geçerli bounding box: {x2-x1}x{y2-y1} piksel")
                
                # Class bilgisi ve koordinatları dönüştür
                class_id = int(float(class_id))  # Önce float'a çevir, sonra int'e yuvarla
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                confidence = float(confidence)
                
                # Model tipine göre sınıf ismini al
                if model_name == 'farm':
                    class_names = self.farm_class_names
                elif model_name == 'general':
                    class_names = self.general_class_names
                elif model_name == 'zararli':
                    class_names = self.zararli_class_names
                    logger.debug(f"Zararlı sınıf isimleri: {class_names}")
                    # Zararlı model için sınıf ID'sini düzelt
                    if class_id >= len(model.names):
                        logger.warning(f"Geçersiz sınıf ID: {class_id}, varsayılan 0 kullanılıyor")
                        class_id = 0
                else:
                    # Domates mineral, hastalık, olgunluk modelleri için direkt model.names kullan
                    class_names = {}
                
                # Sınıf ismini al
                if model_name == 'zararli':
                    # Zararlı model için özel sınıf ismi atama
                    if class_id in self.zararli_class_names:
                        class_name = self.zararli_class_names[class_id]
                    else:
                        class_name = f"zararli_{class_id}"
                    logger.debug(f"Zararlı model için sınıf {class_id} -> {class_name}")
                else:
                    # Diğer modeller için normal sınıf ismi atama
                    if class_names and class_id in class_names:
                        class_name = class_names[class_id]
                    else:
                        class_name = model.names[class_id] if class_id in model.names else f"unknown_{class_id}"
                    logger.debug(f"Model {model_name} için sınıf {class_id} -> {class_name}")
                
                # Detection objesi oluştur
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    area=(x2 - x1) * (y2 - y1),
                    model_type=ModelType.ZARARLI if model_name == 'zararli'
                              else ModelType.GENERAL if model_name == 'general'
                              else ModelType.FARM
                )
                
                detections.append(detection)
        
        except Exception as e:
            logger.error(f"Model inference hatası ({model_name}): {e}")
            return []
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detection_result: MultiModelDetectionResult, show_model_type: bool = True):
        """
        Frame üzerine config-driven multi-model tespit sonuçlarını çiz
        
        Args:
            frame: Input frame
            detection_result: Multi-model detection sonuçları
            show_model_type: Model tipini label'da göster
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Visualization ayarlarını config'ten al
        bbox_settings = self.config_manager.get_visualization_setting('bbox', {})
        thickness = bbox_settings.get('thickness', 2)
        font_scale = bbox_settings.get('font_scale', 0.6)
        font_thickness = bbox_settings.get('font_thickness', 2)
        show_confidence = bbox_settings.get('show_confidence', True)
        show_class_name = bbox_settings.get('show_class_name', True)
        
        # Farm detectionları çiz (yeşil)
        for detection in detection_result.farm_detections:
            try:
                x1, y1, x2, y2 = detection.bbox
                
                # Koordinatları frame sınırları içinde tut (genel model ile aynı yaklaşım)
                x1 = max(0, min(frame.shape[1] - 1, int(x1)))
                y1 = max(0, min(frame.shape[0] - 1, int(y1)))
                x2 = max(0, min(frame.shape[1], int(x2)))
                y2 = max(0, min(frame.shape[0], int(y2)))
                
                # Geçersiz koordinat kontrolü
                if x2 <= x1 or y2 <= y1:
                    continue
                
                color = (50, 205, 50)  # BGR format - Açık yeşil
                
                # Bounding box çiz
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Label hazırla
                label_parts = []
                if show_class_name:
                    label_parts.append(detection.class_name)
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")
                if show_model_type:
                    label_parts.append("(farm)")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            except Exception as e:
                logger.error(f"Farm detection çizim hatası: {e}")
                continue
        
        # General detectionları çiz (mavi)
        for detection in detection_result.general_detections:
            try:
                x1, y1, x2, y2 = detection.bbox
                
                # Koordinatları frame sınırları içinde tut
                x1 = max(0, min(frame.shape[1] - 1, int(x1)))
                y1 = max(0, min(frame.shape[0] - 1, int(y1)))
                x2 = max(0, min(frame.shape[1], int(x2)))
                y2 = max(0, min(frame.shape[0], int(y2)))
                
                # Geçersiz koordinat kontrolü
                if x2 <= x1 or y2 <= y1:
                    continue
                
                color = (255, 128, 0)  # BGR format - Turuncu-mavi karışımı
                
                # Bounding box çiz
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Label hazırla
                label_parts = []
                if show_class_name:
                    label_parts.append(detection.class_name)
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")
                if show_model_type:
                    label_parts.append("(general)")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            except Exception as e:
                logger.error(f"General detection çizim hatası: {e}")
                continue
        
        # Zararlı detectionları çiz (kırmızı)
        for detection in detection_result.zararli_detections:
            try:
                x1, y1, x2, y2 = detection.bbox
                
                # Koordinatları frame sınırları içinde tut
                x1 = max(0, min(frame.shape[1] - 1, int(x1)))
                y1 = max(0, min(frame.shape[0] - 1, int(y1)))
                x2 = max(0, min(frame.shape[1], int(x2)))
                y2 = max(0, min(frame.shape[0], int(y2)))
                
                # Geçersiz koordinat kontrolü
                if x2 <= x1 or y2 <= y1:
                    continue
                
                color = (0, 0, 255)  # BGR format - Kırmızı
                
                # Bounding box çiz
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Label hazırla
                label_parts = []
                if show_class_name:
                    label_parts.append(detection.class_name)
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")
                if show_model_type:
                    label_parts.append("(zararlı)")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            except Exception as e:
                logger.error(f"Zararlı detection çizim hatası: {e}")
                continue
        
        # Domates Mineral detectionları çiz (mor)
        for detection in detection_result.domatesMineral_detections:
            try:
                x1, y1, x2, y2 = detection.bbox
                
                # Koordinatları frame sınırları içinde tut
                x1 = max(0, min(frame.shape[1] - 1, int(x1)))
                y1 = max(0, min(frame.shape[0] - 1, int(y1)))
                x2 = max(0, min(frame.shape[1], int(x2)))
                y2 = max(0, min(frame.shape[0], int(y2)))
                
                # Geçersiz koordinat kontrolü
                if x2 <= x1 or y2 <= y1:
                    continue
                
                color = (255, 0, 255)  # BGR format - Mor
                
                # Bounding box çiz
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Label hazırla
                label_parts = []
                if show_class_name:
                    label_parts.append(detection.class_name)
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")
                if show_model_type:
                    label_parts.append("(mineral)")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            except Exception as e:
                logger.error(f"Domates Mineral detection çizim hatası: {e}")
                continue
        
        # Domates Hastalık detectionları çiz (sarı)
        for detection in detection_result.domatesHastalik_detections:
            try:
                x1, y1, x2, y2 = detection.bbox
                
                # Koordinatları frame sınırları içinde tut
                x1 = max(0, min(frame.shape[1] - 1, int(x1)))
                y1 = max(0, min(frame.shape[0] - 1, int(y1)))
                x2 = max(0, min(frame.shape[1], int(x2)))
                y2 = max(0, min(frame.shape[0], int(y2)))
                
                # Geçersiz koordinat kontrolü
                if x2 <= x1 or y2 <= y1:
                    continue
                
                color = (0, 255, 255)  # BGR format - Sarı
                
                # Bounding box çiz
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Label hazırla
                label_parts = []
                if show_class_name:
                    label_parts.append(detection.class_name)
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")
                if show_model_type:
                    label_parts.append("(hastalık)")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            except Exception as e:
                logger.error(f"Domates Hastalık detection çizim hatası: {e}")
                continue
        
        # Domates Olgunluk detectionları çiz (cyan)
        for detection in detection_result.domatesOlgunluk_detections:
            try:
                x1, y1, x2, y2 = detection.bbox
                
                # Koordinatları frame sınırları içinde tut
                x1 = max(0, min(frame.shape[1] - 1, int(x1)))
                y1 = max(0, min(frame.shape[0] - 1, int(y1)))
                x2 = max(0, min(frame.shape[1], int(x2)))
                y2 = max(0, min(frame.shape[0], int(y2)))
                
                # Geçersiz koordinat kontrolü
                if x2 <= x1 or y2 <= y1:
                    continue
                
                color = (255, 255, 0)  # BGR format - Cyan
                
                # Bounding box çiz
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Label hazırla
                label_parts = []
                if show_class_name:
                    label_parts.append(detection.class_name)
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")
                if show_model_type:
                    label_parts.append("(olgunluk)")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            except Exception as e:
                logger.error(f"Domates Olgunluk detection çizim hatası: {e}")
                continue
        
        # Performance bilgilerini çiz
        self._draw_performance_info(annotated_frame, detection_result)
        
        return annotated_frame
    
    def _draw_performance_info(self, frame: np.ndarray, detection_result: MultiModelDetectionResult):
        """Frame üzerine performance bilgilerini çiz"""
        performance_settings = self.config_manager.get_visualization_setting('performance', {})
        position = performance_settings.get('position', 'top_left')
        
        # Position'a göre koordinatları belirle
        if position == 'top_left':
            x, y = 10, 30
        elif position == 'top_right':
            x, y = frame.shape[1] - 300, 30
        elif position == 'bottom_left':
            x, y = 10, frame.shape[0] - 100
        else:  # bottom_right
            x, y = frame.shape[1] - 300, frame.shape[0] - 100
        
        # Performance text'i çiz
        fps = 1.0 / detection_result.processing_times.get('total', 0.001)
        perf_text = f"FPS: {fps:.1f} | Total: {len(detection_result.all_detections)}"
        
        cv2.putText(frame, perf_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_statistics(self) -> Dict:
        """İstatistikleri döndür"""
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_frames': self.frame_count,
            'elapsed_time': elapsed_time,
            'average_fps': avg_fps,
            'models': self.model_stats
        }
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            'general': {
                'enabled': self.enable_general,
                'path': self.general_model_path,
                'classes': len(self.general_class_names)
            },
            'farm': {
                'enabled': self.enable_farm,
                'path': self.farm_model_path,
                'classes': len(self.farm_class_names)
            },
            'zararli': {
                'enabled': self.enable_zararli,
                'path': self.zararli_model_path,
                'classes': len(self.zararli_model.names) if self.zararli_model else 0
            },
            'domatesMineral': {
                'enabled': self.enable_domatesMineral,
                'path': self.domatesMineral_model_path,
                'classes': len(self.domatesMineral_model.names) if self.domatesMineral_model else 0
            },
            'domatesHastalik': {
                'enabled': self.enable_domatesHastalik,
                'path': self.domatesHastalik_model_path,
                'classes': len(self.domatesHastalik_model.names) if self.domatesHastalik_model else 0
            },
            'domatesOlgunluk': {
                'enabled': self.enable_domatesOlgunluk,
                'path': self.domatesOlgunluk_model_path,
                'classes': len(self.domatesOlgunluk_model.names) if self.domatesOlgunluk_model else 0
            }
        }
    
    def set_model_enabled(self, model_type: str, enabled: bool):
        """Model durumunu değiştir"""
        if model_type == 'general':
            self.enable_general = enabled
        elif model_type == 'farm':
            self.enable_farm = enabled
        elif model_type == 'zararli':
            self.enable_zararli = enabled
        elif model_type == 'domatesMineral':
            self.enable_domatesMineral = enabled
        elif model_type == 'domatesHastalik':
            self.enable_domatesHastalik = enabled
        elif model_type == 'domatesOlgunluk':
            self.enable_domatesOlgunluk = enabled
        else:
            logger.warning(f"Bilinmeyen model tipi: {model_type}")
            return False
        return True
    
    def update_confidence_threshold(self, threshold: float):
        """Güven eşiği güncelle"""
        # Config manager üzerinden güncelle
        for model_name in self.models:
            model_config = self.config_manager.get_model_config(model_name)
            if model_config:
                model_config.confidence_threshold = threshold
    
    def reset_statistics(self):
        """İstatistikleri sıfırla"""
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        for model_name in self.model_stats:
            self.model_stats[model_name] = {'frames': 0, 'total_time': 0.0, 'last_detection_count': 0}