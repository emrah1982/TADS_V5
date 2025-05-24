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
    all_detections: List[Detection]      # Tüm sonuçlar birleşik
    processing_times: Dict[str, float]   # Model bazlı işleme süreleri
    frame_size: Tuple[int, int]

class MultiModelYOLODetector:
    """Multi-model YOLO detector sınıfı"""
    
    def __init__(self, farm_model_path: str, general_model_path: str):
        """
        Config-driven Multi-model YOLO detector'ı başlat
        
        Args:
            farm_model_path: Farm model dosya yolu
            general_model_path: Genel model dosya yolu
        """
        # Model paths
        self.farm_model_path = farm_model_path
        self.general_model_path = general_model_path
        
        # Model instances
        self.farm_model = None
        self.general_model = None
        
        # Model enable flags
        self.enable_general = True
        self.enable_farm = True
        
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
            'general': {'frames': 0, 'total_time': 0.0, 'last_detection_count': 0}
        }
        
        # Class names
        self.general_class_names = {}
        self.farm_class_names = {}
        
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
                self.general_class_names = self.general_model.names
                logger.info(f"Genel model yüklendi. Sınıf sayısı: {len(self.general_class_names)}")
                
                # Models dictionary'e ekle
                self.models['general'] = {
                    'model': self.general_model,
                    'config': self.config_manager.get_model_config('general') if hasattr(self, 'config_manager') else None
                }
            
            # Farm model yükleme
            if self.enable_farm and self.farm_model_path:
                logger.info(f"Farm modeli yükleniyor: {self.farm_model_path}")
                self.farm_model = YOLO(self.farm_model_path)
                self.farm_model.to(self.device)
                self.farm_class_names = self.farm_model.names
                logger.info(f"Farm modeli yüklendi. Sınıf sayısı: {len(self.farm_class_names)}")
                
                # Models dictionary'e ekle
                self.models['farm'] = {
                    'model': self.farm_model,
                    'config': self.config_manager.get_model_config('farm') if hasattr(self, 'config_manager') else None
                }
            
            # Debug: Class names'leri logla
            if self.general_class_names:
                logger.info(f"Genel model sınıfları: {list(self.general_class_names.values())[:10]}...")  # İlk 10 sınıf
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
                # Her aktif model için inference çalıştır
                for model_name, model_data in self.models.items():
                    # Model config kontrolü
                    if model_data.get('config') and not model_data['config'].enabled:
                        continue
                    
                    # Model aktif mi kontrolü (legacy)
                    if model_name == 'general' and not self.enable_general:
                        continue
                    if model_name == 'farm' and not self.enable_farm:
                        continue
                    
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
                
                return MultiModelDetectionResult(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    general_detections=general_detections,
                    farm_detections=farm_detections,
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
                all_detections=[],
                processing_times={'total': time.time() - start_time},
                frame_size=(frame.shape[1], frame.shape[0])
            )
    
    def _run_inference(self, model, frame: np.ndarray, model_name: str, model_config=None) -> List[Detection]:
        """Tek model inference"""
        try:
            # Confidence threshold belirle
            conf_threshold = 0.5  # Default
            if model_config and hasattr(model_config, 'confidence_threshold'):
                conf_threshold = model_config.confidence_threshold
            
            # Model inference
            results = model(frame, conf=conf_threshold, verbose=False)
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Bounding box koordinatları
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    
                    # Confidence ve class
                    conf = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Config varsa sınıf kontrolü yap
                    if hasattr(self, 'config_manager') and self.config_manager:
                        # Config'ten sınıf bilgilerini al
                        class_config = self.config_manager.get_class_config(model_name, class_id)
                        
                        # Sınıf aktif mi kontrol et
                        if not self.config_manager.is_class_enabled(model_name, class_id):
                            continue
                        
                        # Class name (config'ten)
                        if class_config:
                            class_name = class_config.display_name
                        else:
                            # Fallback to model's class names
                            model_class_names = model.names if hasattr(model, 'names') else {}
                            class_name = model_class_names.get(class_id, f"Class_{class_id}")
                    else:
                        # Config yoksa direkt model'den al
                        model_class_names = model.names if hasattr(model, 'names') else {}
                        class_name = model_class_names.get(class_id, f"Class_{class_id}")
                    
                    # Center point ve area
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Model type
                    model_type_enum = ModelType.GENERAL if model_name == 'general' else ModelType.FARM
                    if model_name not in ['general', 'farm']:
                        # Custom model için generic type
                        model_type_enum = ModelType.GENERAL
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        area=area,
                        model_type=model_type_enum
                    )
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference hatası ({model_name}): {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, 
                       detection_result: MultiModelDetectionResult,
                       show_model_type: bool = True) -> np.ndarray:
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
        
        label_settings = self.config_manager.get_visualization_setting('labels', {})
        show_turkish_names = label_settings.get('show_turkish_names', True)
        show_confidence_percentage = label_settings.get('show_confidence_percentage', True)
        text_color = tuple(label_settings.get('text_color', [255, 255, 255]))
        
        # Detection'ları model priority'sine göre sırala
        sorted_detections = sorted(
            detection_result.all_detections,
            key=lambda d: self._get_model_priority(d)
        )
        
        for detection in sorted_detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Config'ten renk al
            color = self._get_color_for_detection(detection)
            
            # Bounding box çiz
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label oluştur
            label_parts = []
            
            if show_model_type:
                model_name = self._get_model_name_from_detection(detection)
                model_config = self.config_manager.get_model_config(model_name)
                if model_config:
                    label_parts.append(f"[{model_config.icon}]")
            
            if show_class_name:
                if show_turkish_names:
                    label_parts.append(detection.class_name)
                else:
                    # Original class name'i al
                    class_config = self._get_class_config_from_detection(detection)
                    original_name = class_config.name if class_config else detection.class_name
                    label_parts.append(original_name)
            
            if show_confidence and show_confidence_percentage:
                label_parts.append(f"{detection.confidence:.2f}")
            
            label = " ".join(label_parts)
            
            # Label background size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Label background çiz
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Label text çiz
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness
            )
            
            # Center point çiz
            cv2.circle(annotated_frame, detection.center, 3, color, -1)
        
        # Performance info çiz
        if self.config_manager.get_visualization_setting('performance', {}).get('show_fps', True):
            self._draw_performance_info(annotated_frame, detection_result)
        
        return annotated_frame
    
    def _get_color_for_detection(self, detection: Detection) -> Tuple[int, int, int]:
        """Config'ten detection rengi döndür"""
        model_name = self._get_model_name_from_detection(detection)
        color = self.config_manager.get_class_color(model_name, detection.class_id)
        
        # BGR formatına çevir (OpenCV için)
        return (int(color[2]), int(color[1]), int(color[0]))
    
    def _get_model_name_from_detection(self, detection: Detection) -> str:
        """Detection'dan model adını çıkar"""
        # Model type'a göre model adını belirle
        if detection.model_type == ModelType.GENERAL:
            return 'general'
        elif detection.model_type == ModelType.FARM:
            return 'farm'
        else:
            # İlk bulunan aktif modeli döndür (fallback)
            for model_name in self.models.keys():
                return model_name
            return 'unknown'
    
    def _get_class_config_from_detection(self, detection: Detection) -> Optional[ClassConfig]:
        """Detection'dan class config'i al"""
        model_name = self._get_model_name_from_detection(detection)
        return self.config_manager.get_class_config(model_name, detection.class_id)
    
    def _get_model_priority(self, detection: Detection) -> int:
        """Detection'ın model priority'sini döndür"""
        model_name = self._get_model_name_from_detection(detection)
        model_config = self.config_manager.get_model_config(model_name)
        return model_config.priority if model_config else 999
    
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
        
        y_offset = 0
        
        # Model bazlı bilgiler
        for model_name, model_data in self.models.items():
            if not model_data['config'].enabled:
                continue
                
            model_config = model_data['config']
            detection_count = len([d for d in detection_result.all_detections 
                                 if self._get_model_name_from_detection(d) == model_name])
            processing_time = detection_result.processing_times.get(model_name, 0)
            
            text = f"{model_config.icon} {model_config.display_name}: {detection_count} ({processing_time*1000:.1f}ms)"
            
            # Model'e göre renk
            if model_name == 'general':
                color = (255, 255, 255)  # Beyaz
            elif model_name == 'farm':
                color = (0, 255, 255)    # Sarı
            else:
                color = (255, 255, 255)  # Beyaz (default)
            
            cv2.putText(frame, text, (x, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            y_offset += 25
        
        # Total bilgi
        total_time = detection_result.processing_times.get('total', 0)
        fps = 1.0 / total_time if total_time > 0 else 0
        text = f"Total: {len(detection_result.all_detections)} ({total_time*1000:.1f}ms | {fps:.1f} FPS)"
        cv2.putText(frame, text, (x, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2)
    
    def get_statistics(self) -> Dict:
        """Config-driven performance istatistiklerini döndür"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        stats = {
            "overall": {
                "frames_processed": self.frame_count,
                "fps": self.frame_count / elapsed_time if elapsed_time > 0 else 0,
                "avg_total_inference_time": self.total_inference_time / self.frame_count if self.frame_count > 0 else 0,
                "total_time": elapsed_time,
                "device": self.device,
                "active_models": len(self.models)
            },
            "models": {}
        }
        
        # Model bazlı istatistikler
        for model_name, model_data in self.models.items():
            model_config = model_data['config']
            model_stat = self.model_stats[model_name]
            
            if model_stat['frames'] > 0:
                stats["models"][model_name] = {
                    "enabled": model_config.enabled,
                    "frames_processed": model_stat['frames'],
                    "avg_inference_time": model_stat['total_time'] / model_stat['frames'],
                    "total_inference_time": model_stat['total_time'],
                    "model_path": model_config.model_path,
                    "display_name": model_config.display_name,
                    "class_count": len(model_config.classes),
                    "enabled_class_count": len(self.config_manager.get_enabled_classes(model_name)),
                    "last_detection_count": model_stat['last_detection_count']
                }
        
        return stats
    
    def get_model_info(self) -> Dict:
        """Config-driven model bilgilerini döndür"""
        models_info = {}
        
        for model_name, model_data in self.models.items():
            model_config = model_data['config']
            
            models_info[model_name] = {
                "enabled": model_config.enabled,
                "path": model_config.model_path,
                "display_name": model_config.display_name,
                "description": model_config.description,
                "icon": model_config.icon,
                "confidence_threshold": model_config.confidence_threshold,
                "loaded": True,
                "classes": {
                    class_id: {
                        "name": class_config.name,
                        "display_name": class_config.display_name,
                        "enabled": class_config.enabled,
                        "alert": class_config.alert,
                        "color": class_config.color
                    }
                    for class_id, class_config in model_config.classes.items()
                }
            }
        
        # Config'de tanımlı ama yüklenmemiş modeller
        all_model_configs = self.config_manager.get_all_model_configs()
        for model_name, model_config in all_model_configs.items():
            if model_name not in models_info:
                models_info[model_name] = {
                    "enabled": model_config.enabled,
                    "path": model_config.model_path,
                    "display_name": model_config.display_name,
                    "description": model_config.description,
                    "icon": model_config.icon,
                    "confidence_threshold": model_config.confidence_threshold,
                    "loaded": False,
                    "error": "Model dosyası bulunamadı veya yüklenemedi"
                }
        
        return {
            "device": self.device,
            "total_models": len(all_model_configs),
            "loaded_models": len(self.models),
            "models": models_info
        }
    
    def set_model_enabled(self, model_name: str, enabled: bool):
        """Config'ten model'i aktif/pasif yap"""
        self.config_manager.set_model_enabled(model_name, enabled)
    
    def update_confidence_threshold(self, model_name: str, threshold: float):
        """Config'ten confidence threshold'unu güncelle"""
        self.config_manager.set_confidence_threshold(model_name, threshold)
    
    def set_class_enabled(self, model_name: str, class_id: int, enabled: bool):
        """Config'ten sınıfı aktif/pasif et"""
        self.config_manager.set_class_enabled(model_name, class_id, enabled)
    
    def set_class_color(self, model_name: str, class_id: int, color: Tuple[int, int, int]):
        """Config'ten sınıf rengini ayarla"""
        self.config_manager.set_class_color(model_name, class_id, color)
    
    def _on_config_change(self, change_type: str, key: str, value):
        """Config değişikliği callback'i"""
        logger.info(f"Config değişikliği: {change_type} - {key} = {value}")
        
        if change_type == 'model_enabled':
            # Model aktif/pasif durumu değişti
            if value:  # Model aktif edildi
                self._load_single_model(key)
            else:  # Model pasif edildi
                self._unload_single_model(key)
        
        elif change_type == 'config_reloaded':
            # Config yeniden yüklendi, tüm modelleri yeniden yükle
            logger.info("Config yeniden yüklendiği için modeller yeniden yükleniyor...")
            self._reload_all_models()
    
    def _load_single_model(self, model_name: str):
        """Tek bir modeli yükle"""
        try:
            model_config = self.config_manager.get_model_config(model_name)
            if not model_config or not model_config.enabled:
                return
            
            if model_name in self.models:
                logger.info(f"Model zaten yüklü: {model_name}")
                return
            
            logger.info(f"Model yükleniyor: {model_name}")
            
            if not self.config_manager.validate_model_path(model_name):
                logger.error(f"Model dosyası bulunamadı: {model_config.model_path}")
                return
            
            model = YOLO(model_config.model_path)
            model.to(self.device)
            
            self.models[model_name] = {
                'model': model,
                'config': model_config
            }
            
            self.model_stats[model_name] = {
                'frames': 0,
                'total_time': 0.0,
                'last_detection_count': 0
            }
            
            logger.info(f"Model başarıyla yüklendi: {model_name}")
            
        except Exception as e:
            logger.error(f"Model yükleme hatası ({model_name}): {e}")
    
    def _unload_single_model(self, model_name: str):
        """Tek bir modeli kaldır"""
        try:
            if model_name in self.models:
                del self.models[model_name]
                if model_name in self.model_stats:
                    del self.model_stats[model_name]
                logger.info(f"Model kaldırıldı: {model_name}")
            
        except Exception as e:
            logger.error(f"Model kaldırma hatası ({model_name}): {e}")
    
    def _reload_all_models(self):
        """Tüm modelleri yeniden yükle"""
        try:
            # Mevcut modelleri temizle
            self.models.clear()
            self.model_stats.clear()
            
            # Modelleri yeniden yükle
            self._load_models()
            
            logger.info("Tüm modeller yeniden yüklendi")
            
        except Exception as e:
            logger.error(f"Modelleri yeniden yükleme hatası: {e}")
    
    def get_config_summary(self) -> Dict:
        """Config özetini döndür"""
        return self.config_manager.get_config_summary()
    
    def reload_config(self):
        """Config'i yeniden yükle"""
        self.config_manager.reload_config()
    
    def export_config(self, format: str = "yaml", file_path: str = None) -> str:
        """Config'i export et"""
        return self.config_manager.export_config(format, file_path)
    
    def import_config(self, file_path: str, merge: bool = False):
        """Config'i import et"""
        self.config_manager.import_config(file_path, merge)
    
    # Legacy methods for backward compatibility
    def get_class_names(self) -> Dict[int, str]:
        """Legacy: Sınıf isimlerini döndür"""
        all_classes = {}
        for model_name in self.models.keys():
            model_config = self.config_manager.get_model_config(model_name)
            if model_config:
                for class_id, class_config in model_config.classes.items():
                    all_classes[class_id] = class_config.display_name
        return all_classes
    
    def reset_statistics(self):
        """İstatistikleri sıfırla"""
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        for model_name in self.model_stats:
            self.model_stats[model_name] = {
                'frames': 0,
                'total_time': 0.0,
                'last_detection_count': 0
            }
    
    
    def _get_color_for_detection(self, detection: Detection) -> Tuple[int, int, int]:
        """Detection için renk döndür"""
        # Basit renk şeması
        if detection.model_type == ModelType.GENERAL:
            # Genel model için mavi tonları
            base_color = 200
            variation = (detection.class_id * 30) % 55
            return (base_color + variation, variation, variation)
        else:  # FARM
            # Farm model için yeşil tonları
            base_color = 200
            variation = (detection.class_id * 30) % 55
            return (variation, base_color + variation, variation)
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Genel model için renk paleti oluştur"""
        colors = []
        for i in range(num_colors):
            hue = int(360 * i / num_colors)
            # HSV'den BGR'ye dönüştür
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr)))
        return colors
    
    def _generate_farm_colors(self) -> List[Tuple[int, int, int]]:
        """Farm model için özel renk paleti"""
        # Tarım teması renkleri
        farm_colors = [
            (0, 255, 0),     # Yeşil - sağlıklı
            (0, 165, 255),   # Turuncu - mineral eksikliği
            (0, 0, 255),     # Kırmızı - hastalık
            (0, 255, 255),   # Sarı - zararlı
            (255, 0, 255),   # Magenta - olgunluk
            (255, 255, 0),   # Cyan - diğer
            (128, 0, 128),   # Mor
            (255, 165, 0),   # Mavi
            (0, 128, 255),   # Açık kırmızı
            (128, 255, 0),   # Açık yeşil
        ]
        return farm_colors
    
    def _draw_model_info(self, frame: np.ndarray, 
                        detection_result: MultiModelDetectionResult):
        """Frame üzerine model bilgilerini çiz"""
        y_offset = 30
        
        # General model bilgileri
        if self.enable_general:
            general_count = len(detection_result.general_detections)
            general_time = detection_result.processing_times.get('general', 0)
            text = f"General: {general_count} detections ({general_time*1000:.1f}ms)"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Farm model bilgileri
        if self.enable_farm:
            farm_count = len(detection_result.farm_detections)
            farm_time = detection_result.processing_times.get('farm', 0)
            text = f"Farm: {farm_count} detections ({farm_time*1000:.1f}ms)"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2)
            y_offset += 25
        
        # Total bilgi
        total_time = detection_result.processing_times.get('total', 0)
        fps = 1.0 / total_time if total_time > 0 else 0
        text = f"Total: {len(detection_result.all_detections)} ({total_time*1000:.1f}ms | {fps:.1f} FPS)"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2)
    
    def get_statistics(self) -> Dict:
        """Multi-model performance istatistiklerini döndür"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        stats = {
            "overall": {
                "frames_processed": self.frame_count,
                "fps": self.frame_count / elapsed_time if elapsed_time > 0 else 0,
                "avg_total_inference_time": self.total_inference_time / self.frame_count if self.frame_count > 0 else 0,
                "total_time": elapsed_time,
                "device": self.device
            },
            "models": {}
        }
        
        # Model bazlı istatistikler
        if self.enable_general and self.model_stats['general']['frames'] > 0:
            general_stats = self.model_stats['general']
            stats["models"]["general"] = {
                "enabled": True,
                "frames_processed": general_stats['frames'],
                "avg_inference_time": general_stats['total_time'] / general_stats['frames'],
                "total_inference_time": general_stats['total_time'],
                "model_path": self.general_model_path,
                "class_count": len(self.general_class_names)
            }
        
        if self.enable_farm and self.model_stats['farm']['frames'] > 0:
            farm_stats = self.model_stats['farm']
            stats["models"]["farm"] = {
                "enabled": True,
                "frames_processed": farm_stats['frames'],
                "avg_inference_time": farm_stats['total_time'] / farm_stats['frames'],
                "total_inference_time": farm_stats['total_time'],
                "model_path": self.farm_model_path,
                "class_count": len(self.farm_class_names)
            }
        
        return stats
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        info = {}
        
        # General model bilgileri
        if 'general' in self.models:
            info['general'] = {
                "enabled": self.enable_general,
                "path": self.general_model_path,
                "loaded": True,
                "class_count": len(self.general_class_names) if self.general_class_names else 0
            }
        else:
            info['general'] = {
                "enabled": self.enable_general,
                "path": self.general_model_path,
                "loaded": False,
                "class_count": 0
            }
        
        # Farm model bilgileri
        if 'farm' in self.models:
            info['farm'] = {
                "enabled": self.enable_farm,
                "path": self.farm_model_path,
                "loaded": True,
                "class_count": len(self.farm_class_names) if self.farm_class_names else 0
            }
        else:
            info['farm'] = {
                "enabled": self.enable_farm,
                "path": self.farm_model_path,
                "loaded": False,
                "class_count": 0
            }
        
        return info
    
    def set_model_enabled(self, model_type: str, enabled: bool):
        """Model'i aktif/pasif yap"""
        if model_type == "general":
            self.enable_general = enabled
            logger.info(f"General model {'aktif' if enabled else 'pasif'}")
        elif model_type == "farm":
            self.enable_farm = enabled
            logger.info(f"Farm model {'aktif' if enabled else 'pasif'}")
        else:
            logger.error(f"Bilinmeyen model tipi: {model_type}")
    
    def update_confidence_threshold(self, model_name: str, threshold: float):
        """Güven eşiğini güncelle"""
        if 0.0 <= threshold <= 1.0:
            if hasattr(self, 'config_manager') and self.config_manager:
                self.config_manager.set_confidence_threshold(model_name, threshold)
            logger.info(f"{model_name} model güven eşiği güncellendi: {threshold}")
        else:
            raise ValueError("Güven eşiği 0.0 ile 1.0 arasında olmalıdır")
    
    def get_detections_by_model(self, detection_result: MultiModelDetectionResult, 
                               model_type: ModelType) -> List[Detection]:
        """Belirli model tipinin detection'larını döndür"""
        if model_type == ModelType.GENERAL:
            return detection_result.general_detections
        elif model_type == ModelType.FARM:
            return detection_result.farm_detections
        else:
            return []
    
    def get_detection_summary(self, detection_result: MultiModelDetectionResult) -> Dict:
        """Multi-model tespit özetini döndür"""
        summary = {
            "total_detections": len(detection_result.all_detections),
            "general_detections": len(detection_result.general_detections),
            "farm_detections": len(detection_result.farm_detections),
            "processing_times": detection_result.processing_times,
            "classes_detected": {}
        }
        
        # Model bazlı sınıf sayıları
        for detection in detection_result.all_detections:
            model_key = detection.model_type.value
            class_name = detection.class_name
            
            if model_key not in summary["classes_detected"]:
                summary["classes_detected"][model_key] = {}
            
            if class_name not in summary["classes_detected"][model_key]:
                summary["classes_detected"][model_key][class_name] = 0
            
            summary["classes_detected"][model_key][class_name] += 1
        
        return summary
    
    def reset_statistics(self):
        """İstatistikleri sıfırla"""
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        self.model_stats = {
            'general': {'frames': 0, 'total_time': 0.0},
            'farm': {'frames': 0, 'total_time': 0.0}
        }