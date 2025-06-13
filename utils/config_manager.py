import yaml
import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConfigLoadError(Exception):
    """Config yükleme hatası"""
    pass

class ConfigValidationError(Exception):
    """Config doğrulama hatası"""
    pass

@dataclass
class ClassConfig:
    """Sınıf konfigürasyon sınıfı"""
    name: str
    display_name: str
    color: Tuple[int, int, int]
    enabled: bool = True
    alert: bool = False
    confidence_threshold: float = None

@dataclass
class ModelConfig:
    """Model konfigürasyon sınıfı"""
    enabled: bool
    model_path: str
    model_type: str
    display_name: str
    description: str
    icon: str
    priority: int
    confidence_threshold: float
    classes: Dict[int, ClassConfig]

# Global config manager instance
_config_manager = None

def get_config_manager() -> 'ConfigManager':
    """Global config manager instance'ı döndür"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

class ConfigManager:
    """Configuration Manager sınıfı"""
    
    def _get_default_config(self) -> Dict:
        """Default config'i döndür"""
        return {
            "system": {
                "max_fps": 30,
                "detection_interval": 0.1,
                "save_detections": True,
                "save_interval": 300
            },
            "models": {
                "general": {
                    "enabled": True,
                    "model_path": "models/yolov8.pt",
                    "model_type": "yolov8",
                    "display_name": "Genel Tespit",
                    "description": "Genel nesne tespiti modeli",
                    "icon": "fas fa-eye",
                    "priority": 1,
                    "confidence_threshold": 0.5,
                    "classes": {
                        "0": {
                            "name": "person",
                            "display_name": "İnsan",
                            "color": [255, 0, 0],
                            "enabled": True,
                            "alert": True
                        }
                    }
                },
                "farm": {
                    "enabled": True,
                    "model_path": "models/farm_best.pt",
                    "model_type": "yolov8",
                    "display_name": "Tarım Tespiti",
                    "description": "Tarım hastalıkları tespiti modeli",
                    "icon": "fas fa-leaf",
                    "priority": 2,
                    "confidence_threshold": 0.5,
                    "classes": {
                        "0": {
                            "name": "healthy",
                            "display_name": "Sağlıklı",
                            "color": [0, 255, 0],
                            "enabled": True,
                            "alert": False
                        },
                        "1": {
                            "name": "diseased",
                            "display_name": "Hastalıklı",
                            "color": [255, 0, 0],
                            "enabled": True,
                            "alert": True
                        }
                    }
                }
            },
            "visualization": {
                "box_thickness": 2,
                "text_size": 1.0,
                "text_thickness": 2,
                "show_labels": True,
                "show_confidence": True,
                "show_fps": True
            },
            "performance": {
                "batch_size": 1,
                "device": "cuda",
                "half_precision": True,
                "num_threads": 4
            }
        }
    
    def __init__(self, config_path: str = "config/models_config.yaml"):
        """
        Config Manager'ı başlat
        
        Args:
            config_path: Config dosya yolu
        """
        self.config_path = Path(config_path)
        self.config_data = {}
        self.model_configs = {}
        self.system_config = {}
        
        # Threading
        self._lock = threading.Lock()
        
        # Config change callbacks
        self.change_callbacks = []
        
        # Default config
        self.default_config = self._get_default_config()
        
        # Load config
        self.load_config()
    
    def load_config(self) -> None:
        """Config dosyasını yükle"""
        try:
            # Config dosyası var mı kontrol et
            if not self.config_path.exists():
                logger.warning(f"Config dosyası bulunamadı: {self.config_path}")
                logger.info("Default config oluşturuluyor...")
                self._create_default_config()
            
            # Config dosyasını yükle
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            
            # Config'i validate et
            self._validate_config()
            
            # Model config'lerini parse et
            self._parse_model_configs()
            
            # System config'i parse et
            self._parse_system_config()
            
            logger.info(f"Config başarıyla yüklendi: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Config yükleme hatası: {e}")
            raise ConfigLoadError(f"Config yüklenemedi: {e}")
    
    def _create_default_config(self):
        """Default config dosyasını oluştur"""
        try:
            # Dizini oluştur
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Default config'i yaz
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.default_config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            
            logger.info(f"Default config oluşturuldu: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Default config oluşturma hatası: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Config'i doğrula"""
        try:
            # Gerekli anahtarları kontrol et
            required_keys = ['system', 'models', 'visualization', 'performance']
            
            for key in required_keys:
                if key not in self.config_data:
                    raise ConfigValidationError(f"Gerekli config anahtarı eksik: {key}")
            
            # Model config'lerini kontrol et
            if 'models' not in self.config_data:
                raise ConfigValidationError("Model konfigürasyonları bulunamadı")
            
            for model_name, model_config in self.config_data['models'].items():
                required_model_keys = ['enabled', 'model_path', 'model_type', 'classes']
                
                for key in required_model_keys:
                    if key not in model_config:
                        raise ConfigValidationError(f"Model {model_name} için gerekli anahtar eksik: {key}")
            
            logger.info("Config doğrulaması başarılı")
            
        except Exception as e:
            logger.error(f"Config doğrulama hatası: {e}")
            raise ConfigValidationError(str(e))
    
    def _parse_model_configs(self) -> None:
        """Model config'lerini parse et"""
        try:
            self.model_configs = {}
            
            for model_name, model_data in self.config_data['models'].items():
                # Sınıfları parse et
                classes = {}
                for class_id, class_data in model_data.get('classes', {}).items():
                    class_config = ClassConfig(
                        name=class_data['name'],
                        display_name=class_data['display_name'],
                        color=tuple(class_data['color']),
                        enabled=class_data.get('enabled', True),
                        alert=class_data.get('alert', False),
                        confidence_threshold=class_data.get('confidence_threshold')
                    )
                    classes[int(class_id)] = class_config
                
                # Model config oluştur
                model_config = ModelConfig(
                    enabled=model_data['enabled'],
                    model_path=model_data['model_path'],
                    model_type=model_data['model_type'],
                    display_name=model_data['display_name'],
                    description=model_data['description'],
                    icon=model_data['icon'],
                    priority=model_data['priority'],
                    confidence_threshold=model_data['confidence_threshold'],
                    classes=classes
                )
                
                self.model_configs[model_name] = model_config
            
            logger.info(f"Model config'leri parse edildi: {list(self.model_configs.keys())}")
            
        except Exception as e:
            logger.error(f"Model config parse hatası: {e}")
            raise
    
    def _parse_system_config(self) -> None:
        """System config'i parse et"""
        try:
            self.system_config = self.config_data.get('system', {})
            logger.info("System config parse edildi")
            
        except Exception as e:
            logger.error(f"System config parse hatası: {e}")
            raise
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Model config'ini döndür"""
        with self._lock:
            return self.model_configs.get(model_name)
    
    def get_all_model_configs(self) -> Dict[str, ModelConfig]:
        """Tüm model config'lerini döndür"""
        with self._lock:
            return self.model_configs.copy()
    
    def get_enabled_models(self) -> List[str]:
        """Aktif modellerin listesini döndür"""
        with self._lock:
            return [name for name, config in self.model_configs.items() if config.enabled]
    
    def get_class_config(self, model_name: str, class_id: int) -> Optional[ClassConfig]:
        """Belirli sınıf config'ini döndür"""
        with self._lock:
            model_config = self.model_configs.get(model_name)
            if model_config:
                return model_config.classes.get(class_id)
            return None
    
    def get_enabled_classes(self, model_name: str) -> Dict[int, ClassConfig]:
        """Model için aktif sınıfları döndür"""
        with self._lock:
            model_config = self.model_configs.get(model_name)
            if model_config:
                return {class_id: class_config for class_id, class_config in model_config.classes.items()
                       if class_config.enabled}
            return {}
    
    def get_visualization_setting(self, setting_name: str, default_value=None):
        """Visualization ayarını döndür"""
        with self._lock:
            visualization_config = self.config_data.get('visualization', {})
            return visualization_config.get(setting_name, default_value)
    
    def get_system_setting(self, setting_name: str, default_value=None):
        """Sistem ayarını döndür"""
        with self._lock:
            system_config = self.config_data.get('system', {})
            return system_config.get(setting_name, default_value)
            model_config = self.model_configs.get(model_name)
            if model_config:
                return {
                    class_id: class_config 
                    for class_id, class_config in model_config.classes.items()
                    if class_config.enabled
                }
            return {}
    
    def get_alert_classes(self, model_name: str) -> Dict[int, ClassConfig]:
        """Model için alert sınıflarını döndür"""
        with self._lock:
            model_config = self.model_configs.get(model_name)
            if model_config:
                return {
                    class_id: class_config 
                    for class_id, class_config in model_config.classes.items()
                    if class_config.alert and class_config.enabled
                }
            return {}
    
    def get_class_color(self, model_name: str, class_id: int) -> Tuple[int, int, int]:
        """Sınıf rengini döndür"""
        class_config = self.get_class_config(model_name, class_id)
        if class_config:
            return class_config.color
        
        # Default color
        return (128, 128, 128)
    
    def get_class_display_name(self, model_name: str, class_id: int) -> str:
        """Sınıf görüntüleme adını döndür"""
        class_config = self.get_class_config(model_name, class_id)
        if class_config:
            return class_config.display_name
        
        # Fallback to original name
        return f"Class_{class_id}"
    
    def is_class_enabled(self, model_name: str, class_id: int) -> bool:
        """Sınıfın aktif olup olmadığını kontrol et"""
        class_config = self.get_class_config(model_name, class_id)
        return class_config.enabled if class_config else False
    
    def is_alert_enabled(self, model_name: str, class_id: int) -> bool:
        """Sınıf için alert'in aktif olup olmadığını kontrol et"""
        class_config = self.get_class_config(model_name, class_id)
        return class_config.alert if class_config else False
    
    def get_system_setting(self, key: str, default=None) -> Any:
        """System ayarını döndür"""
        with self._lock:
            return self.system_config.get(key, default)
    
    def get_visualization_setting(self, key: str, default=None) -> Any:
        """Görselleştirme ayarını döndür"""
        with self._lock:
            visualization = self.config_data.get('visualization', {})
            return visualization.get(key, default)
    
    def get_performance_setting(self, key: str, default=None) -> Any:
        """Performance ayarını döndür"""
        with self._lock:
            performance = self.config_data.get('performance', {})
            return performance.get(key, default)
    
    def set_model_enabled(self, model_name: str, enabled: bool) -> None:
        """Model'i aktif/pasif et"""
        try:
            with self._lock:
                if model_name in self.model_configs:
                    self.model_configs[model_name].enabled = enabled
                    self.config_data['models'][model_name]['enabled'] = enabled
                    
                    # Config dosyasını güncelle
                    self._save_config()
                    
                    # Callback'leri çağır
                    self._notify_change('model_enabled', model_name, enabled)
                    
                    logger.info(f"Model {model_name} {'aktif' if enabled else 'pasif'} edildi")
                else:
                    logger.error(f"Model bulunamadı: {model_name}")
                    
        except Exception as e:
            logger.error(f"Model aktif/pasif etme hatası: {e}")
            raise
    
    def set_class_enabled(self, model_name: str, class_id: int, enabled: bool) -> None:
        """Sınıfı aktif/pasif et"""
        try:
            with self._lock:
                class_config = self.get_class_config(model_name, class_id)
                if class_config:
                    class_config.enabled = enabled
                    self.config_data['models'][model_name]['classes'][str(class_id)]['enabled'] = enabled
                    
                    # Config dosyasını güncelle
                    self._save_config()
                    
                    # Callback'leri çağır
                    self._notify_change('class_enabled', f"{model_name}.{class_id}", enabled)
                    
                    logger.info(f"Sınıf {model_name}.{class_id} {'aktif' if enabled else 'pasif'} edildi")
                else:
                    logger.error(f"Sınıf bulunamadı: {model_name}.{class_id}")
                    
        except Exception as e:
            logger.error(f"Sınıf aktif/pasif etme hatası: {e}")
            raise
    
    def set_confidence_threshold(self, model_name: str, threshold: float) -> None:
        """Model confidence threshold'unu ayarla"""
        
        with self._lock:
            if model_name in self.model_configs:
                self.model_configs[model_name].confidence_threshold = threshold
                self.config_data['models'][model_name]['confidence_threshold'] = threshold
                
                # Config dosyasını güncelle
                self._save_config()
                
                # Callback'leri çağır
                self._notify_change('confidence_threshold', model_name, threshold)
                
                logger.info(f"Model {model_name} confidence threshold: {threshold}")
            else:
                logger.error(f"Model bulunamadı: {model_name}")
    
    def _save_config(self) -> None:
        """Config dosyasını kaydet"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            logger.info(f"Config kaydedildi: {self.config_path}")
        except Exception as e:
            logger.error(f"Config kaydetme hatası: {e}")
            raise
    
    def _notify_change(self, change_type: str, key: str, value: Any) -> None:
        """Config değişikliğini bildir"""
        try:
            for callback in self.change_callbacks:
                try:
                    callback(change_type, key, value)
                except Exception as e:
                    logger.error(f"Callback hatası: {e}")
        except Exception as e:
            logger.error(f"Notify change hatası: {e}")
            
    def get_detection_categories(self) -> Dict[str, List[Dict]]:
        """Tespit kategorilerini ve alt kategorilerini döndür"""
        try:
            categories = {
                "diseases": [],  # Hastalıklar
                "pests": [],    # Zararlılar
                "deficiencies": [], # Besin eksiklikleri
                "ripeness": []  # Olgunluk durumu
            }
            
            with self._lock:
                farm_model = self.model_configs.get("farm")
                if farm_model and farm_model.classes:
                    for class_id, class_config in farm_model.classes.items():
                        if not class_config.enabled:
                            continue
                            
                        class_name = class_config.name.lower()
                        class_data = {
                            "id": class_id,
                            "name": class_config.name,
                            "display_name": class_config.display_name
                        }
                        
                        # Zararlılar kategorisi
                        if any(pest in class_name for pest in ["aphid", "whitefly", "mite", "thrips", "worm", "bug", "beetle", "fly"]):
                            categories["pests"].append(class_data)
                        
                        # Besin eksikliği kategorisi
                        elif "deficiency" in class_name:
                            categories["deficiencies"].append(class_data)
                        
                        # Hastalık kategorisi (virüs, küf, leke, vb.)
                        elif any(disease in class_name for disease in ["virus", "mold", "spot", "blight", "wilt", "rot"]):
                            categories["diseases"].append(class_data)
                        
                        # Olgunluk durumu kategorisi
                        elif any(ripeness in class_name for ripeness in ["ripe", "unripe", "semi_ripe"]):
                            categories["ripeness"].append(class_data)
            
            return categories
            
        except Exception as e:
            logger.error(f"Tespit kategorileri alma hatası: {e}")
            raise