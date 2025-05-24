import os
import sys
import numpy as np
from pathlib import Path

# FastAPI dizinini Python path'e ekle
sys.path.append(str(Path(__file__).parent))

from models.multi_model_detector import MultiModelYOLODetector

def test_basic():
    """Basit YOLO test"""
    print("Basit YOLO Testi")
    print("-" * 50)
    
    # Model dosyalarını kontrol et
    farm_model = "models/farm_best.pt"
    general_model = "models/yolov8n.pt"
    
    print(f"Farm model: {farm_model} - Mevcut: {os.path.exists(farm_model)}")
    print(f"General model: {general_model} - Mevcut: {os.path.exists(general_model)}")
    
    try:
        # Detector oluştur
        print("\nDetector oluşturuluyor...")
        detector = MultiModelYOLODetector(
            farm_model_path=farm_model,
            general_model_path=general_model
        )
        print("✓ Detector oluşturuldu")
        
        # Model bilgilerini göster
        print("\nModel Bilgileri:")
        model_info = detector.get_model_info()
        for model_name, info in model_info.items():
            print(f"\n{model_name}:")
            print(f"  - Yüklendi: {info.get('loaded', False)}")
            print(f"  - Sınıf sayısı: {info.get('class_count', 0)}")
        
        # Basit bir test frame'i oluştur
        print("\nTest frame'i oluşturuluyor...")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (50, 50, 50)  # Gri arka plan
        
        # Detection yap
        print("Detection yapılıyor...")
        result = detector.detect(test_frame)
        
        print(f"\n✓ Detection tamamlandı")
        print(f"Genel model: {len(result.general_detections)} tespit")
        print(f"Farm model: {len(result.farm_detections)} tespit")
        print(f"İşlem süresi: {result.processing_times.get('total', 0)*1000:.2f} ms")
        
        print("\n✅ TEST BAŞARILI!")
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic()