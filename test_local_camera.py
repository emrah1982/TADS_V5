import cv2
import asyncio
import numpy as np
from models.multi_model_detector import MultiModelYOLODetector
from video_sources.local_camera import LocalCameraSource
import time

async def test_local_camera():
    """Local kamera ile YOLO testi"""
    print("Local Kamera YOLO Testi")
    print("-" * 50)
    
    try:
        # Detector oluştur
        print("YOLO Detector oluşturuluyor...")
        detector = MultiModelYOLODetector(
            farm_model_path="models/farm_best.pt",
            general_model_path="models/yolov8n.pt"
        )
        print("✓ Detector oluşturuldu")
        
        # Local kamera source oluştur
        print("\nLocal kamera başlatılıyor...")
        camera = LocalCameraSource(detector, camera_id=0)
        
        # Kamerayı başlat
        success = await camera.initialize()
        if not success:
            print("❌ Kamera başlatılamadı!")
            return
        
        print("✓ Kamera başlatıldı")
        
        # Detection callback
        detection_count = 0
        def on_detection(result):
            nonlocal detection_count
            detection_count += 1
            print(f"\n[Frame {detection_count}] Tespit sonuçları:")
            print(f"  - Genel model: {len(result.general_detections)} tespit")
            print(f"  - Farm model: {len(result.farm_detections)} tespit")
            print(f"  - İşlem süresi: {result.processing_times.get('total', 0)*1000:.1f} ms")
            
            # Tespit detayları
            for det in result.all_detections:
                print(f"    • {det.model_type.value}: {det.class_name} ({det.confidence:.2%})")
        
        # Callback ayarla
        camera.set_detection_callback(on_detection)
        
        # Capture başlat
        print("\nCapture başlatılıyor...")
        camera.start_capture()
        print("✓ Capture başlatıldı")
        
        # 10 saniye bekle
        print("\n10 saniye boyunca tespit yapılacak...")
        print("Kameranın önünde hareket edin veya nesne gösterin")
        await asyncio.sleep(10)
        
        # Durdur
        print("\nKamera durduruluyor...")
        camera.stop_capture()
        await camera.cleanup()
        
        print(f"\n✅ TEST TAMAMLANDI!")
        print(f"Toplam işlenen frame: {detection_count}")
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()

def test_camera_availability():
    """Kamera erişilebilirliğini test et"""
    print("\nKamera Erişim Testi")
    print("-" * 50)
    
    # Kamera 0'ı test et
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Kamera 0 erişilebilir")
        
        # Kamera özelliklerini göster
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  - Çözünürlük: {width}x{height}")
        print(f"  - FPS: {fps}")
        
        # Test frame'i al
        ret, frame = cap.read()
        if ret:
            print("  - Test frame başarıyla alındı")
        else:
            print("  - Test frame alınamadı")
        
        cap.release()
    else:
        print("❌ Kamera 0 erişilemiyor")
        print("   Lütfen kameranızın bağlı olduğundan emin olun")

if __name__ == "__main__":
    # Önce kamera erişimini test et
    test_camera_availability()
    
    # Sonra YOLO testi yap
    response = input("\nYOLO testi yapmak ister misiniz? (e/h): ")
    if response.lower() == 'e':
        asyncio.run(test_local_camera())
    else:
        print("Test iptal edildi.")