import cv2
import numpy as np
from models.multi_model_detector import MultiModelYOLODetector
import time
import os

def test_yolo_detection():
    """YOLO modellerini test et"""
    
    print("YOLO Model Test Başlıyor...")
    print("-" * 50)
    
    # Model dosyalarını kontrol et
    farm_model_path = "models/farm_best.pt"
    general_model_path = "models/yolov8n.pt"
    
    print(f"Farm model dosyası: {farm_model_path}")
    print(f"Farm model mevcut mu: {os.path.exists(farm_model_path)}")
    print(f"General model dosyası: {general_model_path}")
    print(f"General model mevcut mu: {os.path.exists(general_model_path)}")
    print("-" * 50)
    
    try:
        # Detector'ı oluştur
        print("Detector oluşturuluyor...")
        detector = MultiModelYOLODetector(
            farm_model_path=farm_model_path,
            general_model_path=general_model_path
        )
        print("✓ Detector başarıyla oluşturuldu")
        
        # Model bilgilerini göster
        model_info = detector.get_model_info()
        print("\nModel Bilgileri:")
        for model_name, info in model_info.items():
            print(f"\n{model_name.upper()} Model:")
            print(f"  - Yüklendi: {info.get('loaded', False)}")
            print(f"  - Sınıf sayısı: {info.get('class_count', 0)}")
            print(f"  - Cihaz: {detector.device}")
        
        # Test görüntüsü oluştur (640x480 siyah görüntü)
        print("\nTest görüntüsü oluşturuluyor...")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Beyaz dikdörtgen ekle (nesne simülasyonu)
        cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.putText(test_frame, "TEST", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Detection yap
        print("\nDetection yapılıyor...")
        start_time = time.time()
        result = detector.detect(test_frame, frame_id=1)
        detection_time = time.time() - start_time
        
        print(f"\n✓ Detection tamamlandı ({detection_time*1000:.2f} ms)")
        print(f"Genel model tespitleri: {len(result.general_detections)}")
        print(f"Farm model tespitleri: {len(result.farm_detections)}")
        print(f"Toplam tespit: {len(result.all_detections)}")
        
        # Tespit detayları
        if result.all_detections:
            print("\nTespit Detayları:")
            for i, detection in enumerate(result.all_detections):
                print(f"\nTespit {i+1}:")
                print(f"  - Model: {detection.model_type.value}")
                print(f"  - Sınıf: {detection.class_name} (ID: {detection.class_id})")
                print(f"  - Güven: {detection.confidence:.2%}")
                print(f"  - Konum: {detection.bbox}")
                print(f"  - Alan: {detection.area} piksel²")
        
        # Görüntüyü çiz
        print("\nGörüntü işleniyor...")
        annotated_frame = detector.draw_detections(test_frame, result)
        
        # Görüntüyü kaydet
        output_path = "test_output.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"✓ Test görüntüsü kaydedildi: {output_path}")
        
        # İstatistikler
        stats = detector.get_statistics()
        print("\nPerformans İstatistikleri:")
        print(f"  - İşlenen frame: {stats['overall']['frames_processed']}")
        print(f"  - Ortalama FPS: {stats['overall']['fps']:.2f}")
        print(f"  - Cihaz: {stats['overall']['device']}")
        
        print("\n✅ TEST BAŞARILI!")
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()

def test_camera():
    """Kamera ile canlı test"""
    print("\nKamera Testi Başlıyor...")
    print("Çıkmak için 'q' tuşuna basın")
    print("-" * 50)
    
    try:
        # Detector'ı oluştur
        detector = MultiModelYOLODetector(
            farm_model_path="models/farm_best.pt",
            general_model_path="models/yolov8n.pt"
        )
        
        # Kamerayı aç
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Kamera açılamadı!")
            return
        
        print("✓ Kamera açıldı")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame okunamadı!")
                break
            
            # Detection yap
            result = detector.detect(frame, frame_id=frame_count)
            
            # Sonuçları çiz
            annotated_frame = detector.draw_detections(frame, result)
            
            # FPS hesapla
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # FPS'i görüntüye ekle
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Tespit sayılarını ekle
            cv2.putText(annotated_frame, f"Genel: {len(result.general_detections)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Farm: {len(result.farm_detections)}", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Görüntüyü göster
            cv2.imshow('YOLO Multi-Model Test', annotated_frame)
            
            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Temizlik
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Test tamamlandı")
        print(f"Toplam frame: {frame_count}")
        print(f"Ortalama FPS: {fps:.2f}")
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("YOLO Multi-Model Detection Test")
    print("=" * 50)
    
    # Önce basit test
    test_yolo_detection()
    
    # Kullanıcıya sor
    print("\n" + "=" * 50)
    response = input("\nKamera testi yapmak ister misiniz? (e/h): ")
    
    if response.lower() == 'e':
        test_camera()
    else:
        print("Test tamamlandı.")