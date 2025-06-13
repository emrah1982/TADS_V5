// Global değişkenler
let activeVideoSources = new Map();
let websocketConnections = new Map();

// API Base URL
const API_BASE = '';

// Video kaynağı yönetimi
class VideoSourceManager {
    constructor(sourceId, containerElement) {
        this.sourceId = sourceId;
        this.container = containerElement;
        this.isActive = false;
        this.videoElement = null;
        this.canvasElement = null;
        this.ctx = null;
        this.ws = null;
        
        this.initializeElements();
    }
    
    initializeElements() {
        // Video container oluştur
        const videoWrapper = document.createElement('div');
        videoWrapper.className = 'video-wrapper';
        videoWrapper.id = `video-${this.sourceId}`;
        
        // Başlık
        const header = document.createElement('div');
        header.className = 'video-header';
        header.innerHTML = `
            <h3>${this.getSourceTitle()}</h3>
            <button class="btn btn-sm btn-danger" onclick="stopVideoSource('${this.sourceId}')">
                <i class="fas fa-stop"></i> Durdur
            </button>
        `;
        
        // Video elementi
        this.videoElement = document.createElement('img');
        this.videoElement.className = 'video-stream';
        this.videoElement.style.width = '100%';
        this.videoElement.style.height = 'auto';
        
        // Canvas elementi (bounding box çizimi için)
        this.canvasElement = document.createElement('canvas');
        this.canvasElement.className = 'detection-overlay';
        this.canvasElement.style.position = 'absolute';
        this.canvasElement.style.top = '0';
        this.canvasElement.style.left = '0';
        this.canvasElement.style.pointerEvents = 'none';
        
        // Video container'a ekle
        const videoContainer = document.createElement('div');
        videoContainer.style.position = 'relative';
        videoContainer.appendChild(this.videoElement);
        videoContainer.appendChild(this.canvasElement);
        
        videoWrapper.appendChild(header);
        videoWrapper.appendChild(videoContainer);
        
        this.container.appendChild(videoWrapper);
        
        // Canvas context
        this.ctx = this.canvasElement.getContext('2d');
    }
    
    getSourceTitle() {
        if (this.sourceId.startsWith('local_')) {
            return '📷 Local Kamera';
        } else if (this.sourceId === 'dji_drone') {
            return '🚁 DJI Drone';
        } else if (this.sourceId === 'parrot_anafi') {
            return '🦜 Parrot Anafi';
        }
        return this.sourceId;
    }
    
    async start() {
        // Video stream URL'i ayarla
        this.videoElement.src = `/api/sources/${this.sourceId}/stream`;
        
        // WebSocket bağlantısı kur
        this.connectWebSocket();
        
        this.isActive = true;
    }
    
    connectWebSocket() {
        const wsUrl = `ws://localhost:8000/ws/${this.sourceId}`;
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log(`WebSocket bağlantısı kuruldu: ${this.sourceId}`);
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'detection') {
                    this.updateDetections(data.data);
                } else if (data.type === 'multi_model_detection') {
                    this.updateMultiModelDetections(data.data);
                }
            } catch (error) {
                console.error('WebSocket mesaj hatası:', error);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket hatası:', error);
        };
        
        this.ws.onclose = () => {
            console.log(`WebSocket bağlantısı kapandı: ${this.sourceId}`);
        };
    }
    
    updateDetections(detectionData) {
        // Video element yüklenmesini bekle
        if (this.videoElement.complete && this.videoElement.naturalWidth > 0) {
            this.updateCanvas(detectionData);
        } else {
            // Video yüklenene kadar bekle
            this.videoElement.onload = () => {
                this.updateCanvas(detectionData);
            };
        }
    }
    
    updateCanvas(detectionData) {
        // Canvas boyutunu video boyutuna eşitle
        this.canvasElement.width = this.videoElement.clientWidth;
        this.canvasElement.height = this.videoElement.clientHeight;
        
        // Canvas'ı temizle
        this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        
        // Frame size kontrolü
        if (!detectionData.frame_size || detectionData.frame_size.length < 2) {
            console.error('Frame size bilgisi eksik');
            return;
        }
        
        // Ölçek faktörünü hesapla
        const scaleX = this.canvasElement.width / detectionData.frame_size[0];
        const scaleY = this.canvasElement.height / detectionData.frame_size[1];
        
        // Tüm tespitleri çiz
        if (detectionData.all_detections && detectionData.all_detections.length > 0) {
            detectionData.all_detections.forEach(detection => {
                this.drawBoundingBox(detection, scaleX, scaleY);
            });
        }
    }
    
    updateMultiModelDetections(detectionData) {
        // Video element yüklenmesini bekle
        if (this.videoElement.complete && this.videoElement.naturalWidth > 0) {
            this.updateMultiModelCanvas(detectionData);
        } else {
            // Video yüklenene kadar bekle
            this.videoElement.onload = () => {
                this.updateMultiModelCanvas(detectionData);
            };
        }
        
        // İstatistikleri güncelle
        updateDetectionStats(detectionData);
    }
    
    updateMultiModelCanvas(detectionData) {
        // Canvas boyutunu video boyutuna eşitle
        this.canvasElement.width = this.videoElement.clientWidth;
        this.canvasElement.height = this.videoElement.clientHeight;
        
        // Canvas'ı temizle
        this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        
        // Frame size kontrolü
        if (!detectionData.frame_size || detectionData.frame_size.length < 2) {
            console.error('Frame size bilgisi eksik');
            return;
        }
        
        // Ölçek faktörünü hesapla
        const scaleX = this.canvasElement.width / detectionData.frame_size[0];
        const scaleY = this.canvasElement.height / detectionData.frame_size[1];
        
        // General model tespitleri (mavi)
        if (detectionData.general_detections && detectionData.general_detections.length > 0) {
            detectionData.general_detections.forEach(detection => {
                this.drawBoundingBox(detection, scaleX, scaleY, '#667eea');
            });
        }
        
        // Farm model tespitleri (turuncu)
        if (detectionData.farm_detections && detectionData.farm_detections.length > 0) {
            detectionData.farm_detections.forEach(detection => {
                this.drawBoundingBox(detection, scaleX, scaleY, '#ed8936');
            });
        }
        
        // Zararlı model tespitleri (kırmızı)
        if (detectionData.zararli_detections && detectionData.zararli_detections.length > 0) {
            detectionData.zararli_detections.forEach(detection => {
                this.drawBoundingBox(detection, scaleX, scaleY, '#e53e3e');
            });
        }
        
        // Domates Mineral model tespitleri (yeşil)
        if (detectionData.domatesMineral_detections && detectionData.domatesMineral_detections.length > 0) {
            detectionData.domatesMineral_detections.forEach(detection => {
                this.drawBoundingBox(detection, scaleX, scaleY, '#38a169');
            });
        }
        
        // Domates Hastalık model tespitleri (mor)
        if (detectionData.domatesHastalik_detections && detectionData.domatesHastalik_detections.length > 0) {
            detectionData.domatesHastalik_detections.forEach(detection => {
                this.drawBoundingBox(detection, scaleX, scaleY, '#805ad5');
            });
        }
        
        // Domates Olgunluk model tespitleri (pembe)
        if (detectionData.domatesOlgunluk_detections && detectionData.domatesOlgunluk_detections.length > 0) {
            detectionData.domatesOlgunluk_detections.forEach(detection => {
                this.drawBoundingBox(detection, scaleX, scaleY, '#d53f8c');
            });
        }
    }
    
    drawBoundingBox(detection, scaleX, scaleY, color = null) {
        const [x1, y1, x2, y2] = detection.bbox;
        
        // Koordinatları ölçekle ve canvas sınırları içinde tut
        const scaledX1 = Math.max(0, Math.min(this.canvasElement.width, x1 * scaleX));
        const scaledY1 = Math.max(0, Math.min(this.canvasElement.height, y1 * scaleY));
        const scaledX2 = Math.max(0, Math.min(this.canvasElement.width, x2 * scaleX));
        const scaledY2 = Math.max(0, Math.min(this.canvasElement.height, y2 * scaleY));
        
        // Geçersiz koordinatları kontrol et
        if (scaledX2 <= scaledX1 || scaledY2 <= scaledY1) {
            console.warn('Geçersiz bounding box koordinatları:', detection.bbox);
            return;
        }
        
        // Renk belirle
        if (!color) {
            color = detection.model_type === 'general' ? '#667eea' : '#ed8936';
        }
        
        // Bounding box çiz
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);
        
        // Label arka planı
        const label = `${detection.class_name} (${(detection.confidence * 100).toFixed(1)}%)`;
        this.ctx.font = '14px Arial';
        const textWidth = this.ctx.measureText(label).width;
        
        // Label pozisyonunu canvas sınırları içinde tut
        const labelX = Math.max(0, Math.min(this.canvasElement.width - textWidth - 10, scaledX1));
        const labelY = Math.max(20, scaledY1);
        
        this.ctx.fillStyle = color;
        this.ctx.fillRect(labelX, labelY - 20, textWidth + 10, 20);
        
        // Label metni
        this.ctx.fillStyle = 'white';
        this.ctx.fillText(label, labelX + 5, labelY - 5);
    }
    
    stop() {
        if (this.ws) {
            this.ws.close();
        }
        
        this.videoElement.src = '';
        this.container.innerHTML = '';
        this.isActive = false;
    }
}

// Local kamera başlatma
async function startLocalCamera() {
    const cameraId = document.getElementById('camera-id').value;
    const sourceId = `local_${cameraId}`;
    
    try {
        // API'ye istek gönder
        const response = await fetch(`${API_BASE}/api/sources/local-camera/start?camera_id=${cameraId}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Kamera başlatılamadı');
        }
        
        // Video kaynağını başlat
        const videoGrid = document.getElementById('video-grid');
        const videoSource = new VideoSourceManager(sourceId, videoGrid);
        await videoSource.start();
        
        activeVideoSources.set(sourceId, videoSource);
        
        // UI güncelle
        updateSourceStatus('local-camera', true);
        showAlert('Kamera başarıyla başlatıldı', 'success');
        
    } catch (error) {
        console.error('Kamera başlatma hatası:', error);
        showAlert('Kamera başlatılamadı: ' + error.message, 'danger');
    }
}

// Local kamera durdurma
async function stopLocalCamera() {
    const cameraId = document.getElementById('camera-id').value;
    const sourceId = `local_${cameraId}`;
    
    try {
        // API'ye istek gönder
        await fetch(`${API_BASE}/api/sources/${sourceId}/stop`, {
            method: 'POST'
        });
        
        // Video kaynağını durdur
        const videoSource = activeVideoSources.get(sourceId);
        if (videoSource) {
            videoSource.stop();
            activeVideoSources.delete(sourceId);
        }
        
        // UI güncelle
        updateSourceStatus('local-camera', false);
        showAlert('Kamera durduruldu', 'info');
        
    } catch (error) {
        console.error('Kamera durdurma hatası:', error);
        showAlert('Kamera durdurulamadı: ' + error.message, 'danger');
    }
}

// Video kaynağını durdur
function stopVideoSource(sourceId) {
    const videoSource = activeVideoSources.get(sourceId);
    if (videoSource) {
        videoSource.stop();
        activeVideoSources.delete(sourceId);
        
        // İlgili API çağrısını yap
        // API'ye durdurma isteği gönder
        fetch(`${API_BASE}/api/sources/${sourceId}/stop`, { method: 'POST' })
            .then(() => {
                if (sourceId.startsWith('local_')) {
                    updateSourceStatus('local-camera', false);
                }
            })
            .catch(error => console.error('Kaynak durdurma hatası:', error));
    }
}

// Kaynak durumunu güncelle
function updateSourceStatus(sourceType, isActive) {
    const statusElement = document.getElementById(`${sourceType}-status`);
    if (statusElement) {
        statusElement.className = `status-indicator ${isActive ? 'active' : ''}`;
    }
}

// Alert göster
function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // 5 saniye sonra otomatik kapat
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// Tespit istatistiklerini güncelle
function updateDetectionStats(detectionData) {
    // Genel model istatistikleri
    if (detectionData.general_detections) {
        const generalCount = detectionData.general_detections.length;
        document.getElementById('general-detections').textContent = generalCount;
        
        // Tespit listesini güncelle
        updateDetectionList('general-detections-list', detectionData.general_detections);
    }
    
    // Farm model istatistikleri
    if (detectionData.farm_detections) {
        const farmCount = detectionData.farm_detections.length;
        document.getElementById('farm-detections').textContent = farmCount;
        
        // Tespit listesini güncelle
        updateDetectionList('farm-detections-list', detectionData.farm_detections);
    }
    
    // Zararlı model istatistikleri
    if (detectionData.zararli_detections) {
        const zararliCount = detectionData.zararli_detections.length;
        document.getElementById('zararli-detections').textContent = zararliCount;
        
        // Tespit listesini güncelle
        updateDetectionList('zararli-detections-list', detectionData.zararli_detections);
    }
    
    // Domates Mineral model istatistikleri
    if (detectionData.domatesMineral_detections) {
        const domatesMineralCount = detectionData.domatesMineral_detections.length;
        document.getElementById('domates-mineral-detections').textContent = domatesMineralCount;
        
        // Tespit listesini güncelle
        updateDetectionList('domates-mineral-detections-list', detectionData.domatesMineral_detections);
    }
    
    // Domates Hastalık model istatistikleri
    if (detectionData.domatesHastalik_detections) {
        const domatesHastalikCount = detectionData.domatesHastalik_detections.length;
        document.getElementById('domates-hastalik-detections').textContent = domatesHastalikCount;
        
        // Tespit listesini güncelle
        updateDetectionList('domates-hastalik-detections-list', detectionData.domatesHastalik_detections);
    }
    
    // Domates Olgunluk model istatistikleri
    if (detectionData.domatesOlgunluk_detections) {
        const domatesOlgunlukCount = detectionData.domatesOlgunluk_detections.length;
        document.getElementById('domates-olgunluk-detections').textContent = domatesOlgunlukCount;
        
        // Tespit listesini güncelle
        updateDetectionList('domates-olgunluk-detections-list', detectionData.domatesOlgunluk_detections);
    }
    
    // Performance Overview kartlarını güncelle
    updatePerformanceCards(detectionData);
    
    // Toplam istatistikler
    const totalCount = (detectionData.general_detections?.length || 0) +
                      (detectionData.farm_detections?.length || 0) +
                      (detectionData.zararli_detections?.length || 0) +
                      (detectionData.domatesMineral_detections?.length || 0) +
                      (detectionData.domatesHastalik_detections?.length || 0) +
                      (detectionData.domatesOlgunluk_detections?.length || 0);
    document.getElementById('total-combined-detections').textContent = totalCount;
    
    // FPS güncelle
    if (detectionData.processing_times?.total) {
        const fps = (1000 / detectionData.processing_times.total).toFixed(1);
        document.getElementById('system-fps').textContent = fps;
    }
}

// Tespit listesini güncelle
function updateDetectionList(elementId, detections) {
    const listElement = document.getElementById(elementId);
    
    if (detections.length === 0) {
        listElement.innerHTML = '<p class="no-detections">Henüz tespit bulunamadı...</p>';
        return;
    }
    
    const detectionItems = detections.map(det => `
        <div class="detection-item">
            <span class="detection-class">${det.class_name}</span>
            <span class="detection-confidence">${(det.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
    
    listElement.innerHTML = detectionItems;
}

// Performance kartlarını güncelle
function updatePerformanceCards(detectionData) {
    // Genel model performans kartı
    if (detectionData.general_detections) {
        const element = document.getElementById('total-general-detections');
        if (element) element.textContent = detectionData.general_detections.length;
    }
    
    // Farm model performans kartı
    if (detectionData.farm_detections) {
        const element = document.getElementById('total-farm-detections');
        if (element) element.textContent = detectionData.farm_detections.length;
    }
    
    // Zararlı model performans kartı
    if (detectionData.zararli_detections) {
        const element = document.getElementById('total-zararli-detections');
        if (element) element.textContent = detectionData.zararli_detections.length;
    }
    
    // Domates Mineral model performans kartı
    if (detectionData.domatesMineral_detections) {
        const element = document.getElementById('total-domates-mineral-detections');
        if (element) element.textContent = detectionData.domatesMineral_detections.length;
    }
    
    // Domates Hastalık model performans kartı
    if (detectionData.domatesHastalik_detections) {
        const element = document.getElementById('total-domates-hastalik-detections');
        if (element) element.textContent = detectionData.domatesHastalik_detections.length;
    }
    
    // Domates Olgunluk model performans kartı
    if (detectionData.domatesOlgunluk_detections) {
        const element = document.getElementById('total-domates-olgunluk-detections');
        if (element) element.textContent = detectionData.domatesOlgunluk_detections.length;
    }
}

// Model toggle
async function toggleModel(modelType) {
    const toggle = document.getElementById(`${modelType}-model-toggle`);
    const enabled = toggle.checked;
    
    try {
        const response = await fetch(`${API_BASE}/api/models/${modelType}/toggle?enabled=${enabled}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Model durumu değiştirilemedi');
        }
        
        showAlert(`${modelType} model ${enabled ? 'aktif' : 'pasif'} edildi`, 'success');
        
    } catch (error) {
        console.error('Model toggle hatası:', error);
        toggle.checked = !enabled; // Geri al
        showAlert('Model durumu değiştirilemedi', 'danger');
    }
}

// Güven eşiği güncelleme
async function updateConfidenceThreshold(value) {
    document.getElementById('confidence-value').textContent = value;
    
    try {
        const response = await fetch(`${API_BASE}/api/models/confidence`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ threshold: parseFloat(value) })
        });
        
        if (!response.ok) {
            throw new Error('Güven eşiği güncellenemedi');
        }
        
    } catch (error) {
        console.error('Güven eşiği hatası:', error);
    }
}

// Sayfa yüklendiğinde
document.addEventListener('DOMContentLoaded', () => {
    console.log('YOLO Detection System başlatıldı');
    
    // Model bilgilerini yükle
    loadModelInfo();
});

// Model bilgilerini yükle
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/api/models/info`);
        const data = await response.json();
        
        // Model durumlarını güncelle
        if (data.models) {
            const models = data.models;
            
            // General model
            if (models.general) {
                document.getElementById('general-model-status').className =
                    `status-indicator ${models.general.loaded ? 'active' : ''}`;
                document.getElementById('general-classes').textContent =
                    models.general.class_count || '-';
            }
            
            // Farm model
            if (models.farm) {
                document.getElementById('farm-model-status').className =
                    `status-indicator ${models.farm.loaded ? 'active' : ''}`;
                document.getElementById('farm-classes').textContent =
                    models.farm.class_count || '-';
            }
            
            // Zararlı model
            if (models.zararli) {
                const statusElement = document.getElementById('zararli-model-status');
                const classesElement = document.getElementById('zararli-classes');
                if (statusElement) {
                    statusElement.className = `status-indicator ${models.zararli.loaded ? 'active' : ''}`;
                }
                if (classesElement) {
                    classesElement.textContent = models.zararli.class_count || '-';
                }
            }
            
            // Domates Mineral model
            if (models.domatesMineral) {
                const statusElement = document.getElementById('domates-mineral-model-status');
                const classesElement = document.getElementById('domates-mineral-classes');
                if (statusElement) {
                    statusElement.className = `status-indicator ${models.domatesMineral.loaded ? 'active' : ''}`;
                }
                if (classesElement) {
                    classesElement.textContent = models.domatesMineral.class_count || '-';
                }
            }
            
            // Domates Hastalık model
            if (models.domatesHastalik) {
                const statusElement = document.getElementById('domates-hastalik-model-status');
                const classesElement = document.getElementById('domates-hastalik-classes');
                if (statusElement) {
                    statusElement.className = `status-indicator ${models.domatesHastalik.loaded ? 'active' : ''}`;
                }
                if (classesElement) {
                    classesElement.textContent = models.domatesHastalik.class_count || '-';
                }
            }
            
            // Domates Olgunluk model
            if (models.domatesOlgunluk) {
                const statusElement = document.getElementById('domates-olgunluk-model-status');
                const classesElement = document.getElementById('domates-olgunluk-classes');
                if (statusElement) {
                    statusElement.className = `status-indicator ${models.domatesOlgunluk.loaded ? 'active' : ''}`;
                }
                if (classesElement) {
                    classesElement.textContent = models.domatesOlgunluk.class_count || '-';
                }
            }
        } else {
            // Eski format desteği
            if (data.general) {
                document.getElementById('general-model-status').className =
                    `status-indicator ${data.general.loaded ? 'active' : ''}`;
                document.getElementById('general-classes').textContent =
                    data.general.class_count || '-';
            }
            
            if (data.farm) {
                document.getElementById('farm-model-status').className =
                    `status-indicator ${data.farm.loaded ? 'active' : ''}`;
                document.getElementById('farm-classes').textContent =
                    data.farm.class_count || '-';
            }
        }
        
    } catch (error) {
        console.error('Model bilgileri yüklenemedi:', error);
    }
}

// Placeholder fonksiyonlar (diğer özellikler için)
function startDJIDrone() {
    showAlert('DJI Drone desteği yakında eklenecek', 'info');
}

function stopDJIDrone() {
    showAlert('DJI Drone desteği yakında eklenecek', 'info');
}

function startParrotAnafi() {
    showAlert('Parrot Anafi desteği yakında eklenecek', 'info');
}

function stopParrotAnafi() {
    showAlert('Parrot Anafi desteği yakında eklenecek', 'info');
}

function toggleConfigPanel() {
    const overlay = document.getElementById('config-overlay');
    overlay.style.display = overlay.style.display === 'flex' ? 'none' : 'flex';
}

function exportDetectionData() {
    showAlert('Export özelliği yakında eklenecek', 'info');
}

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

function showConfigTab(tabName) {
    // Tüm tab içeriklerini gizle
    document.querySelectorAll('.config-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Tüm tab butonlarını pasif yap
    document.querySelectorAll('.config-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Seçili tab'ı aktif yap
    document.getElementById(`config-${tabName}`).classList.add('active');
    event.target.classList.add('active');
}

// Fonksiyonları global scope'a ekle (HTML onclick için)
Object.assign(window, {
    startLocalCamera,
    stopLocalCamera,
    startDJIDrone,
    stopDJIDrone,
    startParrotAnafi,
    stopParrotAnafi,
    toggleModel,
    updateConfidenceThreshold,
    toggleConfigPanel,
    exportDetectionData,
    toggleFullscreen,
    showConfigTab,
    stopVideoSource
});