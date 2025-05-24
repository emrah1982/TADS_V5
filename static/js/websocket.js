// WebSocket bağlantılarını yönetme
let wsConnections = {};

// Global değişkenler
let generalDetectionHistory = [];
let farmDetectionHistory = [];
let modelStats = {
    general: {
        detections: 0,
        avgTime: 0,
        fps: 0
    },
    farm: {
        detections: 0,
        avgTime: 0,
        fps: 0
    }
};
let systemStats = {};

// WebSocket bağlantısı kurma
async function connectWebSocket(source) {
    try {
        const ws = new WebSocket(`ws://localhost:8000/ws/${source}`);
        
        ws.onopen = () => {
            console.log(`${source} WebSocket bağlantısı kuruldu`);
            if (window.showAlert) {
                window.showAlert(`${source} bağlantısı kuruldu`, 'success');
            }
        };
        
        ws.onclose = (event) => {
            console.log(`${source} WebSocket bağlantısı kapandı:`, event.code, event.reason);
            if (window.showAlert) {
                window.showAlert(`${source} bağlantısı kapandı`, 'warning');
            }
            
            // Bağlantıyı kaldır
            if (wsConnections[source]) {
                delete wsConnections[source];
            }
            
            // Kaynak durumunu güncelle
            if (window.updateSourceStatus) {
                window.updateSourceStatus(source, false);
            }
        };
        
        ws.onerror = (error) => {
            console.error(`${source} WebSocket hatası:`, error);
            if (window.showAlert) {
                window.showAlert(`${source} bağlantı hatası`, 'error');
            }
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(source, data);
            } catch (error) {
                console.error('WebSocket mesaj işleme hatası:', error);
            }
        };
        
        wsConnections[source] = ws;
        return ws;
        
    } catch (error) {
        console.error('WebSocket bağlantı hatası:', error);
        if (window.showAlert) {
            window.showAlert('WebSocket bağlantı hatası: ' + error.message, 'error');
        }
        throw error;
    }
}

// WebSocket mesajlarını işleme
function handleWebSocketMessage(source, data) {
    switch (data.type) {
        case 'detection':
            handleMultiModelDetectionMessage(source, data);
            break;
        case 'statistics':
            handleStatisticsMessage(data);
            break;
        case 'source_started':
            console.log(`${data.source} kaynağı başlatıldı`);
            break;
        case 'ping':
            // Ping mesajlarını yoksay
            break;
        default:
            console.log('Bilinmeyen mesaj tipi:', data.type);
    }
}

// Multi-model tespit mesajlarını işleme
function handleMultiModelDetectionMessage(source, detectionData) {
    // Tespit verilerini işle
    if (detectionData.general_detections) {
        generalDetectionHistory.push({
            timestamp: new Date(),
            source: source,
            detections: detectionData.general_detections
        });
        
        // İstatistikleri güncelle
        modelStats.general.detections += detectionData.general_detections.length;
        if (detectionData.general_time) {
            modelStats.general.avgTime = detectionData.general_time;
        }
    }
    
    if (detectionData.farm_detections) {
        farmDetectionHistory.push({
            timestamp: new Date(),
            source: source,
            detections: detectionData.farm_detections
        });
        
        // İstatistikleri güncelle
        modelStats.farm.detections += detectionData.farm_detections.length;
        if (detectionData.farm_time) {
            modelStats.farm.avgTime = detectionData.farm_time;
        }
    }
    
    // FPS hesapla
    if (detectionData.fps) {
        if (detectionData.general_detections) {
            modelStats.general.fps = detectionData.fps.general || 0;
        }
        if (detectionData.farm_detections) {
            modelStats.farm.fps = detectionData.fps.farm || 0;
        }
    }
    
    // UI güncelle
    if (window.updateDetectionsDisplay) {
        window.updateDetectionsDisplay();
    }
    if (window.updateVideoInfo) {
        window.updateVideoInfo(source, detectionData);
    }
}

// Tespit gösterimini güncelle
function updateDetectionsDisplay() {
    // Bu fonksiyon main.js'den çağrılacak
    console.log('Detections display update requested');
}

// Video bilgilerini güncelle
function updateVideoInfo(source, detectionData) {
    // Bu fonksiyon main.js'den çağrılacak
    console.log('Video info update requested for:', source);
}

// İstatistik mesajlarını işleme
function handleStatisticsMessage(stats) {
    systemStats = stats;
    
    // Model istatistiklerini güncelle
    if (stats.model_stats) {
        if (stats.model_stats.general) {
            modelStats.general = { ...modelStats.general, ...stats.model_stats.general };
        }
        if (stats.model_stats.farm) {
            modelStats.farm = { ...modelStats.farm, ...stats.model_stats.farm };
        }
    }
    
    // Performans göstergelerini güncelle
    if (window.updatePerformanceDisplay) {
        window.updatePerformanceDisplay();
    }
}

// Performans gösterimini güncelle
function updatePerformanceDisplay() {
    // Bu fonksiyon main.js'den çağrılacak
    console.log('Performance display update requested');
}

// WebSocket bağlantılarını temizleme
function cleanupWebSocketConnections() {
    Object.entries(wsConnections).forEach(([source, ws]) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    });
    wsConnections = {};
}

// Sayfa kapatıldığında bağlantıları temizle
window.addEventListener('beforeunload', cleanupWebSocketConnections);

// WebSocket fonksiyonlarını dışa aktar
export {
    wsConnections,
    connectWebSocket,
    handleWebSocketMessage,
    handleMultiModelDetectionMessage,
    handleStatisticsMessage,
    cleanupWebSocketConnections
};
