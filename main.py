# -*- coding: utf-8 -*-
import sys
import joblib
import requests
import statistics
import json
import pandas as pd
from datetime import datetime, timedelta
import geopandas as gpd
import folium

from PyQt6.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDateTimeEdit, QSlider, QMessageBox,
    QProgressDialog
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Sabitler
USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
TURKEY_BBOX = {
    'minlatitude': 36.0,
    'maxlatitude': 42.5,
    'minlongitude': 26.0,
    'maxlongitude': 45.5
}
SHAPEFILE = "gadm41_TUR_1.json"  # İl seviyesinde shapefile kullanıyoruz

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_timeseries(self, events):
        self.axes.clear()
        times = [e['time'] for e in events]
        mags = [e['mag'] for e in events]
        self.axes.plot_date(times, mags, '-o', markersize=4)
        self.axes.set_title('Zamana Göre Deprem Büyüklüğü')
        self.axes.set_xlabel('Zaman')
        self.axes.set_ylabel('Büyüklük')
        self.axes.grid(True)
        self.draw()

class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(float, list, object)

    def __init__(self, model, provinces, start, end, threshold):
        super().__init__()
        self.model = model
        self.provinces = provinces
        self.start = start
        self.end = end
        self.threshold = threshold

    def run(self):
        try:
            events = self.fetch_events(self.start, self.end)
            if not events:
                self.error.emit("Belirtilen aralıkta deprem verisi bulunamadı.")
                self.finished.emit()
                return

            proba = self.predict_risk(events)

            # İl bazında risk hesapla
            provs = self.provinces.copy()
            risks = []
            for _, row in provs.iterrows():
                bounds = row.geometry.bounds
                bbox = {
                    'minlatitude': bounds[1], 
                    'maxlatitude': bounds[3],
                    'minlongitude': bounds[0], 
                    'maxlongitude': bounds[2]
                }
                evs = self.fetch_events(self.start, self.end, bbox)
                risks.append(self.predict_risk(evs) if evs else 0.0)
            provs['risk'] = risks

            self.result.emit(proba, events, provs)
        except Exception as e:
            self.error.emit(f"Hesaplama hatası: {str(e)}")
        finally:
            self.finished.emit()

    def fetch_events(self, start, end, bbox=None):
        params = {
            'format': 'geojson',
            'starttime': start.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime': end.strftime('%Y-%m-%dT%H:%M:%S'),
            'minmagnitude': 1.0
        }
        params.update(bbox or TURKEY_BBOX)
        response = requests.get(USGS_API_URL, params=params, timeout=15)
        response.raise_for_status()
        return self.parse_earthquake_data(response.json())

    def parse_earthquake_data(self, data):
        events = []
        for feature in data.get('features', []):
            props = feature.get('properties', {})
            coords = feature.get('geometry', {}).get('coordinates', [])
            if props.get('mag') is not None and len(coords) >= 3:
                events.append({
                    'time': datetime.utcfromtimestamp(props['time'] / 1000),
                    'mag': props['mag'],
                    'depth': coords[2],
                    'nst': props.get('nst', 0),
                    'gap': props.get('gap', 0.0),
                    'dmin': props.get('dmin', 0.0),
                    'rms': props.get('rms', 0.0)
                })
        return events

    def predict_risk(self, events):
        if not events:
            return 0.0
        training_features = ['mag', 'depth', 'nst', 'gap', 'dmin', 'rms']
        feature_values = {}
        for feat in training_features:
            if feat in events[0]:
                feature_values[feat] = statistics.mean([e[feat] for e in events])
            else:
                raise ValueError(f"'{feat}' özelliği olay verisinde eksik")
        features_df = pd.DataFrame([feature_values], columns=training_features)
        return self.model.predict_proba(features_df)[0][1] * 100
    
class MapWindow(QMainWindow):
    def __init__(self, provinces, shapefile_path):
        super().__init__()
        self.setWindowTitle('Deprem Risk Haritası')
        self.resize(1000, 800)
        
        # GeoJSON verisini yükle
        with open(shapefile_path, 'r', encoding='utf-8') as f:
            self.geojson_data = json.load(f)
        
        # Harita oluştur
        self.webview = QWebEngineView()
        self.setCentralWidget(self.webview)
        self.update_map(provinces)

    def update_map(self, provinces):
        m = folium.Map(location=[39.0, 35.0], zoom_start=5.5, tiles='CartoDB positron')
        
        folium.Choropleth(
            geo_data=self.geojson_data,
            data=provinces,
            columns=['NAME_1', 'risk'],  # İl bazında eşleşme
            key_on='feature.properties.NAME_1',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name='Deprem Riski (%)',
            highlight=True
        ).add_to(m)
        
        self.webview.setHtml(m._repr_html_())

class EarlyWarningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Türkiye Deprem Erken Uyarı Sistemi')
        self.resize(1200, 850)
        self.setStyleSheet("font-family: Arial; font-size: 12px;")

        # Verileri yükle
        try:
            self.model = joblib.load("models/random_forest_turkey_10yr.joblib")
            self.provinces = gpd.read_file(SHAPEFILE, encoding='utf-8')
        except Exception as e:
            QMessageBox.critical(self, "Kritik Hata", f"Başlatma hatası:\n{str(e)}")
            sys.exit(1)

        # Arayüz bileşenleri
        self.init_ui()
        
        # Otomatik yenileme
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.start_analysis)
        self.timer.start(600000)  # 10 dakikada bir

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Kontrol paneli
        control_panel = QVBoxLayout()
        layout.addLayout(control_panel, stretch=1)

        # Tarih seçimleri
        date_style = "QDateTimeEdit { padding: 5px; border-radius: 4px; border: 1px solid #ccc; }"
        self.start_dt = self.create_datetime_edit("Başlangıç Tarihi:", datetime.now() - timedelta(days=30), date_style)
        self.end_dt = self.create_datetime_edit("Bitiş Tarihi:", datetime.now(), date_style)
        
        # Eşik değeri
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.setTickInterval(10)
        self.slider.setStyleSheet("QSlider::handle:horizontal { background: #e74c3c; width: 16px; }")
        self.slider.valueChanged.connect(lambda v: self.thresh_label.setText(f"Eşik Değer: {v}%"))
        self.thresh_label = QLabel("Eşik Değer: 50%")
        
        # Analiz butonu
        self.analyze_btn = QPushButton("Analiz Başlat")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background: #2ecc71;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background: #27ae60; }
        """)
        self.analyze_btn.clicked.connect(self.start_analysis)

        # Bileşenleri panele ekle
        control_panel.addWidget(self.start_dt)
        control_panel.addWidget(self.end_dt)
        control_panel.addWidget(QLabel("Risk Eşik Değeri:"))
        control_panel.addWidget(self.slider)
        control_panel.addWidget(self.thresh_label)
        control_panel.addWidget(self.analyze_btn)
        control_panel.addStretch()

        # Sonuç paneli
        result_panel = QVBoxLayout()
        layout.addLayout(result_panel, stretch=2)

        # Genel risk göstergesi
        self.risk_label = QLabel("Genel Risk: Hesaplanıyor...")
        self.risk_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.risk_label.setStyleSheet("font-size: 18px; padding: 10px;")
        result_panel.addWidget(self.risk_label)

        # Grafik
        self.plot_canvas = PlotCanvas(self, width=7, height=4)
        result_panel.addWidget(self.plot_canvas)

    def create_datetime_edit(self, label, default, style):
        dt_edit = QDateTimeEdit(default)
        dt_edit.setCalendarPopup(True)
        dt_edit.setDisplayFormat("dd.MM.yyyy HH:mm")
        dt_edit.setStyleSheet(style)
        return dt_edit

    def start_analysis(self):
        self.analyze_btn.setEnabled(False)
        
        # İlerleme penceresi
        self.progress = QProgressDialog("Deprem verileri analiz ediliyor...", None, 0, 0, self)
        self.progress.setWindowTitle("Lütfen Bekleyin")
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.show()

        # İş parçacığı yapılandırması
        self.worker_thread = QThread()
        self.worker = Worker(
            self.model,
            self.provinces,
            self.start_dt.dateTime().toPyDateTime(),
            self.end_dt.dateTime().toPyDateTime(),
            self.slider.value()
        )
        self.worker.moveToThread(self.worker_thread)

        # Sinyal bağlantıları
        self.worker_thread.started.connect(self.worker.run)
        self.worker.result.connect(self.handle_results)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.cleanup)

        self.worker_thread.start()

    def handle_results(self, proba, events, provinces):
        self.progress.close()
        
        # Risk güncelleme
        threshold = self.slider.value()
        color = "#e74c3c" if proba >= threshold else "#2ecc71"
        self.risk_label.setText(f"Genel Risk: {proba:.1f}%")
        self.risk_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # Grafik çiz
        self.plot_canvas.plot_timeseries(events)
        
        # Harita penceresi
        self.map_window = MapWindow(provinces, SHAPEFILE)
        self.map_window.show()
        self.analyze_btn.setEnabled(True)

    def handle_error(self, message):
        self.progress.close()
        QMessageBox.warning(self, "Analiz Hatası", message)
        self.analyze_btn.setEnabled(True)

    def cleanup(self):
        self.worker.deleteLater()
        self.worker_thread.deleteLater()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EarlyWarningApp()
    window.show()
    sys.exit(app.exec())