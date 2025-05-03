# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import requests

class EarthquakePredictor:
    def __init__(self, config):
        self.config = config
        self.model = config.model
        self.scaler = self._load_scaler()
        self.resolution = config.get("graphical_geometry", 1775)
        self.feature_columns = [
            'magnitude', 
            'depth_km', 
            'distance_to_coast',
            'historic_activity'
        ]

    def _load_scaler(self):
        """Eğitilmiş scaler'ı yükler"""
        try:
            return joblib.load("models/scaler.joblib")
        except FileNotFoundError:
            print("⚠️ Scaler bulunamadı, yeni bir scaler oluşturuluyor...")
            return StandardScaler()

    def fetch_realtime_data(self):
        """USGS'den gerçek zamanlı veri çeker"""
        params = {
            "format": "geojson",
            "starttime": (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "endtime": datetime.utcnow().strftime("%Y-%m-%d"),
            "minmagnitude": self.config.get("min_magnitude", 2.0),
            "bbox": self.config.get("turkey_bbox", "35.5,25.5,42.5,44.5")
        }
        
        try:
            response = requests.get(
                self.config.get("usgs_api"),
                params=params,
                timeout=10
            )
            return self._parse_data(response.json()['features'])
        except Exception as e:
            print(f"📡 Veri çekme hatası: {str(e)}")
            return pd.DataFrame()

    def _parse_data(self, features):
        """Ham USGS verisini işler"""
        processed = []
        for feature in features:
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            processed.append({
                'timestamp': datetime.utcfromtimestamp(props['time']/1000),
                'latitude': coords[1],
                'longitude': coords[0],
                'depth_km': coords[2],
                'magnitude': props['mag'],
                'location': props['place']
            })
        return pd.DataFrame(processed)

    def predict_risk(self, quake_data):
        """Tek deprem için risk analizi yapar"""
        try:
            processed = self._process_features(quake_data)
            proba = self.model.predict_proba([processed])[0][1]
            return {
                'risk': round(proba, 3),
                'alert': proba > self.config.get("alert_threshold", 0.65),
                'resolution': self.resolution
            }
        except Exception as e:
            print(f"🔮 Tahmin hatası: {str(e)}")
            return None

    def _process_features(self, data):
        """Veriyi model formatına dönüştürür"""
        features = [
            data['magnitude'],
            data['depth_km'],
            self._coast_distance(data['latitude'], data['longitude']),
            self._historic_activity(data['latitude'], data['longitude'])
        ]
        return self.scaler.transform([features])[0]

    def _coast_distance(self, lat, lon):
        """Kıyıya mesafe (basitleştirilmiş)"""
        return abs(lat - 36.0) + abs(lon - 30.0)  # Antalya referans noktası

    def _historic_activity(self, lat, lon):
        """Tarihsel deprem aktivitesi"""
        return 0.75 if lon > 33.0 else 0.25  # Doğu-batı ayrımı

    def optimize_visualization(self, raw_data):
        """Grafik çözünürlüğüne göre veriyi optimize eder"""
        if raw_data.empty:
            return pd.DataFrame()
            
        step_size = max(1, int(len(raw_data) / (self.resolution / 1000)))
        return raw_data.iloc[::step_size]

def train_model():
    """Model eğitim ve kaydetme fonksiyonu"""
    from sklearn.model_selection import train_test_split
    
    # Örnek veri üretimi
    X = np.random.rand(10000, 4) * 10  #magnitude, depth, distance, activity
    y = np.where(X[:,0] > 5.0, 1, 0)  # 5.0+ büyüklük riskli kabul edilsin
    
    # Veri ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model eğitimi
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced'
    )
    model.fit(X_scaled, y)
    
    # Kaydetme
    joblib.dump(model, "models/tr_earthquake_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("✅ Model başarıyla eğitildi ve kaydedildi")

if __name__ == "__main__":
    train_model()