# -*- coding: utf-8 -*-
import os
import json
import joblib
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

class ConfigManager:
    def __init__(self):
        self.config_path = os.path.abspath("config/config.json")
        self.default_config = {
            "settings": {
                "model_path": "models/random_forest_turkey_10yr.joblib",
                "usgs_api": "https://earthquake.usgs.gov/fdsnws/event/1/query.geojson",
                "turkey_bbox": "35.5,25.5,42.5,44.5",  # Türkiye sınırları
                "min_magnitude": 2.0,
                "days_history": 30,
                "alert_threshold": 0.65,
                "graphical_geometry": 1775
            }
        }
        self.config = self.load()
        self.model = self._load_model()

    def _load_model(self):
        """Kayıtlı modeli yükler"""
        model_path = os.path.abspath(self.get("model_path"))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❗ Model dosyası eksik: {model_path}")
        
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"❌ Model yükleme hatası: {str(e)}")
            raise

    def get_historical_quakes(self) -> pd.DataFrame:
        """USGS'den Türkiye deprem tarihçesini çeker"""
        params = {
            "starttime": (datetime.now() - timedelta(days=self.get("days_history"))).strftime("%Y-%m-%d"),
            "endtime": datetime.now().strftime("%Y-%m-%d"),
            "minmagnitude": self.get("min_magnitude"),
            "bbox": self.get("turkey_bbox"),
            "orderby": "time-asc"
        }

        try:
            response = requests.get(
                self.get("usgs_api"),
                params=params,
                timeout=15
            )
            response.raise_for_status()
            return self._parse_quake_data(response.json()['features'])
        except Exception as e:
            print(f"🌍 USGS API hatası: {str(e)}")
            return pd.DataFrame()

    def _parse_quake_data(self, features: List[dict]) -> pd.DataFrame:
        """Ham deprem verisini işler"""
        quakes = []
        for feature in features:
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            quakes.append({
                'timestamp': datetime.utcfromtimestamp(props['time']/1000),
                'latitude': coords[1],
                'longitude': coords[0],
                'depth_km': coords[2],
                'magnitude': props['mag'],
                'location': props['place'],
                'id': feature['id']
            })
        return pd.DataFrame(quakes)

    # Diğer yardımcı metodlar
    def load(self) -> Dict[str, Any]:
        """Config dosyasını yükler"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return self.default_config.copy()
        except Exception as e:
            print(f"⚠️ Config hatası: {str(e)}")
            return self.default_config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        return self.config['settings'].get(key, default)

    def save(self, settings: Dict[str, Any]):
        """Ayarları kaydeder"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump({'settings': settings}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Config kaydetme hatası: {str(e)}")