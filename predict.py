import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
import joblib

logger = logging.getLogger(__name__)

class Predictor:
    """Prediction module extracted from EnhancedTradingModel"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.shap_explainers = {}
    
    def load_model(self, prediction_days: int, model_path: str, scaler_path: str, 
                   feature_columns_path: str, shap_explainer_path: str):
        """Load trained model and associated components"""
        try:
            self.models[prediction_days] = joblib.load(model_path)
            self.scalers[prediction_days] = joblib.load(scaler_path)
            self.feature_columns[prediction_days] = joblib.load(feature_columns_path)
            self.shap_explainers[prediction_days] = joblib.load(shap_explainer_path)
            logger.info(f"Loaded model for {prediction_days} days prediction")
        except Exception as e:
            logger.error(f"Error loading model for {prediction_days} days: {e}")
            raise
    
    def predict_proba(self, features, prediction_days):
        """Make prediction with proper feature alignment"""
        if prediction_days not in self.models:
            return None

        try:
            model = self.models[prediction_days]
            scaler = self.scalers[prediction_days]
            feature_cols = self.feature_columns[prediction_days]

            if not isinstance(features, pd.DataFrame):
                return None

            features_aligned = pd.DataFrame(index=features.index, columns=feature_cols)

            for col in feature_cols:
                if col in features.columns:
                    features_aligned[col] = features[col]
                else:
                    features_aligned[col] = 0

            features_aligned = features_aligned.fillna(0)

            features_array = features_aligned.values

            features_scaled = scaler.transform(features_array)

            proba = model.predict_proba(features_scaled)[0]
            return proba

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None

    def get_signal_shap_explanation(self, features, prediction_days):
        """Get enhanced SHAP explanation showing all feature types"""
        if prediction_days not in self.models or prediction_days not in self.shap_explainers:
            return None

        try:
            feature_cols = self.feature_columns[prediction_days]
            scaler = self.scalers[prediction_days]

            if not isinstance(features, pd.DataFrame):
                return None

            features_aligned = pd.DataFrame(index=features.index, columns=feature_cols)

            for col in feature_cols:
                if col in features.columns:
                    features_aligned[col] = features[col]
                else:
                    features_aligned[col] = 0

            features_aligned = features_aligned.fillna(0)
            features_array = features_aligned.values

            if features_array.shape[0] != 1:
                features_array = features_array[0:1]

            features_scaled = scaler.transform(features_array)

            proba = self.models[prediction_days].predict_proba(features_scaled)[0]

            try:
                shap_values = self.shap_explainers[prediction_days].shap_values(features_scaled)
            except:
                explanation = self.shap_explainers[prediction_days](features_scaled)
                shap_values = explanation.values

            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]

            if shap_values.shape[0] == 1:
                shap_values = shap_values[0]

            top_features = self._get_top_features_by_type(feature_cols, shap_values)
            
            return {
                'prediction_probability': proba,
                'shap_values': shap_values,
                'top_features': top_features,
                'feature_values': features_aligned.iloc[0].to_dict()
            }

        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            return None
    
    def _get_top_features_by_type(self, feature_names, shap_values, n_per_type=5):
        """Get top N features from each category"""
        shap_importance = np.abs(shap_values)
        
        categories = self._categorize_features(feature_names)
        
        top_features_by_type = {}
        
        for category, features in categories.items():
            if not features:
                continue
                
            feature_indices = [i for i, name in enumerate(feature_names) if name in features]
            
            if feature_indices:
                category_importance = [(i, shap_importance[i]) for i in feature_indices]
                category_importance.sort(key=lambda x: x[1], reverse=True)
                
                top_features_by_type[category] = []
                for i, importance in category_importance[:n_per_type]:
                    top_features_by_type[category].append({
                        'feature': feature_names[i],
                        'shap_value': float(shap_values[i]),
                        'importance': float(importance)
                    })
        
        return top_features_by_type
    
    def _categorize_features(self, feature_names):
        """Categorize features by type"""
        categories = {
            'macro': [],
            'proprietary': [],
            'technical': [],
            'interaction': [],
            'regime': [],
            'transformed': []
        }
        
        proprietary_features = ['VIX', 'FNG', 'RSI', 'AnnVolatility', 'Momentum125', 
                               'PriceStrength', 'VolumeBreadth', 'CallPut', 'NewsScore', 
                               'MACD', 'BollingerBandWidth']
        
        for feature in feature_names:
            if any(prop in feature for prop in proprietary_features):
                categories['proprietary'].append(feature)
            elif 'interaction' in feature.lower() or '_x_' in feature:
                categories['interaction'].append(feature)
            elif 'regime' in feature.lower():
                categories['regime'].append(feature)
            elif 'transformed' in feature.lower() or '_sq' in feature or '_log' in feature:
                categories['transformed'].append(feature)
            elif any(macro in feature for macro in ['GDP', 'UNRATE', 'CPI', 'PAYEMS', 'FEDFUNDS']):
                categories['macro'].append(feature)
            else:
                categories['technical'].append(feature)
        
        return categories
