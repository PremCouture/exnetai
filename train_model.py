import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap
import joblib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training module extracted from EnhancedTradingModel"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.shap_explainers = {}
        self.per_stock_metrics = {}
        self.feature_importance = {}
    
    def prepare_training_data(self, stock_data: Dict, prediction_days: int, 
                            min_samples_per_stock: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for the specified prediction horizon"""
        logger.info(f"Preparing training data for {prediction_days}-day prediction...")
        
        all_features = []
        all_targets = []
        
        for ticker, df in stock_data.items():
            if len(df) < min_samples_per_stock:
                logger.warning(f"Skipping {ticker}: insufficient data ({len(df)} < {min_samples_per_stock})")
                continue
            
            try:
                target_col = f'Target_{prediction_days}d'
                if target_col not in df.columns:
                    logger.warning(f"Target column {target_col} not found for {ticker}")
                    continue
                
                valid_mask = df[target_col].notna()
                if valid_mask.sum() < min_samples_per_stock // 2:
                    logger.warning(f"Insufficient valid targets for {ticker}")
                    continue
                
                feature_cols = [col for col in df.columns if not col.startswith('Target_')]
                stock_features = df[feature_cols][valid_mask].copy()
                stock_targets = df[target_col][valid_mask].copy()
                
                stock_features = stock_features.replace([np.inf, -np.inf], np.nan)
                stock_features = stock_features.fillna(0)
                
                all_features.append(stock_features)
                all_targets.append(stock_targets)
                
                logger.info(f"Added {len(stock_features)} samples from {ticker}")
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data found")
        
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        logger.info(f"Combined training data: {len(combined_features)} samples, {len(combined_features.columns)} features")
        
        return combined_features, combined_targets
    
    def train_model(self, stock_data: Dict, prediction_days: int, 
                   test_size: float = 0.2, random_state: int = 42) -> RandomForestClassifier:
        """Train Random Forest model for the specified prediction horizon"""
        logger.info(f"\n{'='*50}")
        logger.info(f"TRAINING MODEL FOR {prediction_days}-DAY PREDICTION")
        logger.info(f"{'='*50}")
        
        X, y = self.prepare_training_data(stock_data, prediction_days)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        logger.info("Training Random Forest model...")
        rf_model.fit(X_train_scaled, y_train)
        
        train_accuracy = rf_model.score(X_train_scaled, y_train)
        test_accuracy = rf_model.score(X_test_scaled, y_test)
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        y_pred = rf_model.predict(X_test_scaled)
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        self.models[prediction_days] = rf_model
        self.scalers[prediction_days] = scaler
        self.feature_columns[prediction_days] = list(X.columns)
        
        logger.info("Creating SHAP explainer...")
        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_test_scaled[:100])
            self.shap_explainers[prediction_days] = explainer
            logger.info("SHAP explainer created successfully")
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
        
        self._analyze_feature_importance(X.columns, rf_model, prediction_days)
        
        self.per_stock_metrics[prediction_days] = self.calculate_per_stock_metrics(stock_data, prediction_days)
        logger.info(f"Calculated metrics for {len(self.per_stock_metrics[prediction_days])} stocks")
        
        return rf_model
    
    def _analyze_feature_importance(self, feature_names: List[str], model: RandomForestClassifier, 
                                  prediction_days: int):
        """Analyze and log feature importance by category"""
        feature_importance = model.feature_importances_
        
        categories = self._categorize_features(feature_names)
        
        category_importance = {}
        for category, features in categories.items():
            if features:
                indices = [i for i, name in enumerate(feature_names) if name in features]
                category_importance[category] = sum(feature_importance[i] for i in indices)
        
        self.feature_importance[prediction_days] = {
            'individual': dict(zip(feature_names, feature_importance)),
            'by_category': category_importance
        }
        
        logger.info(f"\nFeature importance by category for {prediction_days}-day prediction:")
        for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {importance:.4f}")
    
    def calculate_per_stock_metrics(self, stock_data: Dict, prediction_days: int) -> Dict:
        """Calculate performance metrics for each stock"""
        per_stock_metrics = {}
        
        if prediction_days not in self.models:
            return per_stock_metrics
        
        model = self.models[prediction_days]
        scaler = self.scalers[prediction_days]
        feature_cols = self.feature_columns[prediction_days]
        
        for ticker, df in stock_data.items():
            try:
                target_col = f'Target_{prediction_days}d'
                if target_col not in df.columns:
                    continue
                
                valid_mask = df[target_col].notna()
                if valid_mask.sum() < 10:
                    continue
                
                stock_features = df[feature_cols][valid_mask].fillna(0)
                stock_targets = df[target_col][valid_mask]
                
                stock_features_scaled = scaler.transform(stock_features)
                predictions = model.predict(stock_features_scaled)
                
                accuracy = accuracy_score(stock_targets, predictions)
                
                per_stock_metrics[ticker] = {
                    'accuracy': accuracy,
                    'samples': len(stock_targets),
                    'class_distribution': stock_targets.value_counts().to_dict()
                }
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {ticker}: {e}")
                continue
        
        return per_stock_metrics
    
    def save_model(self, prediction_days: int, model_dir: str):
        """Save trained model and associated components"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        if prediction_days in self.models:
            joblib.dump(self.models[prediction_days], f"{model_dir}/model_{prediction_days}d.pkl")
            joblib.dump(self.scalers[prediction_days], f"{model_dir}/scaler_{prediction_days}d.pkl")
            joblib.dump(self.feature_columns[prediction_days], f"{model_dir}/features_{prediction_days}d.pkl")
            
            if prediction_days in self.shap_explainers:
                joblib.dump(self.shap_explainers[prediction_days], f"{model_dir}/shap_{prediction_days}d.pkl")
            
            logger.info(f"Saved model for {prediction_days}-day prediction to {model_dir}")
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
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
            if 'interaction' in feature.lower() or '_x_' in feature.lower():
                categories['interaction'].append(feature)
            elif 'regime' in feature.lower():
                categories['regime'].append(feature)
            elif 'transformed' in feature.lower() or '_sq' in feature or '_log' in feature:
                categories['transformed'].append(feature)
            elif any(prop in feature for prop in proprietary_features):
                categories['proprietary'].append(feature)
            elif any(macro in feature for macro in ['GDP', 'UNRATE', 'CPI', 'PAYEMS', 'FEDFUNDS']):
                categories['macro'].append(feature)
            else:
                categories['technical'].append(feature)
        
        return categories
