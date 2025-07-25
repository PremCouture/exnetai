import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from train_model import ModelTrainer

class TestModelTrainer:
    
    @pytest.fixture
    def trainer(self):
        return ModelTrainer()
    
    @pytest.fixture
    def sample_stock_data(self):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        stock_data = {}
        for ticker in ['AAPL', 'MSFT']:
            df = pd.DataFrame({
                'Close': np.random.randn(200).cumsum() + 100,
                'VIX': np.random.uniform(10, 40, 200),
                'RSI': np.random.uniform(20, 80, 200),
                'MACD': np.random.randn(200) * 0.5,
                'GDP_lag': np.random.uniform(1, 4, 200),
                'UNRATE': np.random.uniform(3, 8, 200),
                'Target_5d': np.random.choice([0, 1], 200, p=[0.6, 0.4]),
                'Target_20d': np.random.choice([0, 1], 200, p=[0.55, 0.45])
            }, index=dates)
            stock_data[ticker] = df
        
        return stock_data
    
    @pytest.fixture
    def insufficient_stock_data(self):
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        
        stock_data = {}
        df = pd.DataFrame({
            'Close': np.random.randn(50).cumsum() + 100,
            'VIX': np.random.uniform(10, 40, 50),
            'Target_5d': np.random.choice([0, 1], 50)
        }, index=dates)
        stock_data['TEST'] = df
        
        return stock_data
    
    def test_prepare_training_data_success(self, trainer, sample_stock_data):
        X, y = trainer.prepare_training_data(sample_stock_data, 5)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0
        assert 'Target_5d' not in X.columns
        assert not X.isnull().any().any()
    
    def test_prepare_training_data_insufficient_samples(self, trainer, insufficient_stock_data):
        with pytest.raises(ValueError, match="No valid training data found"):
            trainer.prepare_training_data(insufficient_stock_data, 5, min_samples_per_stock=100)
    
    def test_prepare_training_data_missing_target(self, trainer, sample_stock_data):
        for ticker in sample_stock_data:
            del sample_stock_data[ticker]['Target_5d']
        
        with pytest.raises(ValueError, match="No valid training data found"):
            trainer.prepare_training_data(sample_stock_data, 5)
    
    @patch('train_model.RandomForestClassifier')
    @patch('train_model.StandardScaler')
    @patch('shap.TreeExplainer')
    def test_train_model_success(self, mock_shap, mock_scaler_class, mock_rf_class, 
                                trainer, sample_stock_data):
        mock_rf = Mock()
        mock_rf.fit.return_value = None
        mock_rf.score.return_value = 0.85
        mock_rf.predict.return_value = np.array([0, 1] * 40)  # Match test set size
        mock_rf.feature_importances_ = np.random.rand(6)  # Match feature count
        mock_rf_class.return_value = mock_rf
        
        mock_scaler = Mock()
        mock_scaler.fit_transform.return_value = np.random.rand(160, 6)
        mock_scaler.transform.return_value = np.random.rand(40, 6)
        mock_scaler_class.return_value = mock_scaler
        
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5)
        mock_shap.return_value = mock_explainer
        
        result = trainer.train_model(sample_stock_data, 5)
        
        assert result == mock_rf
        assert 5 in trainer.models
        assert 5 in trainer.scalers
        assert 5 in trainer.feature_columns
        assert 5 in trainer.shap_explainers
        
        mock_rf.fit.assert_called_once()
        mock_scaler.fit_transform.assert_called_once()
        mock_shap.assert_called_once()
    
    def test_categorize_features(self, trainer):
        feature_names = [
            'VIX', 'RSI_14', 'MACD_signal', 'GDP_lag', 'UNRATE', 
            'interaction_VIX_RSI', 'regime_high_vol', 'RSI_sq_transformed'
        ]
        
        categories = trainer._categorize_features(feature_names)
        
        assert 'VIX' in categories['proprietary']
        assert 'RSI_14' in categories['proprietary']
        assert 'GDP_lag' in categories['macro']
        assert 'interaction_VIX_RSI' in categories['interaction']
        assert 'regime_high_vol' in categories['regime']
        assert 'RSI_sq_transformed' in categories['transformed']
    
    def test_calculate_per_stock_metrics(self, trainer, sample_stock_data):
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1] * 50)
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.rand(200, 5)
        
        trainer.models[5] = mock_model
        trainer.scalers[5] = mock_scaler
        trainer.feature_columns[5] = ['VIX', 'RSI', 'MACD', 'GDP_lag', 'UNRATE']
        
        metrics = trainer.calculate_per_stock_metrics(sample_stock_data, 5)
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 2
        for ticker in ['AAPL', 'MSFT']:
            assert ticker in metrics
            assert 'accuracy' in metrics[ticker]
            assert 'samples' in metrics[ticker]
            assert 'class_distribution' in metrics[ticker]
    
    @patch('joblib.dump')
    @patch('os.makedirs')
    def test_save_model(self, mock_makedirs, mock_dump, trainer):
        mock_model = Mock()
        mock_scaler = Mock()
        mock_explainer = Mock()
        
        trainer.models[5] = mock_model
        trainer.scalers[5] = mock_scaler
        trainer.feature_columns[5] = ['VIX', 'RSI']
        trainer.shap_explainers[5] = mock_explainer
        
        trainer.save_model(5, '/test/dir')
        
        mock_makedirs.assert_called_once_with('/test/dir', exist_ok=True)
        assert mock_dump.call_count == 4
    
    def test_analyze_feature_importance(self, trainer):
        feature_names = ['VIX', 'RSI', 'GDP_lag', 'UNRATE']
        
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])
        
        trainer._analyze_feature_importance(feature_names, mock_model, 5)
        
        assert 5 in trainer.feature_importance
        assert 'individual' in trainer.feature_importance[5]
        assert 'by_category' in trainer.feature_importance[5]
        
        individual = trainer.feature_importance[5]['individual']
        assert individual['VIX'] == 0.4
        assert individual['RSI'] == 0.3
        
        by_category = trainer.feature_importance[5]['by_category']
        assert 'proprietary' in by_category
        assert 'macro' in by_category
    
    def test_prepare_training_data_with_nan_targets(self, trainer, sample_stock_data):
        for ticker in sample_stock_data:
            sample_stock_data[ticker]['Target_5d'].iloc[:50] = np.nan
        
        X, y = trainer.prepare_training_data(sample_stock_data, 5)
        
        assert len(X) > 0
        assert len(y) > 0
        assert not y.isnull().any()
    
    def test_train_model_with_insufficient_data(self, trainer, insufficient_stock_data):
        with pytest.raises(ValueError):
            trainer.train_model(insufficient_stock_data, 5)
