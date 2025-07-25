import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from predict import Predictor

class TestPredictor:
    
    @pytest.fixture
    def predictor(self):
        return Predictor()
    
    @pytest.fixture
    def sample_features(self):
        return pd.DataFrame({
            'VIX': [20.5],
            'RSI': [65.2],
            'MACD': [0.15],
            'GDP_lag': [2.1],
            'UNRATE': [3.8]
        })
    
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        return model
    
    @pytest.fixture
    def mock_scaler(self):
        scaler = Mock()
        scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        return scaler
    
    @pytest.fixture
    def mock_shap_explainer(self):
        explainer = Mock()
        explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, -0.1, 0.05]])
        return explainer
    
    def test_predict_proba_success(self, predictor, sample_features, mock_model, mock_scaler):
        prediction_days = 5
        predictor.models[prediction_days] = mock_model
        predictor.scalers[prediction_days] = mock_scaler
        predictor.feature_columns[prediction_days] = list(sample_features.columns)
        
        result = predictor.predict_proba(sample_features, prediction_days)
        
        assert result is not None
        assert len(result) == 2
        assert result[1] == 0.7
        mock_scaler.transform.assert_called_once()
        mock_model.predict_proba.assert_called_once()
    
    def test_predict_proba_missing_model(self, predictor, sample_features):
        result = predictor.predict_proba(sample_features, 999)
        assert result is None
    
    def test_predict_proba_invalid_features(self, predictor, mock_model, mock_scaler):
        prediction_days = 5
        predictor.models[prediction_days] = mock_model
        predictor.scalers[prediction_days] = mock_scaler
        predictor.feature_columns[prediction_days] = ['VIX', 'RSI']
        
        result = predictor.predict_proba("invalid", prediction_days)
        assert result is None
    
    def test_predict_proba_feature_alignment(self, predictor, mock_model, mock_scaler):
        prediction_days = 5
        predictor.models[prediction_days] = mock_model
        predictor.scalers[prediction_days] = mock_scaler
        predictor.feature_columns[prediction_days] = ['VIX', 'RSI', 'MACD', 'NewFeature']
        
        features = pd.DataFrame({'VIX': [20.5], 'RSI': [65.2]})
        
        result = predictor.predict_proba(features, prediction_days)
        
        assert result is not None
        mock_scaler.transform.assert_called_once()
        transformed_features = mock_scaler.transform.call_args[0][0]
        assert transformed_features.shape[1] == 4
    
    def test_get_signal_shap_explanation_success(self, predictor, sample_features, 
                                               mock_model, mock_scaler, mock_shap_explainer):
        prediction_days = 5
        predictor.models[prediction_days] = mock_model
        predictor.scalers[prediction_days] = mock_scaler
        predictor.shap_explainers[prediction_days] = mock_shap_explainer
        predictor.feature_columns[prediction_days] = list(sample_features.columns)
        
        result = predictor.get_signal_shap_explanation(sample_features, prediction_days)
        
        assert result is not None
        assert 'prediction_probability' in result
        assert 'shap_values' in result
        assert 'top_features' in result
        assert 'feature_values' in result
        
        assert len(result['prediction_probability']) == 2
        assert isinstance(result['top_features'], dict)
    
    def test_get_signal_shap_explanation_missing_explainer(self, predictor, sample_features,
                                                         mock_model, mock_scaler):
        prediction_days = 5
        predictor.models[prediction_days] = mock_model
        predictor.scalers[prediction_days] = mock_scaler
        predictor.feature_columns[prediction_days] = list(sample_features.columns)
        
        result = predictor.get_signal_shap_explanation(sample_features, prediction_days)
        assert result is None
    
    def test_categorize_features(self, predictor):
        feature_names = [
            'VIX', 'RSI_14', 'MACD_signal', 'GDP_lag', 'UNRATE', 
            'interaction_VIX_RSI', 'regime_high_vol', 'RSI_sq_transformed'
        ]
        
        categories = predictor._categorize_features(feature_names)
        
        assert 'VIX' in categories['proprietary']
        assert 'RSI_14' in categories['proprietary']
        assert 'GDP_lag' in categories['macro']
        assert 'interaction_VIX_RSI' in categories['interaction']
        assert 'regime_high_vol' in categories['regime']
        assert 'RSI_sq_transformed' in categories['transformed']
    
    def test_get_top_features_by_type(self, predictor):
        feature_names = ['VIX', 'RSI', 'GDP_lag', 'UNRATE', 'interaction_test']
        shap_values = np.array([0.5, -0.3, 0.2, -0.1, 0.4])
        
        result = predictor._get_top_features_by_type(feature_names, shap_values, n_per_type=2)
        
        assert isinstance(result, dict)
        assert 'proprietary' in result
        assert 'macro' in result
        assert 'interaction' in result
        
        assert len(result['proprietary']) <= 2
        assert result['proprietary'][0]['feature'] == 'VIX'
        assert result['proprietary'][0]['shap_value'] == 0.5
    
    def test_load_model_success(self, predictor):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            scaler_path = os.path.join(temp_dir, 'scaler.pkl')
            features_path = os.path.join(temp_dir, 'features.pkl')
            shap_path = os.path.join(temp_dir, 'shap.pkl')
            
            mock_model = Mock()
            mock_scaler = Mock()
            mock_features = ['VIX', 'RSI']
            mock_explainer = Mock()
            
            with patch('joblib.load') as mock_load:
                mock_load.side_effect = [mock_model, mock_scaler, mock_features, mock_explainer]
                
                predictor.load_model(5, model_path, scaler_path, features_path, shap_path)
                
                assert predictor.models[5] == mock_model
                assert predictor.scalers[5] == mock_scaler
                assert predictor.feature_columns[5] == mock_features
                assert predictor.shap_explainers[5] == mock_explainer
    
    def test_load_model_failure(self, predictor):
        with pytest.raises(Exception):
            predictor.load_model(5, 'nonexistent.pkl', 'nonexistent.pkl', 
                               'nonexistent.pkl', 'nonexistent.pkl')
    
    def test_predict_proba_with_nan_handling(self, predictor, mock_model, mock_scaler):
        prediction_days = 5
        predictor.models[prediction_days] = mock_model
        predictor.scalers[prediction_days] = mock_scaler
        predictor.feature_columns[prediction_days] = ['VIX', 'RSI']
        
        features_with_nan = pd.DataFrame({
            'VIX': [np.nan],
            'RSI': [65.2]
        })
        
        result = predictor.predict_proba(features_with_nan, prediction_days)
        
        assert result is not None
        mock_scaler.transform.assert_called_once()
        transformed_features = mock_scaler.transform.call_args[0][0]
        assert not np.isnan(transformed_features).any()
    
    def test_shap_explanation_with_list_shap_values(self, predictor, sample_features,
                                                   mock_model, mock_scaler):
        prediction_days = 5
        predictor.models[prediction_days] = mock_model
        predictor.scalers[prediction_days] = mock_scaler
        predictor.feature_columns[prediction_days] = list(sample_features.columns)
        
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = [
            np.array([[0.1, -0.2, 0.3, -0.1, 0.05]]),
            np.array([[0.2, -0.1, 0.4, -0.2, 0.1]])
        ]
        predictor.shap_explainers[prediction_days] = mock_explainer
        
        result = predictor.get_signal_shap_explanation(sample_features, prediction_days)
        
        assert result is not None
        assert 'shap_values' in result
        assert len(result['shap_values']) == 5
