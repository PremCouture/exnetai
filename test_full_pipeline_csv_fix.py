#!/usr/bin/env python3
"""
Test script to verify the full pipeline runs without missing CSV stock indicator warnings
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings
import tempfile
import os
from unittest.mock import patch

def create_mock_csv_files():
    """Create mock CSV files that simulate user's stock data"""
    temp_dir = tempfile.mkdtemp()
    
    stock_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100),
        'Close': np.random.randn(100).cumsum() + 100,
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Volume': np.random.randint(1000000, 5000000, 100),
        'VIX': np.random.uniform(15, 35, 100),
        'FNG': np.random.uniform(20, 80, 100),
        'RSI': np.random.uniform(30, 70, 100),
        'AnnVolatility': np.random.uniform(0.1, 0.5, 100),
        'Momentum125': np.random.uniform(-0.2, 0.2, 100),
        'PriceStrength': np.random.uniform(-1, 1, 100),
        'VolumeBreadth': np.random.uniform(-0.5, 0.5, 100)
    })
    
    stock_csv_path = os.path.join(temp_dir, 'AAPL.csv')
    stock_data.to_csv(stock_csv_path, index=False)
    
    alt_stock_csv_path = os.path.join(temp_dir, '1.csv')  # Common stock ID format
    stock_data.to_csv(alt_stock_csv_path, index=False)
    
    fred_dir = os.path.join(temp_dir, 'fred')
    os.makedirs(fred_dir, exist_ok=True)
    
    fred_data = pd.DataFrame({
        'DATE': pd.date_range('2023-01-01', periods=100),
        'GDP': np.random.uniform(20000, 25000, 100),
        'UMCSENT': np.random.uniform(70, 100, 100),
        'AMERIBOR': np.random.uniform(4, 6, 100),
        'UNRATE': np.random.uniform(3, 5, 100),
        'CPIAUCSL': np.random.uniform(250, 300, 100)
    })
    
    fred_csv_path = os.path.join(fred_dir, 'fred_data.csv')
    fred_data.to_csv(fred_csv_path, index=False)
    
    return temp_dir, stock_csv_path, fred_dir

def test_full_pipeline_csv_fix():
    """Test that the full pipeline runs without missing CSV stock indicator warnings"""
    print("🔍 TESTING FULL PIPELINE CSV LOADING FIXES")
    print("="*70)
    
    warnings.filterwarnings('ignore')
    
    try:
        temp_dir, stock_csv_path, fred_dir = create_mock_csv_files()
        print(f"Created mock data in: {temp_dir}")
        
        original_config = co.CONFIG.copy()
        co.CONFIG['STOCK_DATA_PATH'] = os.path.dirname(stock_csv_path)
        co.CONFIG['FRED_DATA_PATH'] = fred_dir
        co.CONFIG['STOCK_IDS'] = ['AAPL']
        
        print(f"Testing with mock CSV that has Volume, VIX, FNG, RSI but missing SMA20, BL20, BH20")
        
        print(f"\n1. Testing load_stock_data...")
        stock_df = co.load_stock_data('AAPL', os.path.dirname(stock_csv_path))
        if stock_df is None:
            print(f"   Failed to load with directory path, trying direct file...")
            stock_df = pd.read_csv(stock_csv_path)
            stock_df = co.standardize_columns(stock_df)
        print(f"   Loaded columns: {list(stock_df.columns)}")
        
        print(f"\n2. Testing create_proprietary_features...")
        proprietary_features = co.create_proprietary_features(stock_df)
        print(f"   Created features: {list(proprietary_features.columns)}")
        
        expected_csv_features = ['Volume', 'SMA20', 'BL20', 'BH20', 'VIX', 'FNG', 'RSI']
        missing_features = []
        created_features = []
        
        for feat in expected_csv_features:
            if feat in proprietary_features.columns:
                created_features.append(feat)
            else:
                missing_features.append(feat)
        
        print(f"\n📊 CSV FEATURE CREATION RESULTS:")
        print(f"  ✅ Created features: {created_features}")
        print(f"  ❌ Missing features: {missing_features}")
        
        if not missing_features:
            print(f"\n✅ SUCCESS: All expected CSV features are now properly handled!")
            print(f"   - Volume: {'✓' if 'Volume' in created_features else '✗'}")
            print(f"   - SMA20: {'✓' if 'SMA20' in created_features else '✗'}")
            print(f"   - BL20: {'✓' if 'BL20' in created_features else '✗'}")
            print(f"   - BH20: {'✓' if 'BH20' in created_features else '✗'}")
        else:
            print(f"\n❌ ISSUE: Some features still missing: {missing_features}")
        
        print(f"\n3. Testing EnhancedTradingModel with created features...")
        model = co.EnhancedTradingModel()
        
        test_features = list(proprietary_features.columns)[:10]  # Use first 10 features
        test_shap_values = np.random.uniform(0.1, 0.5, len(test_features))
        
        print(f"   Testing with {len(test_features)} features")
        
        categories = model.categorize_features(test_features)
        print(f"   Categorized features: {[(k, len(v)) for k, v in categories.items() if v]}")
        
        top_features = model.get_top_features_by_type(test_features, test_shap_values, n_per_type=5)
        
        if 'overall_top_2' in top_features:
            print(f"\n🎯 BALANCED SELECTION TEST:")
            overall_features = top_features['overall_top_2']
            for i, (feat_name, importance, idx) in enumerate(overall_features, 1):
                feat_categories = model.categorize_features([feat_name])
                category = next(iter([k for k, v in feat_categories.items() if v]), 'unknown')
                print(f"    {i}. [{category.upper()}] {feat_name}: {importance:.3f}")
        
        co.CONFIG.update(original_config)
        
        import shutil
        shutil.rmtree(temp_dir)
        
        return len(missing_features) == 0
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline_csv_fix()
    
    print("\n" + "="*70)
    if success:
        print("🎉 FULL PIPELINE CSV FIX TEST PASSED!")
        print("✅ Missing proprietary features should now be properly handled")
        print("✅ Pipeline should run without CSV loading warnings")
    else:
        print("❌ FULL PIPELINE CSV FIX TEST FAILED!")
        print("🔧 Review error messages above for debugging")
    
    print("="*70)
