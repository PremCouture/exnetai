#!/usr/bin/env python3
"""
Test script to verify CSV loading fixes for missing proprietary features
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings

def test_csv_loading_fix():
    """Test that missing CSV features are now properly handled"""
    print("🔍 TESTING CSV LOADING FIXES")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        mock_csv_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Close': np.random.randn(100).cumsum() + 100,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Volume': np.random.randint(1000000, 5000000, 100),
            'VIX': np.random.uniform(15, 35, 100),
            'FNG': np.random.uniform(20, 80, 100),
            'RSI': np.random.uniform(30, 70, 100)
        })
        
        print(f"Mock CSV columns: {list(mock_csv_data.columns)}")
        
        standardized_df = co.standardize_columns(mock_csv_data.copy())
        print(f"After standardize_columns: {list(standardized_df.columns)}")
        
        proprietary_features = co.create_proprietary_features(standardized_df)
        print(f"Created proprietary features: {list(proprietary_features.columns)}")
        
        expected_features = ['Volume', 'SMA20', 'BL20', 'BH20', 'VIX', 'FNG', 'RSI']
        missing_features = []
        created_features = []
        
        for feat in expected_features:
            if feat in proprietary_features.columns:
                created_features.append(feat)
            else:
                missing_features.append(feat)
        
        print(f"\n📊 FEATURE CREATION RESULTS:")
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
        
        return len(missing_features) == 0
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_csv_loading_fix()
    
    print("\n" + "="*60)
    if success:
        print("🎉 CSV LOADING FIX TEST PASSED!")
        print("✅ Missing proprietary features should now be properly handled")
    else:
        print("❌ CSV LOADING FIX TEST FAILED!")
        print("🔧 Review error messages above for debugging")
    
    print("="*60)
