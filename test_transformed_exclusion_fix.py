#!/usr/bin/env python3
"""
Test script to verify transformed features are completely excluded from SHAP selection
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings

def test_transformed_exclusion_comprehensive():
    """Test that ALL transformed features are excluded from SHAP selection"""
    print("🔍 TESTING COMPREHENSIVE TRANSFORMED FEATURE EXCLUSION")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        model = co.EnhancedTradingModel()
        
        test_features = [
            'fred_GDP', 'fred_AMERIBOR', 'fred_UNRATE',  # Macro
            'VIX', 'RSI', 'Volume', 'SMA20',  # Proprietary
            'MACD', 'ATR', 'StochRSI',  # Technical
            'fred_GDP_zscore', 'fred_AMERIBOR_zscore',  # Transformed FRED
            'VIX_log', 'RSI_square', 'Volume_sqrt',  # Transformed proprietary
            '[X]fred_zscore', '[X]vix_transform',  # Explicitly marked
            'fred_zscore', 'macro_zscore_indicator',  # zscore patterns
            'log_returns', 'sqrt_volatility', 'norm_momentum'  # Other transforms
        ]
        
        print(f"Testing {len(test_features)} features including transformed variants")
        
        categories = model.categorize_features(test_features)
        
        print("\nFeature categorization results:")
        for cat_name, feat_list in categories.items():
            if feat_list:
                print(f"  {cat_name}: {len(feat_list)} features")
                for feat in feat_list:
                    print(f"    - {feat}")
        
        transformed_features = categories.get('transformed', [])
        expected_transformed = [
            'fred_GDP_zscore', 'fred_AMERIBOR_zscore',
            'VIX_log', 'RSI_square', 'Volume_sqrt',
            '[X]fred_zscore', '[X]vix_transform',
            'fred_zscore', 'macro_zscore_indicator',
            'log_returns', 'sqrt_volatility', 'norm_momentum'
        ]
        
        print(f"\n📊 EXCLUSION ANALYSIS:")
        print(f"Expected transformed: {len(expected_transformed)}")
        print(f"Actually categorized as transformed: {len(transformed_features)}")
        
        missing_transforms = [feat for feat in expected_transformed if feat not in transformed_features]
        unexpected_in_other_cats = []
        
        for cat_name, feat_list in categories.items():
            if cat_name != 'transformed':
                for feat in feat_list:
                    if feat in expected_transformed:
                        unexpected_in_other_cats.append((feat, cat_name))
        
        print(f"\n🔍 LEAK DETECTION:")
        if missing_transforms:
            print(f"❌ Missing from transformed category: {missing_transforms}")
        else:
            print(f"✅ All expected transformed features correctly categorized")
        
        if unexpected_in_other_cats:
            print(f"❌ Transformed features in wrong categories:")
            for feat, cat in unexpected_in_other_cats:
                print(f"    {feat} -> {cat}")
        else:
            print(f"✅ No transformed features leaked into other categories")
        
        print(f"\n🎯 TESTING SHAP SELECTION WITH TRANSFORMS:")
        
        shap_values = np.random.uniform(0.001, 0.05, len(test_features))
        
        top_features = model.get_top_features_by_type(test_features, shap_values, n_per_type=5)
        selected_features = top_features.get('overall_top_2', [])
        
        print(f"Selected features for SHAP:")
        transforms_in_selection = []
        for feat_name, importance, idx in selected_features:
            if feat_name in expected_transformed:
                transforms_in_selection.append(feat_name)
            print(f"  {feat_name}: {importance:.6f}")
        
        if transforms_in_selection:
            print(f"\n❌ CRITICAL: Transformed features in SHAP selection: {transforms_in_selection}")
            return False
        else:
            print(f"\n✅ SUCCESS: No transformed features in SHAP selection")
            return True
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_transformed_exclusion_comprehensive()
    print(f"\n{'🎉 TRANSFORMED EXCLUSION FIX VERIFIED!' if success else '🔧 STILL NEEDS WORK!'}")
