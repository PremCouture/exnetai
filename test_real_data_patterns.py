#!/usr/bin/env python3
"""
Test script to simulate real data patterns that cause only FRED macro features to be selected
"""

import Code_to_Optimize as co
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict

def test_real_data_patterns():
    """Test with realistic feature patterns that mimic real pipeline behavior"""
    print("🔍 TESTING REAL DATA PATTERNS FOR BALANCED SELECTION")
    print("="*70)
    
    warnings.filterwarnings('ignore')
    
    try:
        model = co.EnhancedTradingModel()
        
        real_features = [
            'fred_UMCSENT', 'fred_GDP', 'fred_AMERIBOR', 'fred_UNRATE', 'fred_CPIAUCSL',
            
            'RSI', 'MACD', 'ATR', 'StochRSI', 'CCI', 'MFI', 'Williams_R',
            
            'VIX', 'FNG', 'Momentum125', 'PriceStrength', 'VolumeBreadth',
            
            'OBV_zscore', 'OBV_rank', 'RSI_zscore', 'MACD_rank', 'VIX_pct',
            'fred_GDP_log', 'fred_UMCSENT_square', 'Momentum125_sqrt',
            
            'VIX_X_fred_GDP', 'RSI_X_MACD', 'fred_UMCSENT_X_Momentum125'
        ]
        
        realistic_shap_values = np.array([
            0.45, 0.52, 0.38, 0.41, 0.35,
            0.15, 0.18, 0.12, 0.14, 0.16, 0.13, 0.11,
            0.28, 0.22, 0.19, 0.25, 0.21,
            0.85, 0.78, 0.72, 0.68, 0.65, 0.58, 0.55, 0.48,
            0.32, 0.29, 0.26
        ])
        
        print(f"Testing with {len(real_features)} realistic features:")
        print(f"  FRED macro: 5 features")
        print(f"  Technical: 7 features") 
        print(f"  Proprietary: 5 features")
        print(f"  Transformed: 8 features (HIGH importance - this causes the issue)")
        print(f"  Interaction: 3 features")
        
        categories = model.categorize_features(real_features)
        print(f"\n📊 CATEGORIZATION RESULTS:")
        for category, features in categories.items():
            if features:
                print(f"  {category}: {len(features)} features - {features[:3]}{'...' if len(features) > 3 else ''}")
        
        print(f"\n🎯 TESTING get_top_features_by_type with realistic patterns:")
        top_features = model.get_top_features_by_type(
            real_features, 
            realistic_shap_values, 
            n_per_type=5
        )
        
        print(f"\n📋 SELECTION RESULTS:")
        for category, features in top_features.items():
            if features and category != 'overall_top_2':
                print(f"  {category}: {len(features)} selected")
                for feat_name, importance, idx in features:
                    print(f"    - {feat_name}: {importance:.3f}")
        
        if 'overall_top_2' in top_features:
            print(f"\n🎯 OVERALL TOP 2 SELECTION:")
            overall_features = top_features['overall_top_2']
            for i, (feat_name, importance, idx) in enumerate(overall_features, 1):
                feat_categories = model.categorize_features([feat_name])
                category = next(iter([k for k, v in feat_categories.items() if v]), 'unknown')
                print(f"    {i}. [{category.upper()}] {feat_name}: {importance:.3f}")
            
            categories_in_top2 = []
            for feat_name, importance, idx in overall_features:
                feat_categories = model.categorize_features([feat_name])
                category = next(iter([k for k, v in feat_categories.items() if v]), 'unknown')
                categories_in_top2.append(category)
            
            macro_count = categories_in_top2.count('macro')
            tech_count = categories_in_top2.count('technical')
            prop_count = categories_in_top2.count('proprietary')
            transformed_count = categories_in_top2.count('transformed')
            
            print(f"\n📊 BALANCE ANALYSIS:")
            print(f"  Macro in top 2: {macro_count}")
            print(f"  Technical in top 2: {tech_count}")
            print(f"  Proprietary in top 2: {prop_count}")
            print(f"  Transformed in top 2: {transformed_count}")
            
            if transformed_count == 2:
                print(f"❌ PROBLEM IDENTIFIED: Only transformed features selected!")
                print(f"   This explains why user sees only FRED macro (transformed FRED features)")
            elif macro_count >= 1 and (tech_count >= 1 or prop_count >= 1):
                print(f"✅ SUCCESS: Balanced selection achieved!")
            else:
                print(f"⚠️  WARNING: Suboptimal balance detected")
        
        return True
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_data_patterns()
    
    print("\n" + "="*70)
    if success:
        print("🎉 REAL DATA PATTERN TEST COMPLETED!")
        print("✅ Check the debug output above to understand the selection issue")
    else:
        print("❌ REAL DATA PATTERN TEST FAILED!")
        print("🔧 Review error messages above for debugging")
    
    print("="*70)
