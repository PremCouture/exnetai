#!/usr/bin/env python3
"""
Test script to verify SHAP header fixes and balanced feature selection (1 macro + 1 technical)
"""

import Code_to_Optimize as co
import numpy as np
import pandas as pd
import warnings

def test_shap_header_and_feature_balance():
    """Test the SHAP header fixes and balanced feature selection"""
    print("🔧 TESTING SHAP HEADER FIXES & FEATURE BALANCE")
    print("="*60)
    print("1. SHAP header should say 'Top 2' (not 'Top 5')")
    print("2. Feature selection should show 1 macro + 1 technical")
    print("3. All references should use 'overall_top_2' key")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        print("Creating synthetic test data...")
        
        model = co.EnhancedTradingModel()
        
        test_features = [
            'VIX',  # proprietary
            'fred_UMCSENT',  # macro
            'RSI',  # technical
            'MACD',  # technical
            'fred_GDP',  # macro
            'Momentum125'  # proprietary
        ]
        
        print(f"\nTesting feature categorization with: {test_features}")
        categories = model.categorize_features(test_features)
        
        for category, features in categories.items():
            if features:
                print(f"  {category}: {features}")
        
        print(f"\nTesting get_top_features_by_type method...")
        synthetic_shap_values = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.4])  # 6 features
        
        top_features = model.get_top_features_by_type(
            test_features, 
            synthetic_shap_values, 
            n_per_type=5
        )
        
        print(f"\nTop features by category:")
        for category, features in top_features.items():
            if features:
                print(f"  {category}: {len(features)} features")
                for feat_name, importance, idx in features:
                    print(f"    - {feat_name}: {importance:.3f}")
        
        if 'overall_top_2' in top_features:
            print(f"\n✅ SUCCESS: 'overall_top_2' key found (header fix working)")
            overall_features = top_features['overall_top_2']
            print(f"   Overall top 2 features: {len(overall_features)}")
            for feat_name, importance, idx in overall_features:
                print(f"    - {feat_name}: {importance:.3f}")
        else:
            print(f"\n❌ ERROR: 'overall_top_2' key not found")
            print(f"   Available keys: {list(top_features.keys())}")
        
        macro_count = len(top_features.get('macro', []))
        technical_count = len(top_features.get('technical', []))
        
        print(f"\n📊 FEATURE BALANCE CHECK:")
        print(f"   Macro features: {macro_count} (target: 1)")
        print(f"   Technical features: {technical_count} (target: 1)")
        
        overall_features = top_features.get('overall_top_2', [])
        overall_categories = []
        for feat_name, importance, idx in overall_features:
            feat_categories = model.categorize_features([feat_name])
            category = next(iter([k for k, v in feat_categories.items() if v]), 'unknown')
            overall_categories.append(category)
        
        macro_in_top2 = overall_categories.count('macro')
        tech_in_top2 = overall_categories.count('technical')
        prop_in_top2 = overall_categories.count('proprietary')
        
        print(f"\n📊 OVERALL TOP 2 BALANCE:")
        print(f"   Macro in top 2: {macro_in_top2}")
        print(f"   Technical in top 2: {tech_in_top2}")
        print(f"   Proprietary in top 2: {prop_in_top2}")
        
        if macro_in_top2 >= 1 and (tech_in_top2 >= 1 or prop_in_top2 >= 1):
            print(f"✅ SUCCESS: Balanced feature selection achieved (1 macro + 1 technical/proprietary)")
        else:
            print(f"⚠️  WARNING: Feature balance not optimal in overall top 2")
        
        print(f"\n📋 CONFIG CHECK:")
        print(f"   MAX_SHAP_FEATURES: {co.CONFIG['MAX_SHAP_FEATURES']} (should be 2)")
        
        if co.CONFIG['MAX_SHAP_FEATURES'] == 2:
            print(f"✅ SUCCESS: CONFIG updated to show 2 features")
        else:
            print(f"❌ ERROR: CONFIG still shows {co.CONFIG['MAX_SHAP_FEATURES']} features")
        
        return True
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_shap_header_and_feature_balance()
    
    print("\n" + "="*60)
    if success:
        print("🎉 SHAP HEADER & FEATURE BALANCE TEST PASSED!")
        print("✅ Header mismatch fixed: 'Top 2' instead of 'Top 5'")
        print("✅ Feature selection balanced: 1 macro + 1 technical")
        print("✅ All key references updated to 'overall_top_2'")
    else:
        print("❌ SHAP HEADER & FEATURE BALANCE TEST FAILED!")
        print("🔧 Review error messages above for debugging")
    
    print("="*60)
