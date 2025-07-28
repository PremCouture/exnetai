#!/usr/bin/env python3
"""
Debug script to investigate why macro features are being excluded from balanced selection
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings

def debug_macro_exclusion():
    """Debug why macro features are not appearing in final SHAP selection"""
    print("🔍 DEBUGGING MACRO FEATURE EXCLUSION")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        model = co.EnhancedTradingModel()
        
        test_features = [
            'fred_ICSA', 'fred_GDP', 'fred_AMERIBOR', 'fred_UNRATE',
            'VIX', 'RSI', 'Volume', 'SMA20', 'BL20', 'BH20',
            'AnnVolatility', 'Momentum125', 'PriceStrength', 'VolumeBreadth',
            'MACD', 'ATR', 'StochRSI', 'PRICE_TO_SMA_200', 'SMA_10_SLOPE', 'Ret_20d'
        ]
        
        shap_values = np.array([
            0.002, 0.001, 0.0015, 0.0008,
            0.008, 0.012, 0.006, 0.009, 0.007, 0.005,
            0.031, 0.015, 0.008, 0.006,
            0.004, 0.003, 0.002, 0.018, 0.032, 0.011
        ])
        
        print(f"Test features: {len(test_features)}")
        print(f"SHAP range: {min(shap_values):.6f} - {max(shap_values):.6f}")
        
        categories = model.categorize_features(test_features)
        print(f"\nFeature categorization:")
        for cat_name, feat_list in categories.items():
            if feat_list:
                print(f"  {cat_name}: {len(feat_list)} features - {feat_list}")
        
        adaptive_threshold = model.calculate_adaptive_threshold(shap_values)
        print(f"\nAdaptive threshold: {adaptive_threshold:.6f}")
        
        macro_features = categories.get('macro', [])
        prop_features = categories.get('proprietary', [])
        tech_features = categories.get('technical', [])
        
        print(f"\nThreshold analysis:")
        for i, (feat_name, importance) in enumerate(zip(test_features, shap_values)):
            passes = importance >= adaptive_threshold
            if feat_name in macro_features:
                category = "MACRO"
            elif feat_name in prop_features:
                category = "PROP"
            elif feat_name in tech_features:
                category = "TECH"
            else:
                category = "OTHER"
            
            status = "✅ PASS" if passes else "❌ FAIL"
            print(f"  {status} {category:5} {feat_name:15} {importance:.6f}")
        
        print(f"\n" + "="*50)
        print("TESTING ACTUAL SELECTION LOGIC:")
        
        top_features = model.get_top_features_by_type(test_features, shap_values, n_per_type=5)
        
        skip_signal = top_features.get('skip_signal', False)
        selected_features = top_features.get('overall_top_2', [])
        
        print(f"Result: {'SIGNAL SKIPPED' if skip_signal else f'{len(selected_features)} features selected'}")
        
        if selected_features:
            print(f"\nSelected features:")
            for feat_name, importance, idx in selected_features:
                # Determine category
                if feat_name in macro_features:
                    category = "MACRO"
                elif feat_name in prop_features:
                    category = "PROPRIETARY"
                elif feat_name in tech_features:
                    category = "TECHNICAL"
                else:
                    category = "OTHER"
                print(f"  [{category[0]}] {feat_name}: {importance:.6f}")
        
        macro_count = sum(1 for feat_name, _, _ in selected_features if feat_name in macro_features)
        prop_count = sum(1 for feat_name, _, _ in selected_features if feat_name in prop_features)
        tech_count = sum(1 for feat_name, _, _ in selected_features if feat_name in tech_features)
        
        print(f"\nSelection breakdown:")
        print(f"  Macro: {macro_count}")
        print(f"  Proprietary: {prop_count}")
        print(f"  Technical: {tech_count}")
        
        if macro_count == 1 and prop_count == 1 and tech_count == 0:
            print(f"  ✅ CORRECT: 1 macro + 1 proprietary")
        else:
            print(f"  ❌ INCORRECT: Should be 1 macro + 1 proprietary")
            print(f"  🔧 Need to fix balanced selection logic")
        
        return True
        
    except Exception as e:
        print(f'❌ Debug failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_macro_exclusion()
    print(f"\n{'✅ DEBUG COMPLETED!' if success else '❌ DEBUG FAILED!'}")
