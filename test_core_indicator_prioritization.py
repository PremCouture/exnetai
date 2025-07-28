#!/usr/bin/env python3
"""
Test script to verify core indicator prioritization over derived indicators
"""

import Code_to_Optimize as co
import numpy as np
import warnings

def test_core_indicator_prioritization():
    """Test that core indicators like VIX are prioritized over derived indicators like ATR"""
    print("🔍 TESTING CORE INDICATOR PRIORITIZATION")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        model = co.EnhancedTradingModel()
        
        test_features = [
            'fred_GDP',           # macro - should be selected
            'VIX',                # core proprietary - should be prioritized
            'FNG',                # core proprietary
            'RSI',                # core technical
            'MACD',               # core technical
            'ATR',                # derived technical - should NOT be selected over VIX
            'trend_strength_60d', # derived technical - should NOT be selected over RSI
            'ADX',                # derived technical
            'CCI',                # derived technical
        ]
        
        realistic_shap_values = np.array([
            0.52,  # fred_GDP (macro) - highest
            0.18,  # VIX (core proprietary) - lower than ATR
            0.15,  # FNG (core proprietary)
            0.22,  # RSI (core technical) - lower than trend_strength_60d
            0.20,  # MACD (core technical)
            0.28,  # ATR (derived technical) - higher than VIX!
            0.26,  # trend_strength_60d (derived) - higher than RSI!
            0.24,  # ADX (derived technical)
            0.19,  # CCI (derived technical)
        ])
        
        print(f"Testing prioritization with derived indicators having higher SHAP:")
        for i, (feat, shap_val) in enumerate(zip(test_features, realistic_shap_values)):
            feat_type = "CORE" if feat in ['VIX', 'FNG', 'RSI', 'MACD'] else "DERIVED" if feat != 'fred_GDP' else "MACRO"
            print(f"  {feat}: {shap_val:.3f} ({feat_type})")
        
        categories = model.categorize_features(test_features)
        print(f"\n📊 CATEGORIZATION:")
        for category, features in categories.items():
            if features:
                print(f"  {category}: {features}")
        
        print(f"\n🎯 TESTING PRIORITIZATION LOGIC:")
        top_features = model.get_top_features_by_type(
            test_features, 
            realistic_shap_values, 
            n_per_type=5
        )
        
        if 'overall_top_2' in top_features:
            print(f"\n🎯 OVERALL TOP 2 SELECTION:")
            overall_features = top_features['overall_top_2']
            for i, (feat_name, importance, idx) in enumerate(overall_features, 1):
                feat_categories = model.categorize_features([feat_name])
                category = next(iter([k for k, v in feat_categories.items() if v]), 'unknown')
                feat_type = "CORE" if feat_name in ['VIX', 'FNG', 'RSI', 'MACD'] else "DERIVED" if feat_name != 'fred_GDP' else "MACRO"
                print(f"    {i}. [{category.upper()}] {feat_name}: {importance:.3f} ({feat_type})")
            
            selected_features = [feat_name for feat_name, _, _ in overall_features]
            vix_selected = 'VIX' in selected_features
            atr_selected = 'ATR' in selected_features
            rsi_selected = 'RSI' in selected_features
            trend_selected = 'trend_strength_60d' in selected_features
            
            print(f"\n📊 PRIORITIZATION ANALYSIS:")
            print(f"  VIX (core) selected: {vix_selected}")
            print(f"  ATR (derived) selected: {atr_selected}")
            print(f"  RSI (core) selected: {rsi_selected}")
            print(f"  trend_strength_60d (derived) selected: {trend_selected}")
            
            if vix_selected and not atr_selected:
                print(f"✅ SUCCESS: VIX (core) prioritized over ATR (derived)")
            elif atr_selected and not vix_selected:
                print(f"❌ PROBLEM: ATR (derived) selected over VIX (core)")
            
            if rsi_selected and not trend_selected:
                print(f"✅ SUCCESS: RSI (core) prioritized over trend_strength_60d (derived)")
            elif trend_selected and not rsi_selected:
                print(f"❌ PROBLEM: trend_strength_60d (derived) selected over RSI (core)")
        
        return True
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_core_indicator_prioritization()
    
    print("\n" + "="*60)
    if success:
        print("🎉 CORE INDICATOR PRIORITIZATION TEST COMPLETED!")
        print("✅ Check the analysis above to verify prioritization logic")
    else:
        print("❌ CORE INDICATOR PRIORITIZATION TEST FAILED!")
        print("🔧 Review error messages above for debugging")
    
    print("="*60)
