#!/usr/bin/env python3
"""
Test script to verify updated proprietary feature list matches user's CSV stock indicators
"""

import Code_to_Optimize as co
import numpy as np
import warnings

def test_updated_proprietary_features():
    """Test that proprietary features match user's CSV stock indicators"""
    print("🔍 TESTING UPDATED PROPRIETARY FEATURE LIST")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        model = co.EnhancedTradingModel()
        
        csv_stock_indicators = [
            'Volume', 'SMA20', 'BL20', 'BH20', 'RSI', 'AnnVolatility', 
            'Momentum125', 'PriceStrength', 'VolumeBreadth', 'VIX', 'FNG'
        ]
        
        test_features = [
            'fred_GDP',           # macro - should be selected
            'fred_AMERIBOR',      # macro
            'Volume',             # proprietary (from CSV)
            'SMA20',              # proprietary (from CSV)
            'VIX',                # proprietary (from CSV) - core indicator
            'FNG',                # proprietary (from CSV) - core indicator
            'RSI',                # proprietary (from CSV) - core indicator
            'AnnVolatility',      # proprietary (from CSV) - core indicator
            'Momentum125',        # proprietary (from CSV) - core indicator
            'ATR',                # technical (not in CSV) - should be deprioritized
            'MACD',               # technical (not in CSV) - should be deprioritized
        ]
        
        realistic_shap_values = np.array([
            0.52,  # fred_GDP (macro) - highest
            0.41,  # fred_AMERIBOR (macro)
            0.15,  # Volume (proprietary)
            0.18,  # SMA20 (proprietary)
            0.22,  # VIX (proprietary) - should be selected over ATR
            0.19,  # FNG (proprietary)
            0.25,  # RSI (proprietary)
            0.17,  # AnnVolatility (proprietary)
            0.16,  # Momentum125 (proprietary)
            0.28,  # ATR (technical) - higher than VIX but should NOT be selected
            0.26,  # MACD (technical) - higher than RSI but should NOT be selected
        ])
        
        print(f"Testing with user's CSV stock indicators:")
        for indicator in csv_stock_indicators:
            print(f"  ✓ {indicator}")
        
        print(f"\nTesting prioritization (proprietary over technical):")
        for i, (feat, shap_val) in enumerate(zip(test_features, realistic_shap_values)):
            feat_type = "PROPRIETARY" if feat in csv_stock_indicators else "TECHNICAL" if feat not in ['fred_GDP', 'fred_AMERIBOR'] else "MACRO"
            print(f"  {feat}: {shap_val:.3f} ({feat_type})")
        
        categories = model.categorize_features(test_features)
        print(f"\n📊 CATEGORIZATION:")
        for category, features in categories.items():
            if features:
                print(f"  {category}: {features}")
        
        print(f"\n🎯 TESTING BALANCED SELECTION:")
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
                feat_type = "CSV_PROPRIETARY" if feat_name in csv_stock_indicators else "NON_CSV"
                print(f"    {i}. [{category.upper()}] {feat_name}: {importance:.3f} ({feat_type})")
            
            selected_features = [feat_name for feat_name, _, _ in overall_features]
            
            macro_selected = any(feat in ['fred_GDP', 'fred_AMERIBOR'] for feat in selected_features)
            csv_proprietary_selected = any(feat in csv_stock_indicators for feat in selected_features)
            non_csv_selected = any(feat in ['ATR', 'MACD'] for feat in selected_features)
            
            print(f"\n📊 SELECTION ANALYSIS:")
            print(f"  Macro selected: {macro_selected}")
            print(f"  CSV proprietary selected: {csv_proprietary_selected}")
            print(f"  Non-CSV technical selected: {non_csv_selected}")
            
            if macro_selected and csv_proprietary_selected and not non_csv_selected:
                print(f"✅ SUCCESS: Perfect balance - 1 macro + 1 CSV proprietary")
            elif macro_selected and csv_proprietary_selected:
                print(f"✅ GOOD: Balanced selection achieved")
            else:
                print(f"❌ ISSUE: Selection doesn't match user requirements")
        
        return True
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_updated_proprietary_features()
    
    print("\n" + "="*60)
    if success:
        print("🎉 UPDATED PROPRIETARY FEATURE TEST COMPLETED!")
        print("✅ Check the analysis above to verify CSV stock indicator prioritization")
    else:
        print("❌ UPDATED PROPRIETARY FEATURE TEST FAILED!")
        print("🔧 Review error messages above for debugging")
    
    print("="*60)
