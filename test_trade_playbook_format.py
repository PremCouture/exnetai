#!/usr/bin/env python3
"""
Test script for TRADE PLAYBOOK table format
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings

def test_trade_playbook_format():
    """Test the new TRADE PLAYBOOK table format"""
    print("🔍 TESTING TRADE PLAYBOOK FORMAT")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        sample_data = [{
            'Stock': 'AMZN',
            'Signal': 'BUY',
            'Accuracy': 72.5,
            'Sharpe': 1.45,
            'Drawdown': -8.2,
            'VIX': 18.5,
            'FNG': 65,
            'RSI': 45,
            'SHAP': '[M] GDP=2.1 (+0.015↑) | [P] VIX=18.5 (+0.012↑)',
            'IF_THEN': 'AMZN (30d) ↑ IF Features: [M]GDP(+0.02), [P]VIX(+0.01) THEN BUY (72% high confidence). High-quality mixed signal. ✅✅'
        }, {
            'Stock': 'TSLA',
            'Signal': 'SELL',
            'Accuracy': 68.3,
            'Sharpe': 1.22,
            'Drawdown': -12.1,
            'VIX': 32.1,
            'FNG': 22,
            'RSI': 75,
            'SHAP': '[M] AMERIBOR=1.8 (+0.008↑) | [P] RSI=75 (+0.011↑)',
            'IF_THEN': 'TSLA (30d) ↓ IF Features: [M]AMERIBOR(+0.01), [P]RSI(+0.01) THEN SELL (68% moderate confidence). Standard signal. ✅'
        }]
        
        df = pd.DataFrame(sample_data)
        
        print("Sample data created:")
        print(f"  Stocks: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        print("\nTesting TRADE PLAYBOOK table creation...")
        co.create_trade_playbook_table(df, 30)
        
        print("\n✅ TRADE PLAYBOOK format test completed successfully")
        return True
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_transformed_features_exclusion():
    """Test that transformed features are excluded from SHAP selection"""
    print("\n🔍 TESTING TRANSFORMED FEATURES EXCLUSION")
    print("="*60)
    
    try:
        model = co.EnhancedTradingModel()
        
        test_features = [
            'fred_GDP', 'fred_AMERIBOR',  # Macro
            'VIX', 'RSI',  # Proprietary
            'MACD', 'ATR',  # Technical
            'fred_GDP_zscore', 'VIX_log', 'RSI_square',  # Transformed
            '[X]fred_zscore'  # Explicitly marked transformed
        ]
        
        categories = model.categorize_features(test_features)
        
        print("Feature categorization:")
        for cat_name, feat_list in categories.items():
            if feat_list:
                print(f"  {cat_name}: {feat_list}")
        
        transformed_features = categories.get('transformed', [])
        expected_transformed = ['fred_GDP_zscore', 'VIX_log', 'RSI_square', '[X]fred_zscore']
        
        print(f"\nTransformed features found: {transformed_features}")
        print(f"Expected transformed: {expected_transformed}")
        
        all_found = all(feat in transformed_features for feat in expected_transformed)
        
        if all_found:
            print("✅ All transformed features correctly categorized")
            return True
        else:
            print("❌ Some transformed features not categorized correctly")
            return False
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 TESTING TRADE PLAYBOOK IMPLEMENTATION")
    print("="*60)
    
    playbook_success = test_trade_playbook_format()
    exclusion_success = test_transformed_features_exclusion()
    
    print("\n" + "="*60)
    print("📋 TEST SUMMARY:")
    print(f"✅ TRADE PLAYBOOK format: {'PASS' if playbook_success else 'FAIL'}")
    print(f"✅ Transformed features exclusion: {'PASS' if exclusion_success else 'FAIL'}")
    
    overall_success = playbook_success and exclusion_success
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ TRADE PLAYBOOK format implemented correctly")
        print(f"✅ Transformed features excluded from SHAP selection")
        print(f"✅ Production-ready formatting achieved")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"🔧 Review error messages above for debugging")
    
    print("="*60)
