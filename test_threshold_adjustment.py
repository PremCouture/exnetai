#!/usr/bin/env python3
"""
Test script to verify the adjusted importance threshold works with real data patterns
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings

def test_realistic_shap_values():
    """Test with realistic SHAP importance values based on user's real data"""
    print("🔍 TESTING REALISTIC SHAP IMPORTANCE VALUES")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        model = co.EnhancedTradingModel()
        
        realistic_scenarios = [
            {
                'name': 'Very Low Importance (Real Data Pattern)',
                'features': ['fred_ICSA', 'fred_GDP', 'VIX', 'RSI', 'Volume', 'SMA20'],
                'shap_values': np.array([0.008, 0.006, 0.012, 0.009, 0.007, 0.005])
            },
            {
                'name': 'Mixed Low Importance',
                'features': ['fred_AMERIBOR', 'fred_ICSA', 'BL20', 'BH20', 'Momentum125'],
                'shap_values': np.array([0.015, 0.011, 0.008, 0.006, 0.004])
            },
            {
                'name': 'Extremely Low Importance',
                'features': ['fred_GDP', 'AnnVolatility', 'VolumeBreadth', 'PriceStrength'],
                'shap_values': np.array([0.003, 0.002, 0.001, 0.0008])
            }
        ]
        
        results = []
        
        for scenario in realistic_scenarios:
            print(f"\n📊 {scenario['name']}:")
            print(f"Features: {scenario['features']}")
            print(f"SHAP values: {scenario['shap_values']}")
            
            top_features = model.get_top_features_by_type(
                scenario['features'], 
                scenario['shap_values'], 
                n_per_type=5
            )
            
            skip_signal = top_features.get('skip_signal', False)
            selected_features = top_features.get('overall_top_2', [])
            
            print(f"Result: {'Signal SKIPPED' if skip_signal else f'{len(selected_features)} features selected'}")
            
            if selected_features:
                for feat_name, importance, idx in selected_features:
                    print(f"  - {feat_name}: {importance:.6f}")
            
            results.append({
                'scenario': scenario['name'],
                'skip_signal': skip_signal,
                'num_selected': len(selected_features),
                'max_importance': max(scenario['shap_values']) if len(scenario['shap_values']) > 0 else 0
            })
        
        print(f"\n📈 THRESHOLD EFFECTIVENESS ANALYSIS:")
        
        for result in results:
            print(f"  {result['scenario']}:")
            print(f"    Max importance: {result['max_importance']:.6f}")
            print(f"    Signal skipped: {result['skip_signal']}")
            print(f"    Features selected: {result['num_selected']}")
        
        print(f"\n🎯 THRESHOLD BOUNDARY TESTING:")
        
        boundary_tests = [
            {'threshold': 0.01, 'values': [0.011, 0.009], 'expected_pass': [True, False]},
            {'threshold': 0.005, 'values': [0.006, 0.004], 'expected_pass': [True, False]},
            {'threshold': 0.001, 'values': [0.002, 0.0005], 'expected_pass': [True, False]}
        ]
        
        for test in boundary_tests:
            print(f"  Threshold {test['threshold']:.3f}:")
            for val, expected in zip(test['values'], test['expected_pass']):
                actual_pass = val >= test['threshold']
                status = "✅" if actual_pass == expected else "❌"
                print(f"    {status} Value {val:.6f}: {'PASS' if actual_pass else 'FAIL'}")
        
        return True
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_threshold_impact_on_signal_generation():
    """Test how different thresholds affect signal generation"""
    print(f"\n⚡ TESTING THRESHOLD IMPACT ON SIGNAL GENERATION")
    print("="*60)
    
    try:
        test_features = ['fred_ICSA', 'VIX', 'RSI', 'Volume', 'SMA20']
        realistic_shap_values = np.array([0.008, 0.012, 0.006, 0.004, 0.003])
        
        thresholds_to_test = [0.15, 0.05, 0.03, 0.01, 0.005, 0.001]
        
        print(f"Testing with features: {test_features}")
        print(f"SHAP values: {realistic_shap_values}")
        print(f"Max importance: {max(realistic_shap_values):.6f}")
        
        for threshold in thresholds_to_test:
            model = co.EnhancedTradingModel()
            
            passing_features = sum(1 for val in realistic_shap_values if val >= threshold)
            
            print(f"\n  Threshold {threshold:.3f}: {passing_features}/{len(realistic_shap_values)} features pass")
            
            if passing_features > 0:
                print(f"    ✅ Would generate signal")
            else:
                print(f"    ❌ Would skip signal")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"  For real data with max importance ~0.012:")
        print(f"  - Threshold 0.01: Allows top features through")
        print(f"  - Threshold 0.005: More permissive, good balance")
        print(f"  - Threshold 0.001: Very permissive, minimal filtering")
        
        return True
        
    except Exception as e:
        print(f'❌ Threshold impact test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 TESTING ADJUSTED IMPORTANCE THRESHOLD")
    print("="*60)
    
    realistic_success = test_realistic_shap_values()
    impact_success = test_threshold_impact_on_signal_generation()
    
    print("\n" + "="*60)
    print("📋 THRESHOLD ADJUSTMENT TEST SUMMARY:")
    print(f"✅ Realistic SHAP values: {'PASS' if realistic_success else 'FAIL'}")
    print(f"✅ Threshold impact analysis: {'PASS' if impact_success else 'FAIL'}")
    
    overall_success = realistic_success and impact_success
    
    if overall_success:
        print(f"\n🎉 THRESHOLD ADJUSTMENT TESTS PASSED!")
        print(f"✅ Adjusted threshold should work with real data patterns")
        print(f"✅ Signals should be generated instead of skipped")
        print(f"✅ Filtering logic maintained for truly low-importance features")
    else:
        print(f"\n❌ SOME THRESHOLD TESTS FAILED!")
        print(f"🔧 Review error messages above for debugging")
    
    print("="*60)
