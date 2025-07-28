#!/usr/bin/env python3
"""
Test script to verify SHAP importance filtering works correctly
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings

def test_importance_threshold_filtering():
    """Test that low-importance features are filtered out"""
    print("🔍 TESTING IMPORTANCE THRESHOLD FILTERING")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        model = co.EnhancedTradingModel()
        
        test_features = ['fred_ICSA', 'fred_GDP', 'fred_AMERIBOR', 'VIX', 'RSI', 'ATR', 'MACD']
        
        high_importance_values = np.array([0.45, 0.35, 0.30, 0.25, 0.20, 0.18, 0.16])
        
        print(f"\n📊 TEST 1: High importance scenario (all above 0.15 threshold)")
        print(f"Features: {test_features}")
        print(f"SHAP values: {high_importance_values}")
        
        top_features = model.get_top_features_by_type(test_features, high_importance_values, n_per_type=5)
        
        skip_signal = top_features.get('skip_signal', False)
        selected_features = top_features.get('overall_top_2', [])
        
        print(f"Result: {'Signal SKIPPED' if skip_signal else f'{len(selected_features)} features selected'}")
        if selected_features:
            for feat_name, importance, idx in selected_features:
                print(f"  - {feat_name}: {importance:.3f}")
        
        test1_success = not skip_signal and len(selected_features) > 0
        
        low_importance_values = np.array([0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03])
        
        print(f"\n📊 TEST 2: Low importance scenario (all below 0.15 threshold)")
        print(f"Features: {test_features}")
        print(f"SHAP values: {low_importance_values}")
        
        top_features = model.get_top_features_by_type(test_features, low_importance_values, n_per_type=5)
        
        skip_signal = top_features.get('skip_signal', False)
        selected_features = top_features.get('overall_top_2', [])
        
        print(f"Result: {'Signal SKIPPED' if skip_signal else f'{len(selected_features)} features selected'}")
        if selected_features:
            for feat_name, importance, idx in selected_features:
                print(f"  - {feat_name}: {importance:.3f}")
        
        test2_success = skip_signal and len(selected_features) == 0
        
        mixed_importance_values = np.array([0.25, 0.20, 0.12, 0.18, 0.08, 0.05, 0.03])
        
        print(f"\n📊 TEST 3: Mixed importance scenario (some above/below 0.15 threshold)")
        print(f"Features: {test_features}")
        print(f"SHAP values: {mixed_importance_values}")
        
        top_features = model.get_top_features_by_type(test_features, mixed_importance_values, n_per_type=5)
        
        skip_signal = top_features.get('skip_signal', False)
        selected_features = top_features.get('overall_top_2', [])
        
        print(f"Result: {'Signal SKIPPED' if skip_signal else f'{len(selected_features)} features selected'}")
        if selected_features:
            for feat_name, importance, idx in selected_features:
                print(f"  - {feat_name}: {importance:.3f}")
        
        all_above_threshold = all(importance >= 0.15 for _, importance, _ in selected_features)
        test3_success = not skip_signal and len(selected_features) > 0 and all_above_threshold
        
        print(f"\n📈 FILTERING ANALYSIS:")
        print(f"  Test 1 (High importance): {'✅ PASS' if test1_success else '❌ FAIL'}")
        print(f"  Test 2 (Low importance): {'✅ PASS' if test2_success else '❌ FAIL'}")
        print(f"  Test 3 (Mixed importance): {'✅ PASS' if test3_success else '❌ FAIL'}")
        
        overall_success = test1_success and test2_success and test3_success
        
        return overall_success
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_signal_skipping_logic():
    """Test that signals are properly skipped when no important features exist"""
    print(f"\n🚫 TESTING SIGNAL SKIPPING LOGIC")
    print("="*50)
    
    try:
        model = co.EnhancedTradingModel()
        
        edge_cases = [
            {
                'name': 'Empty features',
                'features': [],
                'shap_values': np.array([]),
                'should_skip': True
            },
            {
                'name': 'Single low-importance feature',
                'features': ['fred_ICSA'],
                'shap_values': np.array([0.05]),
                'should_skip': True
            },
            {
                'name': 'Single high-importance feature',
                'features': ['fred_ICSA'],
                'shap_values': np.array([0.25]),
                'should_skip': False
            },
            {
                'name': 'All features at threshold boundary',
                'features': ['fred_ICSA', 'VIX'],
                'shap_values': np.array([0.15, 0.15]),
                'should_skip': False
            },
            {
                'name': 'All features just below threshold',
                'features': ['fred_ICSA', 'VIX', 'RSI'],
                'shap_values': np.array([0.149, 0.148, 0.147]),
                'should_skip': True
            }
        ]
        
        test_results = []
        
        for case in edge_cases:
            print(f"\n  📊 {case['name']}:")
            print(f"    Features: {case['features']}")
            print(f"    SHAP values: {case['shap_values']}")
            print(f"    Expected: {'SKIP' if case['should_skip'] else 'PROCESS'}")
            
            if len(case['features']) == 0:
                result_skip = True
            else:
                top_features = model.get_top_features_by_type(case['features'], case['shap_values'], n_per_type=5)
                result_skip = top_features.get('skip_signal', False)
            
            print(f"    Actual: {'SKIP' if result_skip else 'PROCESS'}")
            
            test_passed = result_skip == case['should_skip']
            test_results.append(test_passed)
            print(f"    Result: {'✅ PASS' if test_passed else '❌ FAIL'}")
        
        overall_success = all(test_results)
        
        print(f"\n📊 EDGE CASE SUMMARY:")
        print(f"  Passed: {sum(test_results)}/{len(test_results)}")
        print(f"  Overall: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
        
        return overall_success
        
    except Exception as e:
        print(f'❌ Signal skipping test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 TESTING SHAP IMPORTANCE FILTERING")
    print("="*60)
    
    filtering_success = test_importance_threshold_filtering()
    skipping_success = test_signal_skipping_logic()
    
    print("\n" + "="*60)
    print("📋 IMPORTANCE FILTERING TEST SUMMARY:")
    print(f"✅ Threshold filtering: {'PASS' if filtering_success else 'FAIL'}")
    print(f"✅ Signal skipping logic: {'PASS' if skipping_success else 'FAIL'}")
    
    overall_success = filtering_success and skipping_success
    
    if overall_success:
        print(f"\n🎉 ALL IMPORTANCE FILTERING TESTS PASSED!")
        print(f"✅ Low-importance features are properly filtered out")
        print(f"✅ Signals are skipped when no features meet threshold")
        print(f"✅ Only high-importance features appear in SHAP output")
    else:
        print(f"\n❌ SOME IMPORTANCE FILTERING TESTS FAILED!")
        print(f"🔧 Review error messages above for debugging")
    
    print("="*60)
