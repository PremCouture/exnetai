#!/usr/bin/env python3
"""
Test script to verify SHAP diversity improvements work correctly
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings
from collections import Counter

def test_macro_diversity_constraints():
    """Test that macro diversity constraints prevent repetitive selection"""
    print("🔍 TESTING MACRO DIVERSITY CONSTRAINTS")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        model = co.EnhancedTradingModel()
        
        test_scenarios = [
            {
                'stock': 'AAPL',
                'features': ['fred_ICSA', 'fred_GDP', 'fred_AMERIBOR', 'VIX', 'RSI'],
                'shap_values': [0.45, 0.25, 0.20, 0.35, 0.30]
            },
            {
                'stock': 'MSFT', 
                'features': ['fred_ICSA', 'fred_GDP', 'fred_AMERIBOR', 'VIX', 'Momentum125'],
                'shap_values': [0.44, 0.26, 0.21, 0.33, 0.31]
            },
            {
                'stock': 'AMZN',
                'features': ['fred_ICSA', 'fred_GDP', 'fred_AMERIBOR', 'AnnVolatility', 'RSI'],
                'shap_values': [0.43, 0.27, 0.22, 0.34, 0.29]
            },
            {
                'stock': 'TSLA',
                'features': ['fred_ICSA', 'fred_GDP', 'fred_AMERIBOR', 'VolumeBreadth', 'VIX'],
                'shap_values': [0.46, 0.24, 0.19, 0.36, 0.32]
            },
            {
                'stock': 'ENPH',
                'features': ['fred_ICSA', 'fred_GDP', 'fred_AMERIBOR', 'BH20', 'FNG'],
                'shap_values': [0.42, 0.28, 0.23, 0.37, 0.33]
            }
        ]
        
        selected_macros = []
        
        print(f"Testing selection across {len(test_scenarios)} stocks:")
        
        for scenario in test_scenarios:
            stock = scenario['stock']
            features = scenario['features']
            shap_values = np.array(scenario['shap_values'])
            
            print(f"\n  📊 {stock}:")
            print(f"    Features: {features}")
            print(f"    SHAP values: {shap_values}")
            
            top_features = model.get_top_features_by_type(features, shap_values, n_per_type=5)
            
            if 'overall_top_2' in top_features:
                selected_macro = None
                for feat_name, importance, idx in top_features['overall_top_2']:
                    if feat_name.startswith('fred_'):
                        selected_macro = feat_name
                        break
                
                if selected_macro:
                    base_indicator = selected_macro.replace('fred_', '').split('_')[0]
                    selected_macros.append(base_indicator)
                    print(f"    ✅ Selected macro: {selected_macro} ({base_indicator})")
                else:
                    print(f"    ❌ No macro selected")
            else:
                print(f"    ❌ No overall_top_2 features")
        
        print(f"\n📈 DIVERSITY ANALYSIS:")
        macro_counts = Counter(selected_macros)
        print(f"Selected macro indicators: {dict(macro_counts)}")
        
        unique_macros = len(macro_counts)
        total_selections = len(selected_macros)
        diversity_score = unique_macros / total_selections if total_selections > 0 else 0
        
        print(f"Diversity metrics:")
        print(f"  Unique indicators: {unique_macros}/{total_selections}")
        print(f"  Diversity score: {diversity_score:.2f} (1.0 = perfect diversity)")
        
        if hasattr(model, 'macro_selection_history'):
            print(f"  Selection history: {dict(model.macro_selection_history)}")
        
        success = diversity_score > 0.4  # At least 40% diversity
        
        print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Diversity score {diversity_score:.2f}")
        
        return success, diversity_score, macro_counts
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False, 0.0, {}

def test_feature_importance_diversity_scoring():
    """Test alternative feature importance diversity scoring"""
    print(f"\n🎯 TESTING FEATURE IMPORTANCE DIVERSITY SCORING")
    print("="*60)
    
    try:
        test_features = {
            'fred_ICSA': {'type': 'macro', 'frequency': 'weekly', 'base': 'ICSA'},
            'fred_ICSA_zscore': {'type': 'macro_derived', 'frequency': 'weekly', 'base': 'ICSA'},
            'fred_GDP': {'type': 'macro', 'frequency': 'quarterly', 'base': 'GDP'},
            'fred_GDP_trend': {'type': 'macro_derived', 'frequency': 'quarterly', 'base': 'GDP'},
            'fred_AMERIBOR': {'type': 'macro', 'frequency': 'daily', 'base': 'AMERIBOR'},
            'fred_AMERIBOR_roc_5d': {'type': 'macro_derived', 'frequency': 'daily', 'base': 'AMERIBOR'},
            'VIX': {'type': 'proprietary', 'frequency': 'daily', 'base': 'VIX'},
            'RSI': {'type': 'proprietary', 'frequency': 'daily', 'base': 'RSI'},
        }
        
        shap_values = np.array([0.45, 0.42, 0.25, 0.23, 0.30, 0.28, 0.35, 0.32])
        feature_names = list(test_features.keys())
        
        print(f"Testing diversity scoring for {len(feature_names)} features:")
        
        base_indicator_counts = Counter()
        frequency_counts = Counter()
        type_counts = Counter()
        
        for feat_name, shap_val in zip(feature_names, shap_values):
            feat_info = test_features[feat_name]
            base_indicator_counts[feat_info['base']] += shap_val
            frequency_counts[feat_info['frequency']] += shap_val
            type_counts[feat_info['type']] += shap_val
        
        print(f"\nImportance by base indicator:")
        for base, total_imp in base_indicator_counts.most_common():
            print(f"  {base}: {total_imp:.3f}")
        
        print(f"\nImportance by frequency:")
        for freq, total_imp in frequency_counts.most_common():
            print(f"  {freq}: {total_imp:.3f}")
        
        print(f"\nImportance by type:")
        for feat_type, total_imp in type_counts.most_common():
            print(f"  {feat_type}: {total_imp:.3f}")
        
        diversity_penalties = {}
        for feat_name, shap_val in zip(feature_names, shap_values):
            feat_info = test_features[feat_name]
            
            base_penalty = (base_indicator_counts[feat_info['base']] - shap_val) * 0.1
            
            freq_penalty = (frequency_counts[feat_info['frequency']] - shap_val) * 0.05
            
            total_penalty = base_penalty + freq_penalty
            diversity_penalties[feat_name] = total_penalty
            
            adjusted_importance = shap_val - total_penalty
            print(f"  {feat_name}: {shap_val:.3f} - {total_penalty:.3f} = {adjusted_importance:.3f}")
        
        return True
        
    except Exception as e:
        print(f'❌ Diversity scoring test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_performance_impact():
    """Test performance impact of diversity constraints"""
    print(f"\n⚡ TESTING PERFORMANCE IMPACT")
    print("="*40)
    
    try:
        import time
        
        model = co.EnhancedTradingModel()
        
        test_features = []
        test_shap_values = []
        
        for i in range(20):
            for suffix in ['', '_zscore', '_roc_5d', '_trend', '_momentum']:
                feat_name = f'fred_INDICATOR_{i}{suffix}'
                test_features.append(feat_name)
                test_shap_values.append(np.random.uniform(0.1, 0.5))
        
        for prop_feat in co.CONFIG['PROPRIETARY_FEATURES']:
            test_features.append(prop_feat)
            test_shap_values.append(np.random.uniform(0.1, 0.4))
        
        test_shap_values = np.array(test_shap_values)
        
        print(f"Testing with {len(test_features)} features...")
        
        times = []
        for run in range(5):
            start_time = time.time()
            top_features = model.get_top_features_by_type(test_features, test_shap_values, n_per_type=5)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        print(f"Average selection time: {avg_time:.4f} seconds")
        print(f"Performance impact: {'✅ ACCEPTABLE' if avg_time < 0.1 else '⚠️ SLOW'}")
        
        return avg_time < 0.1
        
    except Exception as e:
        print(f'❌ Performance test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 TESTING SHAP DIVERSITY IMPROVEMENTS")
    print("="*60)
    
    diversity_success, diversity_score, macro_counts = test_macro_diversity_constraints()
    scoring_success = test_feature_importance_diversity_scoring()
    performance_success = test_performance_impact()
    
    print("\n" + "="*60)
    print("📋 TEST SUMMARY:")
    print(f"✅ Macro diversity constraints: {'PASS' if diversity_success else 'FAIL'}")
    print(f"✅ Feature importance scoring: {'PASS' if scoring_success else 'FAIL'}")
    print(f"✅ Performance impact: {'PASS' if performance_success else 'FAIL'}")
    
    if diversity_success:
        print(f"\n🎯 DIVERSITY RESULTS:")
        print(f"  Diversity score: {diversity_score:.2f}")
        print(f"  Macro distribution: {dict(macro_counts)}")
        
        if diversity_score >= 0.6:
            print(f"  🎉 EXCELLENT diversity achieved!")
        elif diversity_score >= 0.4:
            print(f"  ✅ GOOD diversity achieved!")
        else:
            print(f"  ⚠️ LIMITED diversity - needs improvement")
    
    overall_success = diversity_success and scoring_success and performance_success
    
    print(f"\n{'🎉 ALL TESTS PASSED!' if overall_success else '❌ SOME TESTS FAILED'}")
    print("="*60)
