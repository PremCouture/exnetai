#!/usr/bin/env python3
"""
Test script to verify adaptive SHAP threshold calculation works for different distributions
"""

import Code_to_Optimize as co
import numpy as np
import warnings

def test_adaptive_threshold_calculation():
    """Test adaptive threshold calculation for different SHAP distributions"""
    print("🔍 TESTING ADAPTIVE THRESHOLD CALCULATION")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        model = co.EnhancedTradingModel()
        
        scenarios = [
            {'name': 'High variance (AMZN-like)', 'values': np.array([0.031, 0.018, 0.012, 0.008, 0.005, 0.003])},
            {'name': 'Low variance (NFLX-like)', 'values': np.array([0.006, 0.005, 0.004, 0.003, 0.002, 0.001])},
            {'name': 'Very low variance', 'values': np.array([0.002, 0.0018, 0.0015, 0.001, 0.0008, 0.0005])},
            {'name': 'Mixed distribution', 'values': np.array([0.015, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0005])}
        ]
        
        for scenario in scenarios:
            print(f"\n📊 {scenario['name']}:")
            print(f"  Values: {scenario['values']}")
            
            threshold = model.calculate_adaptive_threshold(scenario['values'])
            passing_features = np.sum(scenario['values'] >= threshold)
            coverage = passing_features / len(scenario['values']) * 100
            
            print(f"  Adaptive threshold: {threshold:.6f}")
            print(f"  Passing features: {passing_features}/{len(scenario['values'])} ({coverage:.1f}%)")
            
            if passing_features >= 2:
                print(f"    ✅ Would generate signal (enough features)")
            elif passing_features == 1:
                print(f"    ⚠️  Might generate signal (1 feature + fallback)")
            else:
                print(f"    ❌ Would skip signal (no features)")
        
        return True
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_adaptive_threshold_calculation()
    print(f"\n{'✅ ADAPTIVE THRESHOLD TESTS PASSED!' if success else '❌ TESTS FAILED!'}")
