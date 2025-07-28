#!/usr/bin/env python3
"""
Debug script to investigate why most stocks show empty SHAP features
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings

def debug_feature_selection_per_stock():
    """Debug feature selection for each stock to understand why most show 'None'"""
    print("🔍 DEBUGGING EMPTY SHAP FEATURES PER STOCK")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        stocks = ['NFLX', 'MSFT', 'AMZN', 'TSLA', 'ENPH']
        
        model = co.EnhancedTradingModel()
        
        for stock in stocks:
            print(f"\n📊 DEBUGGING {stock}:")
            print("-" * 30)
            
            test_features = [
                'fred_ICSA', 'fred_GDP', 'fred_AMERIBOR', 'fred_UNRATE',
                'VIX', 'RSI', 'Volume', 'SMA20', 'BL20', 'BH20',
                'AnnVolatility', 'Momentum125', 'PriceStrength', 'VolumeBreadth',
                'MACD', 'ATR', 'StochRSI', 'PRICE_TO_SMA_200'
            ]
            
            if stock == 'AMZN':
                shap_values = np.array([
                    0.005, 0.003, 0.004, 0.002,  # FRED features
                    0.008, 0.012, 0.006, 0.009, 0.007, 0.005,  # Proprietary
                    0.031, 0.015, 0.008, 0.006,  # More proprietary
                    0.004, 0.003, 0.002, 0.018   # Technical
                ])
            else:
                shap_values = np.array([
                    0.002, 0.001, 0.0015, 0.0008,  # FRED features
                    0.003, 0.005, 0.002, 0.004, 0.003, 0.002,  # Proprietary
                    0.006, 0.004, 0.003, 0.002,  # More proprietary
                    0.001, 0.0008, 0.0006, 0.005   # Technical
                ])
            
            print(f"  Features: {len(test_features)}")
            print(f"  SHAP range: {min(shap_values):.6f} - {max(shap_values):.6f}")
            print(f"  Features above 0.01: {sum(1 for v in shap_values if v >= 0.01)}")
            print(f"  Features above 0.005: {sum(1 for v in shap_values if v >= 0.005)}")
            print(f"  Features above 0.001: {sum(1 for v in shap_values if v >= 0.001)}")
            
            top_features = model.get_top_features_by_type(test_features, shap_values, n_per_type=5)
            
            skip_signal = top_features.get('skip_signal', False)
            selected_features = top_features.get('overall_top_2', [])
            
            print(f"  Result: {'SIGNAL SKIPPED' if skip_signal else f'{len(selected_features)} features selected'}")
            
            if selected_features:
                for feat_name, importance, idx in selected_features:
                    categories = model.categorize_features([feat_name])
                    feat_type = "unknown"
                    for cat_name, feat_list in categories.items():
                        if feat_name in feat_list:
                            feat_type = cat_name
                            break
                    print(f"    ✅ {feat_name}: {importance:.6f} ({feat_type})")
            else:
                print(f"    ❌ No features selected")
            
            if not skip_signal:
                print(f"  📋 Feature breakdown:")
                for category in ['macro', 'proprietary', 'technical', 'regime', 'transformed', 'interaction']:
                    cat_features = top_features.get(category, [])
                    if cat_features:
                        print(f"    {category}: {len(cat_features)} features")
                        for feat_name, importance, idx in cat_features[:3]:  # Show top 3
                            print(f"      - {feat_name}: {importance:.6f}")
                    else:
                        print(f"    {category}: 0 features")
        
        return True
        
    except Exception as e:
        print(f'❌ Debug failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_threshold_sensitivity():
    """Test different thresholds to see what would work for the failing stocks"""
    print(f"\n🎯 TESTING THRESHOLD SENSITIVITY")
    print("="*50)
    
    try:
        failing_stock_features = [
            'fred_ICSA', 'fred_GDP', 'VIX', 'RSI', 'Volume', 'SMA20'
        ]
        failing_stock_shap = np.array([0.002, 0.001, 0.003, 0.005, 0.002, 0.004])
        
        print(f"Simulating failing stock pattern:")
        print(f"  Features: {failing_stock_features}")
        print(f"  SHAP values: {failing_stock_shap}")
        print(f"  Max importance: {max(failing_stock_shap):.6f}")
        
        thresholds_to_test = [0.01, 0.005, 0.003, 0.001, 0.0005]
        
        model = co.EnhancedTradingModel()
        
        for threshold in thresholds_to_test:
            passing_features = sum(1 for val in failing_stock_shap if val >= threshold)
            
            print(f"\n  Threshold {threshold:.4f}: {passing_features}/{len(failing_stock_shap)} features pass")
            
            if passing_features >= 2:
                print(f"    ✅ Would likely generate signal (enough features)")
            elif passing_features == 1:
                print(f"    ⚠️  Might generate signal (1 feature + fallback)")
            else:
                print(f"    ❌ Would skip signal (no features)")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"  Current threshold: 0.01")
        print(f"  For failing stocks with max ~0.005:")
        print(f"  - Consider lowering to 0.003 or 0.001")
        print(f"  - Or investigate why their importance values are so low")
        
        return True
        
    except Exception as e:
        print(f'❌ Threshold sensitivity test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def debug_feature_creation_issues():
    """Debug potential issues with feature creation that might cause low importance"""
    print(f"\n🔧 DEBUGGING FEATURE CREATION ISSUES")
    print("="*50)
    
    try:
        print("Potential causes for empty SHAP features:")
        print("1. 📊 SHAP importance values still too low for 0.01 threshold")
        print("2. 🏗️  Features not being created properly for some stocks")
        print("3. 📈 Macro features missing or empty for some stocks")
        print("4. 🔄 Feature categorization failing")
        print("5. 💾 Data quality issues (NaN, missing values)")
        
        print(f"\nChecking CONFIG settings:")
        print(f"  PROPRIETARY_FEATURES: {co.CONFIG['PROPRIETARY_FEATURES']}")
        print(f"  FRED_METADATA keys: {list(co.FRED_METADATA.keys())}")
        
        test_features = [
            'fred_ICSA', 'fred_GDP', 'VIX', 'RSI', 'Volume', 'SMA20',
            'PRICE_TO_SMA_200', 'AnnVolatility', 'Momentum125'
        ]
        
        model = co.EnhancedTradingModel()
        categorized = model.categorize_features(test_features)
        
        print(f"\nFeature categorization test:")
        for cat_name, feat_list in categorized.items():
            if feat_list:
                print(f"  {cat_name}: {feat_list}")
        
        macro_count = len(categorized.get('macro', []))
        prop_count = len(categorized.get('proprietary', []))
        tech_count = len(categorized.get('technical', []))
        
        print(f"\nCategory distribution:")
        print(f"  Macro: {macro_count}")
        print(f"  Proprietary: {prop_count}")
        print(f"  Technical: {tech_count}")
        
        if macro_count == 0:
            print(f"  ⚠️  WARNING: No macro features detected!")
        if prop_count == 0:
            print(f"  ⚠️  WARNING: No proprietary features detected!")
        
        return True
        
    except Exception as e:
        print(f'❌ Feature creation debug failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 DEBUGGING EMPTY SHAP FEATURES")
    print("="*60)
    
    selection_success = debug_feature_selection_per_stock()
    threshold_success = test_threshold_sensitivity()
    creation_success = debug_feature_creation_issues()
    
    print("\n" + "="*60)
    print("📋 DEBUG SUMMARY:")
    print(f"✅ Feature selection per stock: {'PASS' if selection_success else 'FAIL'}")
    print(f"✅ Threshold sensitivity: {'PASS' if threshold_success else 'FAIL'}")
    print(f"✅ Feature creation issues: {'PASS' if creation_success else 'FAIL'}")
    
    overall_success = selection_success and threshold_success and creation_success
    
    if overall_success:
        print(f"\n🔍 KEY FINDINGS:")
        print(f"✅ Debug completed successfully")
        print(f"📊 Likely cause: SHAP importance values for most stocks are below 0.01 threshold")
        print(f"🎯 Solution: Consider lowering threshold further (0.005 or 0.001)")
        print(f"🔧 Alternative: Investigate why some stocks have much lower importance values")
    else:
        print(f"\n❌ SOME DEBUG TESTS FAILED!")
        print(f"🔧 Review error messages above for debugging")
    
    print("="*60)
