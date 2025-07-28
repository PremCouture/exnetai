#!/usr/bin/env python3
"""
Debug script to investigate FRED data availability and SHAP diversity patterns
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict, Counter
import tempfile
import os

def debug_fred_data_availability():
    """Debug which FRED indicators are actually loaded and their data quality"""
    print("🔍 DEBUGGING FRED DATA AVAILABILITY AND QUALITY")
    print("="*80)
    
    warnings.filterwarnings('ignore')
    
    try:
        print(f"📊 FRED METADATA CONFIGURATION:")
        print(f"Total indicators in FRED_METADATA: {len(co.FRED_METADATA)}")
        
        for indicator, metadata in list(co.FRED_METADATA.items())[:10]:
            print(f"  {indicator}: {metadata.get('name', 'Unknown')} ({metadata.get('frequency', 'Unknown')} frequency)")
        
        print(f"\n🔧 TESTING FRED DATA LOADING PIPELINE:")
        
        original_config = co.CONFIG.copy()
        
        temp_dir = tempfile.mkdtemp()
        fred_dir = os.path.join(temp_dir, 'fred')
        os.makedirs(fred_dir, exist_ok=True)
        
        mock_fred_indicators = {
            'GDP': np.random.uniform(20000, 25000, 100),
            'AMERIBOR': np.random.uniform(4, 6, 100),
            'ICSA': np.random.uniform(200000, 400000, 100),
            'UMCSENT': np.random.uniform(70, 100, 100),
            'UNRATE': np.random.uniform(3, 5, 100),
            'CPIAUCSL': np.random.uniform(250, 300, 100),
            'DGS10': np.random.uniform(3, 5, 100),
            'DEXUSEU': np.random.uniform(0.8, 1.2, 100),
            'HOUST': np.random.uniform(1000, 1500, 100),
            'PAYEMS': np.random.uniform(140000, 160000, 100)
        }
        
        dates = pd.date_range('2023-01-01', periods=100)
        
        for indicator, values in mock_fred_indicators.items():
            indicator_dir = os.path.join(fred_dir, indicator)
            os.makedirs(indicator_dir, exist_ok=True)
            
            fred_data = pd.DataFrame({
                'DATE': dates,
                'VALUE': values
            })
            
            fred_csv_path = os.path.join(indicator_dir, f'{indicator}.csv')
            fred_data.to_csv(fred_csv_path, index=False)
        
        co.CONFIG['FRED_ROOT_PATH'] = fred_dir
        
        print(f"Created mock FRED data for {len(mock_fred_indicators)} indicators")
        
        print(f"\n1. Testing load_fred_data_from_folders...")
        fred_data_raw = co.load_fred_data_from_folders(use_cache=False)
        print(f"   Loaded {len(fred_data_raw)} raw FRED indicators")
        
        for indicator, df in fred_data_raw.items():
            print(f"     {indicator}: {len(df)} rows, columns: {list(df.columns)}")
        
        print(f"\n2. Testing fix_macro_data_alignment...")
        aligned_fred_data = co.fix_macro_data_alignment(fred_data_raw)
        print(f"   Aligned {len(aligned_fred_data)} FRED indicators")
        
        print(f"\n3. Testing data quality patterns...")
        quality_stats = {}
        
        for indicator, df in aligned_fred_data.items():
            if not df.empty and 'Value' in df.columns:
                values = df['Value']
                quality_stats[indicator] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'variance': values.var(),
                    'min': values.min(),
                    'max': values.max(),
                    'completeness': values.notna().sum() / len(values),
                    'coefficient_of_variation': values.std() / values.mean() if values.mean() != 0 else 0
                }
        
        print(f"\n📈 DATA QUALITY ANALYSIS:")
        print(f"{'Indicator':<12} {'Mean':<10} {'Std':<10} {'CV':<8} {'Complete':<10}")
        print("-" * 60)
        
        for indicator, stats in quality_stats.items():
            cv = stats['coefficient_of_variation']
            completeness = stats['completeness']
            print(f"{indicator:<12} {stats['mean']:<10.2f} {stats['std']:<10.2f} {cv:<8.3f} {completeness:<10.1%}")
        
        print(f"\n4. Testing macro feature creation...")
        
        mock_stock_data = pd.DataFrame({
            'Date': dates,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 100),
        })
        
        merged_stock_data, macro_metadata = co.merge_macro_with_stock({'TEST': mock_stock_data}, aligned_fred_data)
        
        if 'TEST' in merged_stock_data:
            test_df = merged_stock_data['TEST']
            macro_cols = [col for col in test_df.columns if col.startswith('fred_')]
            print(f"   Created {len(macro_cols)} macro columns in merged data")
            
            for col in macro_cols[:10]:
                print(f"     {col}")
            
            macro_features = co.create_macro_features(test_df, macro_metadata)
            print(f"   Generated {len(macro_features.columns)} macro features")
            
            feature_variance = {}
            for col in macro_features.columns:
                if col.startswith('fred_'):
                    variance = macro_features[col].var()
                    feature_variance[col] = variance
            
            print(f"\n📊 MACRO FEATURE VARIANCE ANALYSIS:")
            sorted_variance = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)
            
            print(f"{'Feature':<30} {'Variance':<15}")
            print("-" * 45)
            for feat, var in sorted_variance[:15]:
                print(f"{feat:<30} {var:<15.6f}")
        
        co.CONFIG.update(original_config)
        
        import shutil
        shutil.rmtree(temp_dir)
        
        return quality_stats, feature_variance if 'feature_variance' in locals() else {}
        
    except Exception as e:
        print(f'❌ Debug failed with error: {e}')
        import traceback
        traceback.print_exc()
        return {}, {}

def debug_shap_importance_patterns():
    """Debug SHAP importance calculation patterns and biases"""
    print(f"\n🎯 DEBUGGING SHAP IMPORTANCE PATTERNS")
    print("="*60)
    
    try:
        model = co.EnhancedTradingModel()
        
        scenarios = {
            'high_variance_bias': {
                'fred_ICSA': {'importance': 0.45, 'variance': 1000000},
                'fred_GDP': {'importance': 0.25, 'variance': 100},
                'fred_AMERIBOR': {'importance': 0.20, 'variance': 0.5},
                'fred_UMCSENT': {'importance': 0.15, 'variance': 50},
                'fred_UNRATE': {'importance': 0.12, 'variance': 0.8},
            },
            'frequency_bias': {
                'fred_ICSA_weekly': {'importance': 0.40, 'frequency': 'weekly'},
                'fred_GDP_quarterly': {'importance': 0.30, 'frequency': 'quarterly'},
                'fred_AMERIBOR_daily': {'importance': 0.35, 'frequency': 'daily'},
                'fred_UMCSENT_monthly': {'importance': 0.25, 'frequency': 'monthly'},
            },
            'feature_engineering_amplification': {
                'fred_ICSA': 0.20,
                'fred_ICSA_zscore': 0.25,
                'fred_ICSA_roc_5d': 0.30,
                'fred_ICSA_trend': 0.35,
                'fred_ICSA_momentum': 0.28,
                'fred_GDP': 0.15,
                'fred_GDP_zscore': 0.18,
                'fred_AMERIBOR': 0.22,
                'fred_AMERIBOR_roc_20d': 0.24,
            }
        }
        
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n  📊 SCENARIO: {scenario_name.upper()}")
            
            if scenario_name == 'feature_engineering_amplification':
                feature_names = list(scenario_data.keys())
                shap_values = np.array(list(scenario_data.values()))
                
                print(f"    Testing feature engineering amplification effect:")
                
                base_indicators = defaultdict(list)
                for feat in feature_names:
                    if feat.startswith('fred_'):
                        base = feat.split('_')[1]
                        base_indicators[base].append((feat, scenario_data[feat]))
                
                for base, features in base_indicators.items():
                    total_importance = sum(imp for _, imp in features)
                    print(f"      {base}: {len(features)} features, total importance: {total_importance:.3f}")
                    for feat, imp in features:
                        print(f"        {feat}: {imp:.3f}")
                
                top_features = model.get_top_features_by_type(feature_names, shap_values, n_per_type=5)
                
                if 'overall_top_2' in top_features:
                    print(f"    Selected features:")
                    for feat_name, importance, idx in top_features['overall_top_2']:
                        categories = model.categorize_features([feat_name])
                        category = next(iter([k for k, v in categories.items() if v]), 'unknown')
                        print(f"      [{category.upper()}] {feat_name}: {importance:.3f}")
            
            else:
                feature_names = list(scenario_data.keys())
                if scenario_name == 'high_variance_bias':
                    shap_values = np.array([data['importance'] for data in scenario_data.values()])
                    print(f"    High variance features may dominate SHAP selection:")
                    for feat, data in scenario_data.items():
                        print(f"      {feat}: importance={data['importance']:.3f}, variance={data['variance']}")
                
                elif scenario_name == 'frequency_bias':
                    shap_values = np.array([data['importance'] for data in scenario_data.values()])
                    print(f"    Higher frequency data may have more predictive signal:")
                    for feat, data in scenario_data.items():
                        print(f"      {feat}: importance={data['importance']:.3f}, frequency={data['frequency']}")
        
        return True
        
    except Exception as e:
        print(f'❌ SHAP pattern debug failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

def debug_current_selection_logic():
    """Debug the current get_top_features_by_type selection logic"""
    print(f"\n🔧 DEBUGGING CURRENT SELECTION LOGIC")
    print("="*50)
    
    try:
        model = co.EnhancedTradingModel()
        
        test_features = [
            'fred_ICSA', 'fred_ICSA_zscore', 'fred_ICSA_roc_5d',
            'fred_GDP', 'fred_GDP_zscore', 
            'fred_AMERIBOR', 'fred_AMERIBOR_trend',
            'VIX', 'RSI', 'AnnVolatility',
            'ATR', 'MACD', 'StochRSI'
        ]
        
        test_shap_values = np.array([
            0.45, 0.42, 0.40,  # ICSA features (high)
            0.25, 0.23,        # GDP features (medium)
            0.30, 0.28,        # AMERIBOR features (medium-high)
            0.35, 0.32, 0.29,  # Proprietary features
            0.20, 0.18, 0.15   # Technical features
        ])
        
        print(f"Test features and SHAP values:")
        for feat, shap_val in zip(test_features, test_shap_values):
            print(f"  {feat}: {shap_val:.3f}")
        
        categories = model.categorize_features(test_features)
        print(f"\nFeature categorization:")
        for category, features in categories.items():
            if features:
                print(f"  {category}: {features}")
        
        top_features = model.get_top_features_by_type(test_features, test_shap_values, n_per_type=5)
        
        print(f"\nCurrent selection results:")
        for category, features in top_features.items():
            if features and category != 'overall_top_2':
                print(f"  {category}: {[f[0] for f in features]}")
        
        if 'overall_top_2' in top_features:
            print(f"\nOverall top 2 selection:")
            for i, (feat_name, importance, idx) in enumerate(top_features['overall_top_2'], 1):
                feat_categories = model.categorize_features([feat_name])
                category = next(iter([k for k, v in feat_categories.items() if v]), 'unknown')
                print(f"    {i}. [{category.upper()}] {feat_name}: {importance:.3f}")
        
        print(f"\n🔍 ANALYSIS:")
        macro_features = [f for f in test_features if f.startswith('fred_')]
        macro_importances = [test_shap_values[i] for i, f in enumerate(test_features) if f.startswith('fred_')]
        
        print(f"Macro feature importance ranking:")
        macro_ranking = sorted(zip(macro_features, macro_importances), key=lambda x: x[1], reverse=True)
        for feat, imp in macro_ranking:
            base_indicator = feat.replace('fred_', '').split('_')[0]
            print(f"  {feat} ({base_indicator}): {imp:.3f}")
        
        return True
        
    except Exception as e:
        print(f'❌ Selection logic debug failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE FRED & SHAP DIVERSITY DEBUG")
    print("="*80)
    
    quality_stats, feature_variance = debug_fred_data_availability()
    shap_success = debug_shap_importance_patterns()
    logic_success = debug_current_selection_logic()
    
    print("\n" + "="*80)
    print("📋 DEBUG SUMMARY:")
    print(f"✅ FRED data availability: {'PASS' if quality_stats else 'FAIL'}")
    print(f"✅ SHAP pattern analysis: {'PASS' if shap_success else 'FAIL'}")
    print(f"✅ Selection logic analysis: {'PASS' if logic_success else 'FAIL'}")
    
    if quality_stats:
        print(f"\n🔍 KEY FINDINGS:")
        print(f"- {len(quality_stats)} FRED indicators successfully loaded")
        
        if feature_variance:
            high_variance_features = [k for k, v in feature_variance.items() if v > np.median(list(feature_variance.values()))]
            print(f"- High variance features that may dominate: {len(high_variance_features)}")
            print(f"  Examples: {high_variance_features[:3]}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"1. Implement diversity constraints to prevent same indicator dominance")
        print(f"2. Consider variance normalization in SHAP importance calculation")
        print(f"3. Add base indicator tracking to ensure variety across stocks")
        print(f"4. Test alternative feature selection methods beyond pure SHAP ranking")
    
    print("="*80)
