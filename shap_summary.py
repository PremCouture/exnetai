import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    """SHAP analysis and visualization module"""
    
    def __init__(self):
        self.explainers = {}
        self.feature_categories = {}
    
    def create_shap_summary_plot(self, shap_values: np.ndarray, features: pd.DataFrame, 
                                feature_names: List[str], title: str = "SHAP Summary Plot",
                                max_display: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """Create comprehensive SHAP summary plot with feature categorization"""
        
        plt.figure(figsize=figsize)
        
        shap.summary_plot(shap_values, features, feature_names=feature_names, 
                         max_display=max_display, show=False)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    def analyze_feature_importance_by_category(self, shap_values: np.ndarray, 
                                             feature_names: List[str]) -> Dict[str, Dict]:
        """Analyze SHAP feature importance by category"""
        
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        categories = self._categorize_features(feature_names)
        
        category_analysis = {}
        
        for category, features in categories.items():
            if not features:
                continue
            
            feature_indices = [i for i, name in enumerate(feature_names) if name in features]
            
            if feature_indices:
                category_importance = sum(shap_importance[i] for i in feature_indices)
                
                top_features = []
                for i in feature_indices:
                    top_features.append({
                        'feature': feature_names[i],
                        'importance': float(shap_importance[i]),
                        'mean_shap': float(shap_values[:, i].mean()),
                        'std_shap': float(shap_values[:, i].std())
                    })
                
                top_features.sort(key=lambda x: x['importance'], reverse=True)
                
                category_analysis[category] = {
                    'total_importance': float(category_importance),
                    'feature_count': len(features),
                    'avg_importance': float(category_importance / len(features)),
                    'top_features': top_features[:10]
                }
        
        return category_analysis
    
    def create_feature_category_plot(self, category_analysis: Dict[str, Dict], 
                                   figsize: Tuple[int, int] = (12, 6)):
        """Create visualization of feature importance by category"""
        
        categories = list(category_analysis.keys())
        importances = [category_analysis[cat]['total_importance'] for cat in categories]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        ax1.bar(categories, importances, color=colors)
        ax1.set_title('Total SHAP Importance by Feature Category', fontweight='bold')
        ax1.set_ylabel('Total SHAP Importance')
        ax1.tick_params(axis='x', rotation=45)
        
        avg_importances = [category_analysis[cat]['avg_importance'] for cat in categories]
        ax2.bar(categories, avg_importances, color=colors)
        ax2.set_title('Average SHAP Importance by Feature Category', fontweight='bold')
        ax2.set_ylabel('Average SHAP Importance')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def explain_shap_insights(self, category_analysis: Dict[str, Dict]) -> str:
        """Generate textual explanation of SHAP analysis insights"""
        
        insights = []
        insights.append("SHAP ANALYSIS INSIGHTS")
        insights.append("=" * 50)
        
        sorted_categories = sorted(category_analysis.items(), 
                                 key=lambda x: x[1]['total_importance'], reverse=True)
        
        insights.append(f"\n1. FEATURE CATEGORY RANKING (by total importance):")
        for i, (category, data) in enumerate(sorted_categories, 1):
            insights.append(f"   {i}. {category.upper()}: {data['total_importance']:.4f}")
            insights.append(f"      - {data['feature_count']} features")
            insights.append(f"      - Avg importance per feature: {data['avg_importance']:.4f}")
        
        insights.append(f"\n2. TOP FEATURES BY CATEGORY:")
        for category, data in sorted_categories:
            insights.append(f"\n   {category.upper()}:")
            for i, feat in enumerate(data['top_features'][:3], 1):
                insights.append(f"      {i}. {feat['feature']}: {feat['importance']:.4f}")
        
        insights.append(f"\n3. BUSINESS INSIGHTS:")
        
        if 'proprietary' in category_analysis:
            prop_importance = category_analysis['proprietary']['total_importance']
            total_importance = sum(data['total_importance'] for data in category_analysis.values())
            prop_percentage = (prop_importance / total_importance) * 100
            insights.append(f"   - Proprietary features account for {prop_percentage:.1f}% of model decisions")
        
        if 'macro' in category_analysis and 'technical' in category_analysis:
            macro_imp = category_analysis['macro']['total_importance']
            tech_imp = category_analysis['technical']['total_importance']
            if macro_imp > tech_imp:
                insights.append("   - Macro-economic factors are more influential than technical indicators")
            else:
                insights.append("   - Technical indicators dominate over macro-economic factors")
        
        if 'interaction' in category_analysis:
            int_importance = category_analysis['interaction']['total_importance']
            total_importance = sum(data['total_importance'] for data in category_analysis.values())
            int_percentage = (int_importance / total_importance) * 100
            if int_percentage > 15:
                insights.append(f"   - Feature interactions are significant ({int_percentage:.1f}% of decisions)")
            else:
                insights.append("   - Feature interactions have limited impact on predictions")
        
        insights.append(f"\n4. MODEL INTERPRETABILITY:")
        feature_diversity = len([cat for cat, data in category_analysis.items() 
                               if data['total_importance'] > 0.01])
        if feature_diversity >= 4:
            insights.append("   - High feature diversity indicates robust, well-balanced model")
        elif feature_diversity >= 2:
            insights.append("   - Moderate feature diversity - model relies on multiple signal types")
        else:
            insights.append("   - Low feature diversity - model may be overly dependent on single signal type")
        
        return "\n".join(insights)
    
    def create_shap_waterfall_plot(self, shap_values: np.ndarray, features: pd.DataFrame,
                                  feature_names: List[str], sample_idx: int = 0,
                                  max_display: int = 15, figsize: Tuple[int, int] = (12, 8)):
        """Create SHAP waterfall plot for individual prediction explanation"""
        
        if len(shap_values.shape) > 1:
            sample_shap = shap_values[sample_idx]
            sample_features = features.iloc[sample_idx]
        else:
            sample_shap = shap_values
            sample_features = features.iloc[0] if isinstance(features, pd.DataFrame) else features
        
        plt.figure(figsize=figsize)
        
        feature_importance = list(zip(feature_names, sample_shap, sample_features))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = feature_importance[:max_display]
        
        feature_names_top = [f[0] for f in top_features]
        shap_values_top = [f[1] for f in top_features]
        feature_values_top = [f[2] for f in top_features]
        
        colors = ['red' if val < 0 else 'blue' for val in shap_values_top]
        
        y_pos = np.arange(len(feature_names_top))
        
        plt.barh(y_pos, shap_values_top, color=colors, alpha=0.7)
        plt.yticks(y_pos, [f"{name} = {val:.3f}" for name, val in 
                          zip(feature_names_top, feature_values_top)])
        plt.xlabel('SHAP Value (impact on prediction)')
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type for analysis"""
        categories = {
            'macro': [],
            'proprietary': [],
            'technical': [],
            'interaction': [],
            'regime': [],
            'transformed': []
        }
        
        proprietary_features = ['VIX', 'FNG', 'RSI', 'AnnVolatility', 'Momentum125', 
                               'PriceStrength', 'VolumeBreadth', 'CallPut', 'NewsScore', 
                               'MACD', 'BollingerBandWidth']
        
        macro_indicators = ['GDP', 'UNRATE', 'CPI', 'PAYEMS', 'FEDFUNDS', 'UMCSENT', 
                           'ICSA', 'VIXCLS', 'DGS10', 'DGS2', 'T10Y2Y', 'INDPRO']
        
        for feature in feature_names:
            if any(prop in feature for prop in proprietary_features):
                categories['proprietary'].append(feature)
            elif 'interaction' in feature.lower() or '_x_' in feature:
                categories['interaction'].append(feature)
            elif 'regime' in feature.lower():
                categories['regime'].append(feature)
            elif 'transformed' in feature.lower() or '_sq' in feature or '_log' in feature:
                categories['transformed'].append(feature)
            elif any(macro in feature for macro in macro_indicators):
                categories['macro'].append(feature)
            else:
                categories['technical'].append(feature)
        
        return categories

def format_shap_feature_display(feat_name: str, feat_value: float, shap_val: float, 
                               category: str) -> str:
    """Format feature display for SHAP explanations"""
    
    display_name = feat_name[:20] if len(feat_name) > 20 else feat_name
    
    if isinstance(feat_value, (int, float)) and not np.isnan(feat_value):
        if abs(feat_value) >= 1000:
            value_str = f"{feat_value:.0f}"
        elif abs(feat_value) >= 10:
            value_str = f"{feat_value:.1f}"
        elif abs(feat_value) >= 1:
            value_str = f"{feat_value:.2f}"
        else:
            value_str = f"{feat_value:.3f}"
    else:
        value_str = str(feat_value)[:10]
    
    direction = "↑" if shap_val > 0 else "↓"
    
    cat_abbrev = {
        'macro': 'M',
        'proprietary': 'P',
        'technical': 'T',
        'interaction': 'I',
        'regime': 'R',
        'transformed': 'X'
    }.get(category, 'U')
    
    return f"[{cat_abbrev}] {display_name}={value_str} ({shap_val:+.3f}{direction})"

def label_fng(val: int) -> str:
    """Get textual label for Fear & Greed Index"""
    if val >= 85:
        return "ExGreed"
    elif val >= 75:
        return "Greed"
    elif val >= 60:
        return "Greed+"
    elif val >= 40:
        return "Neutral"
    elif val >= 25:
        return "Fear"
    elif val >= 15:
        return "Fear+"
    else:
        return "ExFear"
