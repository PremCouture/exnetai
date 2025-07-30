#!/usr/bin/env python3
"""
Basic test to verify ensemble classes can be imported and instantiated
"""

import sys
sys.path.append('/home/ubuntu/exnetai')

def test_basic_ensemble():
    """Test that ensemble classes can be imported and instantiated"""
    print("🔍 TESTING BASIC ENSEMBLE FUNCTIONALITY")
    print("="*50)
    
    try:
        import Code_to_Optimize as co
        print("✅ Code_to_Optimize imported successfully")
        
        ensemble = co.EnsembleTradingModel()
        print("✅ EnsembleTradingModel instantiated successfully")
        
        macro_model = co.UniversalMacroModel()
        print("✅ UniversalMacroModel instantiated successfully")
        
        tech_model = co.StockTechnicalModel()
        print("✅ StockTechnicalModel instantiated successfully")
        
        assert hasattr(ensemble, 'train_ensemble'), "EnsembleTradingModel missing train_ensemble method"
        assert hasattr(ensemble, 'predict_ensemble_proba'), "EnsembleTradingModel missing predict_ensemble_proba method"
        assert hasattr(macro_model, 'train_macro_model'), "UniversalMacroModel missing train_macro_model method"
        assert hasattr(tech_model, 'train_technical_model'), "StockTechnicalModel missing train_technical_model method"
        
        print("✅ All required methods exist")
        
        assert hasattr(co, 'generate_signals_with_ensemble'), "Missing generate_signals_with_ensemble function"
        print("✅ generate_signals_with_ensemble function exists")
        
        print("\n🎉 ALL BASIC TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Error in basic ensemble test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_ensemble()
