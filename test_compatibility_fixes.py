#!/usr/bin/env python3
"""
Test script to verify backward compatibility fixes in Code_to_Optimize.py
"""

import sys
import traceback
import inspect

def test_import():
    """Test if Code_to_Optimize.py can be imported"""
    try:
        import Code_to_Optimize
        print("✅ Code_to_Optimize.py imports successfully")
        return True, Code_to_Optimize
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False, None

def test_function_signatures(module):
    """Test function signatures for backward compatibility"""
    try:
        sig = inspect.signature(module.load_stock_data)
        print(f"✅ load_stock_data signature: {sig}")
        
        sig = inspect.signature(module.create_all_features)
        print(f"✅ create_all_features signature: {sig}")
        
        sig = inspect.signature(module.load_fred_data_from_folders)
        print(f"✅ load_fred_data_from_folders signature: {sig}")
        
        return True
    except Exception as e:
        print(f"❌ Function signature test failed: {e}")
        traceback.print_exc()
        return False

def test_main_execution(module):
    """Test if main() function is accessible"""
    try:
        if hasattr(module, 'main'):
            print("✅ main() function is accessible")
            return True
        else:
            print("❌ main() function not found")
            return False
    except Exception as e:
        print(f"❌ Main execution test failed: {e}")
        traceback.print_exc()
        return False

def test_cache_variables(module):
    """Test if cache variables are defined"""
    try:
        if hasattr(module, 'DATA_CACHE'):
            print("✅ DATA_CACHE is defined")
        else:
            print("❌ DATA_CACHE not found")
            return False
            
        if hasattr(module, 'FEATURE_CACHE'):
            print("✅ FEATURE_CACHE is defined")
        else:
            print("❌ FEATURE_CACHE not found")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Cache variables test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Code_to_Optimize.py compatibility fixes...")
    print("="*60)
    
    import_ok, module = test_import()
    if import_ok:
        test_function_signatures(module)
        test_main_execution(module)
        test_cache_variables(module)
    
    print("="*60)
    print("Compatibility test complete")
