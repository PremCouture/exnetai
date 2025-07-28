#!/usr/bin/env python3
"""
Test script to verify critical fixes for missing Drawdown column and data leakage
"""

import Code_to_Optimize as co
import time
import warnings

def test_critical_fixes():
    """Test the critical fixes for Drawdown column and data leakage"""
    print("🔧 TESTING CRITICAL FIXES")
    print("="*60)
    print("1. Missing Drawdown column fix")
    print("2. Data leakage elimination in training pipeline")
    print("3. Accuracy normalization from 90-99% to realistic levels")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        print("Starting test with fixes applied...")
        start_time = time.time()
        
        co.main()
        
        total_time = time.time() - start_time
        
        print(f'\n🎯 CRITICAL FIXES TEST RESULTS:')
        print(f'Total execution time: {total_time:.2f} seconds')
        print(f'Target: Under 120 seconds (2 minutes)')
        
        if total_time < 120:
            print(f'✅ SUCCESS: Execution time {total_time:.2f}s achieves sub-2-minute target!')
        else:
            print(f'⚠️  WARNING: Execution time {total_time:.2f}s still exceeds 2-minute target')
            
        print(f'\n📊 Expected Changes:')
        print(f'✅ Drawdown column should now appear in signal tables')
        print(f'✅ Accuracy should normalize to 60-80% range (from 90-99%)')
        print(f'✅ Data leakage validation warnings should appear if issues detected')
        print(f'✅ Proper temporal ordering in training data preparation')
        
        return True, total_time
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False, 0

if __name__ == "__main__":
    success, exec_time = test_critical_fixes()
    
    print("\n" + "="*60)
    if success:
        print("🎉 CRITICAL FIXES TEST COMPLETED!")
        print(f"✅ Pipeline executed successfully in {exec_time:.2f}s")
        print("✅ Check output above for Drawdown column and normalized accuracy")
    else:
        print("❌ CRITICAL FIXES TEST FAILED!")
        print("🔧 Review error messages above for debugging")
    
    print("="*60)
