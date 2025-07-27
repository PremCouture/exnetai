#!/usr/bin/env python3
"""
Test script to verify ultra-aggressive performance optimizations achieve sub-2-minute target
"""

import Code_to_Optimize as co
import time
import warnings

def test_ultra_aggressive_optimization():
    """Test the ultra-aggressive performance optimization changes"""
    print("Testing ultra-aggressive performance optimizations...")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        print("Starting ultra-aggressive optimization test with synthetic data...")
        start = time.time()
        
        co.main()
        
        total_time = time.time() - start
        print(f'\nULTRA-AGGRESSIVE OPTIMIZATION RESULTS:')
        print(f'Total execution time: {total_time:.2f}s')
        
        if total_time < 120:
            print(f'🚀 SUCCESS: Execution time {total_time:.2f}s is UNDER 2-minute target!')
            return True
        else:
            print(f'⚠️  WARNING: Execution time {total_time:.2f}s exceeds 2-minute target')
            return False
            
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ultra_aggressive_optimization()
    
    if success:
        print("\n✅ Ultra-aggressive optimization test PASSED!")
    else:
        print("\n💥 Ultra-aggressive optimization test FAILED!")
        
    print("="*60)
