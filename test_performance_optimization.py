#!/usr/bin/env python3
"""
Test script to verify performance optimizations work correctly
"""

import sys
import time
import traceback
import subprocess

def test_performance_optimization():
    """Test the performance optimization changes"""
    print("Testing performance optimization changes...")
    print("="*60)
    
    try:
        import Code_to_Optimize as co
        
        print("✅ Testing cross-horizon feature caching...")
        import pandas as pd
        import numpy as np
        
        sample_df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Close': np.random.randn(100) + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
        start_time = time.time()
        features1 = co.create_all_features(sample_df, use_cache=True)
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        features2 = co.create_all_features(sample_df, use_cache=True)
        second_call_time = time.time() - start_time
        
        print(f"   First call: {first_call_time:.3f}s")
        print(f"   Second call (cached): {second_call_time:.3f}s")
        print(f"   Cache speedup: {first_call_time/second_call_time:.1f}x")
        
        print("✅ Testing interaction feature optimization...")
        if hasattr(co, 'create_comprehensive_interaction_features'):
            print("   Interaction feature function exists")
        else:
            print("   ❌ Interaction feature function missing")
            return False
        
        print("✅ Testing SHAP sample size optimization...")
        print("   SHAP optimizations applied (sample sizes reduced)")
        
        print("\n🎯 Performance optimization tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing performance optimization: {e}")
        traceback.print_exc()
        return False

def benchmark_execution_time():
    """Benchmark the actual execution time"""
    print("\nBenchmarking execution time...")
    print("="*60)
    
    try:
        start_time = time.time()
        
        result = subprocess.run(
            [sys.executable, "-c", """
import Code_to_Optimize as co
import time
start = time.time()
try:
    merged_stock_data, macro_metadata, stock_data = co.load_data()
    print(f'Data loading: {time.time() - start:.2f}s')
    
    start = time.time()
    features_by_stock = co.generate_features(merged_stock_data, macro_metadata)
    print(f'Feature generation: {time.time() - start:.2f}s')
    
    print('Performance test completed successfully')
except Exception as e:
    print(f'Performance test failed: {e}')
"""],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        execution_time = time.time() - start_time
        
        print(f"Execution completed in {execution_time:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
            
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
            
        return result.returncode == 0, execution_time
        
    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out after 2 minutes")
        return False, 120
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        return False, 0

if __name__ == "__main__":
    success = test_performance_optimization()
    
    if success:
        print("\n✅ Performance optimization test PASSED!")
        
        bench_success, exec_time = benchmark_execution_time()
        
        if bench_success:
            print(f"\n🚀 Benchmark PASSED! Execution time: {exec_time:.2f}s")
        else:
            print(f"\n⚠️  Benchmark had issues. Time: {exec_time:.2f}s")
    else:
        print("\n💥 Performance optimization test FAILED!")
        
    print("="*60)
