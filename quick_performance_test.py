#!/usr/bin/env python3
"""
Quick performance test for aggressive optimizations
"""

import Code_to_Optimize as co
import time

def test_performance():
    print("Testing aggressive optimization performance...")
    start = time.time()
    
    try:
        data_start = time.time()
        merged_stock_data, macro_metadata, stock_data = co.load_data()
        data_time = time.time() - data_start
        print(f'Data loading: {data_time:.2f}s')
        
        feature_start = time.time()
        features_by_stock = co.generate_features(merged_stock_data, macro_metadata)
        feature_time = time.time() - feature_start
        print(f'Feature generation: {feature_time:.2f}s')
        
        total_time = time.time() - start
        print(f'Total test time: {total_time:.2f}s')
        
        if total_time < 120:
            print(f'SUCCESS: Execution time {total_time:.2f}s is under 2-minute target!')
        else:
            print(f'WARNING: Execution time {total_time:.2f}s exceeds 2-minute target')
            
        print('Aggressive optimization test completed successfully')
        return True
        
    except Exception as e:
        print(f'Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_performance()
