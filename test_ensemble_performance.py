#!/usr/bin/env python3
"""
Test ensemble model performance vs single model approach
"""

import sys
import time
import os
sys.path.append('/home/ubuntu/exnetai')

def benchmark_ensemble_vs_single():
    """Compare performance of ensemble vs single model approach"""
    print("🔍 BENCHMARKING ENSEMBLE VS SINGLE MODEL")
    print("="*60)
    
    try:
        import Code_to_Optimize as co
        
        print("Loading data...")
        merged_stock_data, macro_metadata, stock_data = co.load_data()
        
        if not merged_stock_data:
            print("❌ No data loaded")
            return
            
        test_stocks = dict(list(merged_stock_data.items())[:5])
        
        print("\n📊 Testing Single Model Approach...")
        single_start = time.time()
        
        single_times = []
        for ticker, df in test_stocks.items():
            stock_start = time.time()
            features = co.create_all_features(df, macro_metadata, use_cache=False)  # No cache to measure true time
            stock_time = time.time() - stock_start
            single_times.append(stock_time)
            print(f"  {ticker}: {stock_time:.2f}s")
            
        single_total = time.time() - single_start
        
        print("\n🚀 Testing Ensemble Model Approach...")
        ensemble_start = time.time()
        
        ensemble_model = co.EnsembleTradingModel()
        macro_train_start = time.time()
        ensemble_model.macro_model.train_macro_model(test_stocks, macro_metadata, 30)
        macro_train_time = time.time() - macro_train_start
        
        ensemble_times = []
        for ticker, df in test_stocks.items():
            stock_start = time.time()
            if ticker not in ensemble_model.technical_models:
                ensemble_model.technical_models[ticker] = co.StockTechnicalModel()
            ensemble_model.technical_models[ticker].train_technical_model(ticker, df, macro_metadata, 30)
            stock_time = time.time() - stock_start
            ensemble_times.append(stock_time)
            print(f"  {ticker} (technical only): {stock_time:.2f}s")
            
        ensemble_total = time.time() - ensemble_start
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"Single Model Total Time: {single_total:.2f}s")
        print(f"Ensemble Model Total Time: {ensemble_total:.2f}s")
        print(f"Macro Model Training Time: {macro_train_time:.2f}s (one-time cost)")
        
        print(f"\n🔄 SIMULATING NEW STOCK ADDITION:")
        new_stock_single = sum(single_times) / len(single_times)  # Average time per stock
        new_stock_ensemble = sum(ensemble_times) / len(ensemble_times)  # Only technical training needed
        
        print(f"New Stock - Single Model: {new_stock_single:.2f}s")
        print(f"New Stock - Ensemble (macro cached): {new_stock_ensemble:.2f}s")
        
        savings_percent = ((new_stock_single - new_stock_ensemble) / new_stock_single) * 100
        print(f"Time Savings for New Stocks: {savings_percent:.1f}%")
        
        if savings_percent >= 30:
            print("✅ SUCCESS: Achieved target 30-40% time savings!")
        else:
            print(f"⚠️  WARNING: Only achieved {savings_percent:.1f}% savings (target: 30-40%)")
            
        return savings_percent >= 30
        
    except Exception as e:
        print(f"❌ Error in benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    benchmark_ensemble_vs_single()
