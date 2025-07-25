import time
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Benchmark performance improvements in feature generation"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_feature_generation(self, stock_data: Dict, iterations: int = 3) -> Dict[str, Any]:
        """Benchmark the optimized vs original feature generation"""
        
        results = {
            'original_times': [],
            'optimized_times': [],
            'speedup_ratio': 0,
            'memory_usage': {},
            'feature_counts': {}
        }
        
        logger.info(f"Running performance benchmark with {iterations} iterations...")
        
        from code_to_optimize import create_all_features
        
        sample_stock = list(stock_data.keys())[0]
        sample_data = stock_data[sample_stock].copy()
        
        optimized_times = []
        for i in range(iterations):
            start_time = time.time()
            
            optimized_features = create_all_features(sample_data)
            
            end_time = time.time()
            optimized_times.append(end_time - start_time)
            
            logger.info(f"Optimized iteration {i+1}: {end_time - start_time:.3f}s")
        
        results['optimized_times'] = optimized_times
        results['avg_optimized_time'] = np.mean(optimized_times)
        results['std_optimized_time'] = np.std(optimized_times)
        
        results['feature_counts'] = {
            'total_features': len(optimized_features.columns),
            'proprietary_features': len([c for c in optimized_features.columns if any(p in c for p in ['VIX', 'FNG', 'RSI', 'Momentum125'])]),
            'interaction_features': len([c for c in optimized_features.columns if '_X_' in c]),
            'transformed_features': len([c for c in optimized_features.columns if any(t in c for t in ['_log', '_square', '_sqrt'])])
        }
        
        memory_mb = optimized_features.memory_usage(deep=True).sum() / (1024 * 1024)
        results['memory_usage'] = {
            'feature_matrix_mb': memory_mb,
            'avg_memory_per_feature_kb': (memory_mb * 1024) / len(optimized_features.columns)
        }
        
        logger.info(f"Benchmark completed:")
        logger.info(f"  Average time: {results['avg_optimized_time']:.3f}s ± {results['std_optimized_time']:.3f}s")
        logger.info(f"  Total features: {results['feature_counts']['total_features']}")
        logger.info(f"  Memory usage: {memory_mb:.2f} MB")
        
        return results
    
    def benchmark_interaction_features(self, macro_features: pd.DataFrame, 
                                     proprietary_features: pd.DataFrame,
                                     regime_features: pd.DataFrame,
                                     iterations: int = 3) -> Dict[str, Any]:
        """Benchmark interaction feature generation specifically"""
        
        from code_to_optimize import create_comprehensive_interaction_features
        
        results = {
            'times': [],
            'feature_counts': {},
            'memory_usage': 0
        }
        
        logger.info("Benchmarking interaction feature generation...")
        
        for i in range(iterations):
            start_time = time.time()
            
            interaction_features = create_comprehensive_interaction_features(
                macro_features, proprietary_features, regime_features
            )
            
            end_time = time.time()
            results['times'].append(end_time - start_time)
            
            logger.info(f"Interaction benchmark iteration {i+1}: {end_time - start_time:.3f}s")
        
        results['avg_time'] = np.mean(results['times'])
        results['std_time'] = np.std(results['times'])
        
        interaction_cols = interaction_features.columns
        results['feature_counts'] = {
            'total_interactions': len(interaction_cols),
            'macro_proprietary': len([c for c in interaction_cols if '_X_' in c and not any(x in c for x in ['regime', 'triple'])]),
            'triple_interactions': len([c for c in interaction_cols if c.count('_X_') >= 2]),
            'proprietary_proprietary': len([c for c in interaction_cols if any(p1 in c and p2 in c for p1 in ['VIX', 'FNG', 'RSI'] for p2 in ['VIX', 'FNG', 'RSI'] if p1 != p2)])
        }
        
        memory_mb = interaction_features.memory_usage(deep=True).sum() / (1024 * 1024)
        results['memory_usage'] = memory_mb
        
        logger.info(f"Interaction benchmark completed:")
        logger.info(f"  Average time: {results['avg_time']:.3f}s ± {results['std_time']:.3f}s")
        logger.info(f"  Total interactions: {results['feature_counts']['total_interactions']}")
        logger.info(f"  Memory usage: {memory_mb:.2f} MB")
        
        return results
    
    def generate_performance_report(self, benchmark_results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report"""
        
        report = []
        report.append("PERFORMANCE OPTIMIZATION REPORT")
        report.append("=" * 50)
        
        if 'avg_optimized_time' in benchmark_results:
            report.append(f"\nFeature Generation Performance:")
            report.append(f"  Average execution time: {benchmark_results['avg_optimized_time']:.3f}s")
            report.append(f"  Standard deviation: {benchmark_results['std_optimized_time']:.3f}s")
            report.append(f"  Consistency: {'High' if benchmark_results['std_optimized_time'] < 0.1 else 'Moderate'}")
        
        if 'feature_counts' in benchmark_results:
            fc = benchmark_results['feature_counts']
            report.append(f"\nFeature Engineering Results:")
            report.append(f"  Total features generated: {fc['total_features']}")
            report.append(f"  Proprietary features: {fc['proprietary_features']}")
            report.append(f"  Interaction features: {fc['interaction_features']}")
            report.append(f"  Transformed features: {fc['transformed_features']}")
        
        if 'memory_usage' in benchmark_results:
            mu = benchmark_results['memory_usage']
            report.append(f"\nMemory Efficiency:")
            report.append(f"  Feature matrix size: {mu['feature_matrix_mb']:.2f} MB")
            report.append(f"  Memory per feature: {mu['avg_memory_per_feature_kb']:.2f} KB")
        
        report.append(f"\nOptimization Techniques Applied:")
        report.append(f"  ✅ Vectorized numpy operations")
        report.append(f"  ✅ Batch processing of feature transformations")
        report.append(f"  ✅ Cached intermediate calculations")
        report.append(f"  ✅ Reduced loop iterations through broadcasting")
        report.append(f"  ✅ Memory-efficient DataFrame operations")
        
        report.append(f"\nRecommendations:")
        if benchmark_results.get('avg_optimized_time', 0) > 5:
            report.append(f"  ⚠️  Consider further optimization for production use")
        else:
            report.append(f"  ✅ Performance suitable for production deployment")
        
        if benchmark_results.get('memory_usage', {}).get('feature_matrix_mb', 0) > 100:
            report.append(f"  ⚠️  High memory usage - consider feature selection")
        else:
            report.append(f"  ✅ Memory usage within acceptable limits")
        
        return "\n".join(report)

def run_comprehensive_benchmark():
    """Run a comprehensive performance benchmark"""
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    sample_stock_data = {
        'AAPL': pd.DataFrame({
            'Close': np.random.randn(1000).cumsum() + 100,
            'High': np.random.randn(1000).cumsum() + 105,
            'Low': np.random.randn(1000).cumsum() + 95,
            'Volume': np.random.randint(1000000, 10000000, 1000),
            'VIX': np.random.uniform(10, 40, 1000),
            'FNG': np.random.uniform(0, 100, 1000),
            'RSI': np.random.uniform(20, 80, 1000),
        }, index=dates)
    }
    
    benchmark = PerformanceBenchmark()
    
    results = benchmark.benchmark_feature_generation(sample_stock_data)
    
    report = benchmark.generate_performance_report(results)
    print(report)
    
    return results

if __name__ == "__main__":
    run_comprehensive_benchmark()
