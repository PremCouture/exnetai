#!/usr/bin/env python3
"""
Create minimal sample data to test the pipeline locally
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_stock_data_for_ticker(ticker):
    """Create sample stock CSV file for a specific ticker"""
    os.makedirs('data', exist_ok=True)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(300)]
    
    base_price = np.random.uniform(100, 200, 300)
    
    stock_data = {
        'Date': dates,
        'Open': base_price + np.random.uniform(-2, 2, 300),
        'High': base_price + np.random.uniform(0, 5, 300),
        'Low': base_price + np.random.uniform(-5, 0, 300),
        'Close': base_price,
        'Volume': np.random.uniform(1000000, 5000000, 300),
        'SMA20': np.random.uniform(95, 205, 300),
        'BL20': np.random.uniform(90, 195, 300),
        'BH20': np.random.uniform(105, 210, 300),
        'RSI': np.random.uniform(20, 80, 300),
        'AnnVolatility': np.random.uniform(15, 45, 300),
        'Momentum125': np.random.uniform(-30, 50, 300),
        'PriceStrength': np.random.uniform(-50, 100, 300),
        'VolumeBreadth': np.random.uniform(0.3, 2.0, 300),
        'VIX': np.random.uniform(10, 40, 300),
        'FNG': np.random.uniform(0, 100, 300),
    }
    
    df = pd.DataFrame(stock_data)
    df.to_csv(f'data/{ticker}.csv', index=False)
    print(f"Created data/{ticker}.csv")

def create_sample_stock_data():
    """Create sample stock CSV files for testing"""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM']
    for ticker in tickers:
        create_sample_stock_data_for_ticker(ticker)

def create_sample_fred_data():
    """Create sample FRED data for testing"""
    os.makedirs('fred_data', exist_ok=True)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i*30) for i in range(12)]  # Monthly data
    
    fred_indicators = {
        'GDP': ('GDP', np.random.uniform(20000, 25000, 12)),
        'UNEMPLOYMENT': ('UNRATE', np.random.uniform(3.5, 6.5, 12)),
        'INFLATION': ('CPIAUCSL', np.random.uniform(250, 300, 12)),
    }
    
    for folder_name, (indicator, values) in fred_indicators.items():
        folder_path = f'fred_data/{folder_name}'
        os.makedirs(folder_path, exist_ok=True)
        
        df = pd.DataFrame({
            'DATE': dates,
            indicator: values
        })
        df.to_csv(f'{folder_path}/{indicator}.csv', index=False)
        print(f"Created {folder_path}/{indicator}.csv")

if __name__ == "__main__":
    print("Creating sample data for local testing...")
    create_sample_stock_data()
    create_sample_fred_data()
    print("Sample data created successfully!")
