#!/usr/bin/env python3
"""
Test script to verify MACD KeyError fix in table creation
"""

import Code_to_Optimize as co
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch
import io
import sys

def test_macd_table_creation():
    """Test that MACD KeyError is fixed in create_complete_playbook_tables"""
    print("🔍 TESTING MACD TABLE CREATION FIX")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    try:
        mock_signals_df = pd.DataFrame({
            'Stock': ['AAPL', 'MSFT'],
            'Signal': ['BUY', 'HOLD'],
            'Accuracy': [85.5, 78.2],
            'Sharpe': [1.2, 0.9],
            'CAGR': [15.5, 12.3],
            'Drawdown': [-5.2, -8.1],
            'VIX': [25.5, 28.1],
            'FNG': [45, 52],
            'RSI': [65.2, 58.7],
            'AnnVolatility': [22.1, 18.9],
            'Momentum125': [8.5, -2.1],
            'PriceStrength': [0.75, 0.42],
            'VolumeBreadth': [0.65, 0.38],
            'MACD': [0.85, -0.23],  # Include MACD to test normal case
            'ATR': [2.15, 1.87],
            'ADX': [45.2, 38.9],
            'StochRSI': [0.72, 0.45],
            'CCI': [125.5, -89.2],
            'MFI': [68.5, 42.1],
            'NewsScore': [0.65, 0.32],
            'CallPut': [1.15, 0.89]
        })
        
        print(f"Mock signals dataframe columns: {list(mock_signals_df.columns)}")
        print(f"MACD values: {mock_signals_df['MACD'].tolist()}")
        
        print(f"\n📊 TEST 1: Normal case with MACD present")
        
        captured_output = io.StringIO()
        with patch('sys.stdout', captured_output):
            try:
                co.create_complete_playbook_tables(mock_signals_df, "30")
                test1_success = True
                test1_error = None
            except Exception as e:
                test1_success = False
                test1_error = str(e)
        
        output_text = captured_output.getvalue()
        
        if test1_success:
            print(f"   ✅ Table creation succeeded")
            print(f"   ✅ Output contains table headers: {'MACD' in output_text}")
        else:
            print(f"   ❌ Table creation failed: {test1_error}")
        
        print(f"\n📊 TEST 2: Edge case with MACD missing from dataframe")
        
        mock_signals_df_no_macd = mock_signals_df.drop(columns=['MACD'])
        print(f"   Removed MACD column, remaining: {list(mock_signals_df_no_macd.columns)}")
        
        captured_output2 = io.StringIO()
        with patch('sys.stdout', captured_output2):
            try:
                co.create_complete_playbook_tables(mock_signals_df_no_macd, "30")
                test2_success = True
                test2_error = None
            except Exception as e:
                test2_success = False
                test2_error = str(e)
        
        output_text2 = captured_output2.getvalue()
        
        if test2_success:
            print(f"   ✅ Table creation succeeded with missing MACD")
            print(f"   ✅ Error handling worked: {'N/A' in output_text2}")
        else:
            print(f"   ❌ Table creation failed: {test2_error}")
        
        print(f"\n📊 TEST 3: Verify CONFIG proprietary features")
        print(f"   CONFIG['PROPRIETARY_FEATURES']: {co.CONFIG['PROPRIETARY_FEATURES']}")
        print(f"   MACD in CONFIG: {'MACD' in co.CONFIG['PROPRIETARY_FEATURES']}")
        
        overall_success = test1_success and test2_success
        
        print(f"\n📊 SUMMARY:")
        print(f"   Test 1 (Normal case): {'✅ PASS' if test1_success else '❌ FAIL'}")
        print(f"   Test 2 (Missing MACD): {'✅ PASS' if test2_success else '❌ FAIL'}")
        print(f"   Overall: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
        
        return overall_success
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_macd_table_creation()
    
    print("\n" + "="*60)
    if success:
        print("🎉 MACD TABLE CREATION FIX TEST PASSED!")
        print("✅ KeyError: 'MACD' should now be resolved")
        print("✅ Table creation handles both present and missing MACD cases")
    else:
        print("❌ MACD TABLE CREATION FIX TEST FAILED!")
        print("🔧 Review error messages above for debugging")
    
    print("="*60)
