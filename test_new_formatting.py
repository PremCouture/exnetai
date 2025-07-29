#!/usr/bin/env python3
"""Test script for new IF/THEN logic and TRADE PLAYBOOK formatting"""

import Code_to_Optimize as co
import pandas as pd

def test_new_formatting():
    """Test the new formatting functions"""
    print("🔍 TESTING NEW FORMATTING")
    print("="*60)
    
    sample = {
        "ticker": "ENPH",
        "signal": "STRONG BUY",
        "direction": "up",
        "accuracy": 82.4,
        "sharpe": 1.84,
        "drawdown": -57.3,
        "trigger": "Fear(0)",
        "shap_features": [
            {"value": "AMERIBOR=0.039", "shap": +0.059, "category": "macro"},
            {"value": "PriceStrength=0.587", "shap": -0.004, "category": "proprietary"},
        ],
        "guide": "High confidence mixed signal ✅✅"
    }
    
    print("Testing format_trade_playbook_table with sample data:")
    formatted_table = co.format_trade_playbook_table([sample], 30)
    print(formatted_table)
    
    print("\n" + "="*60)
    print("✅ New formatting test completed")

if __name__ == "__main__":
    test_new_formatting()
