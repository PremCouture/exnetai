#!/usr/bin/env python3
"""
Create HTML version of trading analysis results with better formatting
"""

import os
import re
from datetime import datetime

def create_html_results():
    """Create HTML version of trading analysis results"""
    
    if os.path.exists('pipeline_output.txt'):
        with open('pipeline_output.txt', 'r') as f:
            content = f.read()
    elif os.path.exists('trading_analysis_results.txt'):
        with open('trading_analysis_results.txt', 'r') as f:
            content = f.read()
    else:
        content = "No trading analysis results found. Please run the pipeline first."
    
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Signal Analysis Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }
        h3 {
            color: #2980b9;
            margin-top: 25px;
        }
        .analysis-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .analysis-table th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .analysis-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        .analysis-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .analysis-table tr:hover {
            background-color: #e8f4f8;
        }
        .emoji {
            font-size: 1.2em;
        }
        .positive {
            color: #27ae60;
            font-weight: bold;
        }
        .negative {
            color: #e74c3c;
            font-weight: bold;
        }
        .neutral {
            color: #95a5a6;
        }
        .confidence-high {
            background-color: #d5f4e6;
            color: #27ae60;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }
        .confidence-medium {
            background-color: #fff3cd;
            color: #856404;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }
        .confidence-low {
            background-color: #f8d7da;
            color: #721c24;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }
        .code-block {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        .timestamp {
            color: #6c757d;
            font-size: 0.9em;
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }
        .summary-box {
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 COMPREHENSIVE TRADING SIGNAL ANALYSIS</h1>
        
        <div class="summary-box">
            <h3>📊 Analysis Overview</h3>
            <p><strong>Pipeline Status:</strong> Optimized Random Forest ML Pipeline with SHAP Explainability</p>
            <p><strong>Features:</strong> Stock Selection Expansion (3→10 stocks) + FRED Feature Name Simplification</p>
            <p><strong>Generated:</strong> {timestamp}</p>
        </div>
        
        <div class="code-block">{content}</div>
        
        <div class="warning-box">
            <h3>⚠️ Important Notes</h3>
            <ul>
                <li><strong>SHAP Emoji Indicators:</strong> 🟢 Positive impact, 🔴 Negative impact, ⚪ Neutral</li>
                <li><strong>FRED Features:</strong> Simplified names (e.g., "Unemployment" instead of "fred_UNEMPLOYMENT_m18_momentum")</li>
                <li><strong>Stock Coverage:</strong> Expanded analysis from 3 to 10 stocks</li>
                <li><strong>Confidence Levels:</strong> Based on model accuracy and feature importance</li>
            </ul>
        </div>
        
        <div class="timestamp">
            Generated by Optimized Random Forest ML Pipeline<br>
            Session: <a href="https://app.devin.ai/sessions/4fbfce1584a94376be1164d06cbf83c3">Devin AI Session</a><br>
            Requested by: @PremCouture
        </div>
    </div>
</body>
</html>
"""
    
    formatted_content = content.replace('<', '&lt;').replace('>', '&gt;')
    
    formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_content)
    formatted_content = re.sub(r'🟢', '<span class="emoji positive">🟢</span>', formatted_content)
    formatted_content = re.sub(r'🔴', '<span class="emoji negative">🔴</span>', formatted_content)
    formatted_content = re.sub(r'⚪', '<span class="emoji neutral">⚪</span>', formatted_content)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    html_content = html_template.format(
        content=formatted_content,
        timestamp=timestamp
    )
    
    html_filename = 'trading_analysis_results.html'
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Created HTML results: {html_filename}")
    return html_filename

if __name__ == "__main__":
    create_html_results()
