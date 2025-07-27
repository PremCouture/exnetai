#!/usr/bin/env python3
"""
ENHANCED TRADING SYSTEM WITH FULL PROPRIETARY FEATURE INTEGRATION
Version 5.0 - Complete implementation with data loader
- Includes all data loading functions
- Forces inclusion of ALL proprietary/technical features
- Creates extensive interaction features
- Adds non-linear transformations
- Shows complete feature presence in all outputs
- MODIFIED: Limited to 5 stocks for Colab memory constraints
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys
import shap
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, Counter
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import time
import gc

# Global caches for optimization
DATA_CACHE = {}
FEATURE_CACHE = {}
SHAP_CACHE = {}
EXPLAINER_CACHE = {}

# Global variable for Colab detection
IN_COLAB = False

# ==========================
# LOGGING CONFIGURATION
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================
# CONFIGURATION BLOCK
# ==========================

CONFIG = {
    # Paths
    'STOCK_DATA_PATH': '/content/drive/MyDrive/csv_files/stock_csvs/data',
    'FRED_ROOT_PATH': '/content/drive/MyDrive/csv_files/stock_csvs',

    # Analysis parameters
    'HORIZONS': [30, 45, 60],  # Prediction horizons in days
    'MIN_SAMPLES_PER_TICKER': 200,  # Minimum samples required
    'MIN_SAMPLES_FOR_TRAINING': 100,  # Minimum samples for model training
    'MAX_STOCKS': 5,  # LIMIT TO 5 STOCKS FOR COLAB

    # Feature configuration
    'EXCLUDE_FROM_SHAP': ['USREC', 'recession', 'binary_feature'],  # Features to exclude from SHAP
    'BINARY_VARIANCE_THRESHOLD': 0.05,  # Threshold for binary feature detection

    # Model parameters - UPDATED FOR BETTER FEATURE DIVERSITY
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42,
    'N_ESTIMATORS': 150,  # Increased for better feature exploration
    'MAX_DEPTH': 12,  # Increased depth
    'MAX_FEATURES': 0.5,  # Use 50% of features at each split (was 'sqrt')
    'MIN_SAMPLES_SPLIT': 30,  # Reduced to allow more splits
    'MIN_SAMPLES_LEAF': 10,  # Reduced for more granular decisions

    # Display parameters
    'MAX_SHAP_FEATURES': 5,  # Show top 5 features
    'CONFIDENCE_THRESHOLD': 45,

    # ALL Proprietary/Technical features to ALWAYS include
    'PROPRIETARY_FEATURES': [
        'VIX', 'FNG', 'RSI', 'AnnVolatility', 'Momentum125',
        'PriceStrength', 'VolumeBreadth', 'CallPut', 'NewsScore',
        'MACD', 'BollingerBandWidth', 'ATR', 'StochRSI',
        'OBV', 'CMF', 'ADX', 'Williams_R', 'CCI', 'MFI'
    ],

    # Regime thresholds for binary flags
    'REGIME_THRESHOLDS': {
        'VIX': {'extreme_high': 40, 'high': 30, 'low': 15, 'extreme_low': 10},
        'FNG': {'extreme_high': 85, 'high': 75, 'low': 25, 'extreme_low': 15},
        'RSI': {'extreme_high': 80, 'high': 70, 'low': 30, 'extreme_low': 20},
        'AnnVolatility': {'extreme_high': 50, 'high': 40, 'low': 20, 'extreme_low': 15},
        'VolumeBreadth': {'extreme_high': 2.0, 'high': 1.5, 'low': 0.5, 'extreme_low': 0.3},
        'Momentum125': {'extreme_high': 50, 'high': 30, 'low': -10, 'extreme_low': -30},
        'PriceStrength': {'extreme_high': 100, 'high': 50, 'low': -25, 'extreme_low': -50}
    },

    # Non-linear transformations to apply
    'TRANSFORMATIONS': ['log', 'square', 'sqrt', 'rank']
}

# Stock to ID mapping
STOCK_ALTERNATIVE_NAMES = {
    'MSFT': ['49462172'], 'AMZN': ['6248713'], 'NFLX': ['265768'], 'ENPH': ['105368327'],
    'TSLA': ['272093'], 'HPE': ['209411798'], 'META': ['270662'], 'AAPL': ['4661'],
    'KEYS': ['170167160'], 'AMD': ['265681'], 'GOOGL': ['5552'], 'VRSN': ['4173050'],
    'ACN': ['67889930'], 'STX': ['491932113'], 'NVDA': ['273544'], 'CDW': ['130432552'],
    'NXPI': ['77791077'], 'FTNT': ['70236214'], 'JNPR': ['6248722'], 'SNPS': ['274499'],
    'CRM': ['271568'], 'PYPL': ['270869'], 'FIS': ['37866641'], 'INTC': ['13096'],
}

# Create reverse mapping from ID to ticker name
STOCK_ID_TO_NAME = {}
for ticker, ids in STOCK_ALTERNATIVE_NAMES.items():
    for stock_id in ids:
        STOCK_ID_TO_NAME[stock_id] = ticker

# FRED indicator metadata with lag times
FRED_METADATA = {
    'GDP': {
        'name': 'Gross Domestic Product',
        'frequency': 'Quarterly',
        'lag_days': 45,
        'report_window': 90,
        'typical_release': 'Last Thursday of month after quarter end'
    },
    'UNRATE': {
        'name': 'Unemployment Rate',
        'frequency': 'Monthly',
        'lag_days': 15,
        'report_window': 30,
        'typical_release': 'First Friday of following month'
    },
    'CPIAUCSL': {
        'name': 'Consumer Price Index',
        'frequency': 'Monthly',
        'lag_days': 15,
        'report_window': 30,
        'typical_release': 'Mid-month for prior month'
    },
    'PAYEMS': {
        'name': 'Non-Farm Payrolls',
        'frequency': 'Monthly',
        'lag_days': 15,
        'report_window': 30,
        'typical_release': 'First Friday of following month'
    },
    'FEDFUNDS': {
        'name': 'Federal Funds Rate',
        'frequency': 'Daily',
        'lag_days': 2,
        'report_window': 1,
        'typical_release': 'Daily with 1-2 day lag'
    },
    'UMCSENT': {
        'name': 'Consumer Sentiment',
        'frequency': 'Monthly',
        'lag_days': 18,
        'report_window': 30,
        'typical_release': 'Mid and end of month'
    },
    'ICSA': {
        'name': 'Initial Jobless Claims',
        'frequency': 'Weekly',
        'lag_days': 5,
        'report_window': 7,
        'typical_release': 'Thursday for prior week'
    },
    'VIXCLS': {
        'name': 'VIX Volatility Index',
        'frequency': 'Daily',
        'lag_days': 1,
        'report_window': 1,
        'typical_release': 'Real-time/Daily'
    },
    'DGS10': {
        'name': '10-Year Treasury Rate',
        'frequency': 'Daily',
        'lag_days': 1,
        'report_window': 1,
        'typical_release': 'Daily'
    },
    'DGS2': {
        'name': '2-Year Treasury Rate',
        'frequency': 'Daily',
        'lag_days': 1,
        'report_window': 1,
        'typical_release': 'Daily'
    },
    'T10Y2Y': {
        'name': '10Y-2Y Treasury Spread',
        'frequency': 'Daily',
        'lag_days': 1,
        'report_window': 1,
        'typical_release': 'Daily'
    },
    'INDPRO': {
        'name': 'Industrial Production Index',
        'frequency': 'Monthly',
        'lag_days': 15,
        'report_window': 30,
        'typical_release': 'Mid-month for prior month'
    },
    'HOUST': {
        'name': 'Housing Starts',
        'frequency': 'Monthly',
        'lag_days': 18,
        'report_window': 30,
        'typical_release': 'Third week of following month'
    },
    'RETAILSL': {
        'name': 'Retail Sales',
        'frequency': 'Monthly',
        'lag_days': 15,
        'report_window': 30,
        'typical_release': 'Mid-month for prior month'
    },
    'AMERIBOR': {
        'name': 'American Interbank Offered Rate',
        'frequency': 'Daily',
        'lag_days': 2,
        'report_window': 1,
        'typical_release': 'Daily with 1-2 day lag'
    },
    'USREC': {
        'name': 'US Recession Indicator',
        'frequency': 'Monthly',
        'lag_days': 60,
        'report_window': 30,
        'typical_release': 'NBER declaration with significant lag',
        'is_binary': True
    }
}

# FRED indicator mapping from folder names to readable names - COMPREHENSIVE VERSION
FRED_INDICATOR_MAP = {
    'USREC_1': 'NBER Recession Indicator',
    'USPHCI_1': 'Housing Cost Index',
    'PCEPI_1': 'Core PCE Price Index',
    'PAYEMS_1': 'Non-Farm Payrolls',
    'OPHNFB_1': 'Productivity: Output per Hour',
    'MEHOINUSA672N_1': 'Median Household Income',
    'JHDUSRGDPBR_1': 'GDP-Based Recession Risk',
    'CPIAUCSL_1': 'Consumer Price Index',
    'AMERIBOR_1': 'AMERIBOR Rate',
    'GDP': 'Gross Domestic Product',
    'DGS10': '10-Year Treasury Rate',
    'RSAFS': 'Retail Sales',
    'FredFunds': 'Fed Funds Rate',
    'ICSA': 'Initial Jobless Claims',
    'UMCSENT': 'Consumer Sentiment Index',
    'STLFS12': 'St. Louis Financial Stress Index'
}

# ==========================
# HELPER FUNCTIONS
# ==========================

def get_stock_name_from_id(file_id):
    """Get stock ticker name from file ID using STOCK_ALTERNATIVE_NAMES mapping"""
    for ticker, ids in STOCK_ALTERNATIVE_NAMES.items():
        if file_id in ids:
            return ticker
    return None

def get_fred_name_from_folder(folder):
    """Get readable FRED indicator name from folder name"""
    return FRED_INDICATOR_MAP.get(folder, folder)

def load_all_stock_and_fred_data_enhanced(stock_path, fred_root, use_cache=True):
    """Enhanced data loader that scans directories for all available files"""
    stock_data = {}
    fred_data = {}
    
    if use_cache:
        cache_key = f"enhanced_loader_{stock_path}_{fred_root}"
        if cache_key in DATA_CACHE:
            logger.debug("Using cached enhanced data loader results")
            return DATA_CACHE[cache_key]
    
    logger.info(f"🔍 Loading STOCK files from: {stock_path}")
    if os.path.exists(stock_path):
        for file in os.listdir(stock_path):
            if file.endswith('.csv') and not file.endswith('.gsheet.csv'):
                file_id = file.replace('.csv', '')
                file_path = os.path.join(stock_path, file)
                try:
                    df = pd.read_csv(file_path)
                    stock_name = get_stock_name_from_id(file_id)
                    key = stock_name if stock_name else file_id
                    stock_data[key] = df
                    logger.info(f"✅ [Stock] {file} → {key} ({df.shape[0]} rows)")
                except Exception as e:
                    logger.warning(f"❌ Failed to load {file}: {e}")
    else:
        logger.warning(f"⚠️ Stock path does not exist: {stock_path}")

    logger.info(f"🔍 Loading FRED indicators from: {fred_root}")
    if os.path.exists(fred_root):
        for folder in sorted(os.listdir(fred_root)):
            folder_path = os.path.join(fred_root, folder)
            if not os.path.isdir(folder_path):
                continue
            
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if not csv_files:
                logger.debug(f"⚠️ Skipped (no CSV): {folder}")
                continue
            
            csv_path = os.path.join(folder_path, csv_files[0])
            try:
                df = pd.read_csv(csv_path)
                readable_name = get_fred_name_from_folder(folder)
                
                if len(df.columns) >= 2:
                    date_col = df.columns[0]
                    value_col = df.columns[1]
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(d in col_lower for d in ['date', 'time', 'period']):
                            date_col = col
                        elif col != date_col and any(v in col_lower for v in ['value', 'val', folder.replace('_1', '').lower()]):
                            value_col = col
                    
                    standardized_df = pd.DataFrame({
                        'Date': pd.to_datetime(df[date_col], errors='coerce'),
                        'Value': pd.to_numeric(df[value_col], errors='coerce')
                    }).dropna()
                    
                    if len(standardized_df) > 0:
                        fred_data[readable_name] = standardized_df
                        logger.info(f"✅ [FRED] {folder}/{csv_files[0]} → {readable_name} ({standardized_df.shape[0]} rows)")
                    else:
                        logger.warning(f"⚠️ No valid data in {folder}/{csv_files[0]}")
                else:
                    logger.warning(f"⚠️ Insufficient columns in {folder}/{csv_files[0]}")
                    
            except Exception as e:
                logger.warning(f"❌ Failed to load FRED file in {folder}: {e}")
    else:
        logger.warning(f"⚠️ FRED root path does not exist: {fred_root}")

    logger.info(f"📊 Loaded {len(stock_data)} stock datasets")
    logger.info(f"📊 Loaded {len(fred_data)} FRED indicators")
    
    result = (stock_data, fred_data)
    
    if use_cache:
        DATA_CACHE[cache_key] = result
        
    return result

def convert_np(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

def ensure_scalar(value):
    """Convert numpy/pandas objects to Python scalars safely"""
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.flatten()[0])
        elif value.size == 0:
            return 0
        else:
            return float(value[0])

    if isinstance(value, pd.Series):
        if len(value) == 1:
            return float(value.iloc[0])
        elif len(value) == 0:
            return 0
        else:
            return float(value.iloc[0])

    if hasattr(value, 'item'):
        try:
            return value.item()
        except ValueError:
            return 0

    try:
        return float(value)
    except:
        return value

def display_value(value, decimal_places=1):
    """Format value for display, showing '—' for missing values"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '—'
    elif isinstance(value, (int, np.integer)):
        return str(int(value))
    elif isinstance(value, (float, np.floating)):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        elif abs(value) >= 10:
            return f"{value:.{decimal_places}f}"
        elif abs(value) >= 1:
            return f"{value:.{decimal_places+1}f}"
        else:
            return f"{value:.{decimal_places+2}f}"
    else:
        return str(value)

def identify_binary_features(df, threshold=0.05):
    """Identify binary or low-variance features"""
    binary_features = []

    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 2:
                binary_features.append(col)
            else:
                # Check if variance is very low
                if df[col].std() / (df[col].mean() + 1e-8) < threshold:
                    binary_features.append(col)

    return binary_features

# ==========================
# TECHNICAL INDICATOR CALCULATIONS
# ==========================

def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    # Convert to pandas Series if it's a numpy array
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    # Convert to pandas Series if it's a numpy array
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line

    bullish_cross = False
    bearish_cross = False
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        try:
            curr_macd = float(macd_line.iloc[-1])
            prev_macd = float(macd_line.iloc[-2])
            curr_signal = float(signal_line.iloc[-1])
            prev_signal = float(signal_line.iloc[-2])

            if not any(pd.isna([curr_macd, prev_macd, curr_signal, prev_signal])):
                bullish_cross = bool(curr_macd > curr_signal and prev_macd <= prev_signal)
                bearish_cross = bool(curr_macd < curr_signal and prev_macd >= prev_signal)
        except:
            pass

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram,
        'bullish_cross': bullish_cross,
        'bearish_cross': bearish_cross
    }

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    # Convert to pandas Series if it's a numpy array
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)

    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band,
        'bandwidth': (upper_band - lower_band) / sma,
        'percent_b': (prices - lower_band) / (upper_band - lower_band)
    }

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    # Convert to pandas Series if they are numpy arrays
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr

def calculate_stochastic_rsi(rsi, period=14):
    """Calculate Stochastic RSI"""
    # Convert to pandas Series if it's a numpy array
    if isinstance(rsi, np.ndarray):
        rsi = pd.Series(rsi)
    
    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()

    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-8) * 100
    return stoch_rsi

def calculate_obv(prices, volumes):
    """Calculate On Balance Volume"""
    # Convert to pandas Series if they are numpy arrays
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    if isinstance(volumes, np.ndarray):
        volumes = pd.Series(volumes)
    
    price_change = prices.diff()
    obv = (volumes * np.sign(price_change)).cumsum()
    return obv

def calculate_cmf(high, low, close, volume, period=20):
    """Calculate Chaikin Money Flow"""
    # Convert to pandas Series if they are numpy arrays
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    if isinstance(volume, np.ndarray):
        volume = pd.Series(volume)
    
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-8)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(period).sum() / volume.rolling(period).sum()
    return cmf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # Convert to pandas Series if they are numpy arrays
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    
    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = calculate_atr(high, low, close, 1)

    plus_di = 100 * (plus_dm.rolling(period).sum() / tr.rolling(period).sum())
    minus_di = 100 * (minus_dm.rolling(period).sum() / tr.rolling(period).sum())

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(period).mean()

    return adx

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R"""
    # Convert to pandas Series if they are numpy arrays
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()

    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
    return williams_r

def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index"""
    # Convert to pandas Series if they are numpy arrays
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))

    cci = (typical_price - sma) / (0.015 * mad + 1e-8)
    return cci

def calculate_mfi(high, low, close, volume, period=14):
    """Calculate Money Flow Index"""
    # Convert to pandas Series if they are numpy arrays
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    if isinstance(volume, np.ndarray):
        volume = pd.Series(volume)
    
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()

    mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))
    return mfi

def calculate_momentum_125(prices):
    """Calculate 125-day momentum (approximately 6 months)"""
    if len(prices) < 125:
        return pd.Series(index=prices.index, dtype=float)
    return prices.pct_change(125) * 100

def calculate_price_strength(prices, volume=None):
    """Calculate price strength indicator"""
    # Price momentum over different periods
    mom_5 = prices.pct_change(5)
    mom_20 = prices.pct_change(20)
    mom_60 = prices.pct_change(60)

    # Weighted average
    price_strength = (mom_5 * 0.5 + mom_20 * 0.3 + mom_60 * 0.2) * 100

    # If volume available, incorporate volume-weighted strength
    if volume is not None and len(volume) > 0:
        vol_ratio = volume / volume.rolling(20).mean()
        price_strength = price_strength * vol_ratio.clip(0.5, 2.0)

    return price_strength

def calculate_volume_breadth(volume, prices):
    """Calculate volume breadth indicator"""
    if volume is None or len(volume) == 0:
        return pd.Series(index=prices.index, data=1.0)

    # Up volume vs down volume
    price_change = prices.diff()
    up_volume = volume.where(price_change > 0, 0)
    down_volume = volume.where(price_change < 0, 0)

    # Rolling sums
    up_vol_sum = up_volume.rolling(20).sum()
    down_vol_sum = down_volume.rolling(20).sum()

    # Breadth ratio
    breadth = up_vol_sum / (down_vol_sum + 1e-8)
    return breadth.clip(0.1, 10.0)

def calculate_trend_strength(series, period):
    """Calculate trend strength using linear regression slope"""
    def trend_calc(window):
        if len(window) < period // 2:
            return 0
        x = np.arange(len(window))
        y = window.values
        if np.std(y) == 0:
            return 0
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope / (np.std(y) + 1e-6)
        except:
            return 0

    return series.rolling(period).apply(trend_calc)

def calculate_performance_metrics(y_true, y_pred, returns, prediction_days):
    """Calculate comprehensive performance metrics"""
    # Basic metrics
    accuracy = (y_true == y_pred).mean() * 100

    # Direction-based returns
    strategy_returns = returns * (2 * y_pred - 1)  # Convert 0/1 to -1/1

    # Annualized metrics
    periods_per_year = 252 / prediction_days

    # Sharpe ratio
    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
        sharpe = np.sqrt(periods_per_year) * strategy_returns.mean() / strategy_returns.std()
    else:
        sharpe = 0

    # Win rate
    winning_trades = strategy_returns > 0
    win_rate = winning_trades.mean() * 100 if len(winning_trades) > 0 else 0

    # Maximum drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown_series = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown_series.min() * 100

    # Annualized return
    avg_return = strategy_returns.mean()
    annualized_return = (1 + avg_return) ** periods_per_year - 1

    return {
        'accuracy': accuracy,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'annualized_return': annualized_return * 100,
        'avg_return_per_trade': avg_return * 100
    }

# ==========================
# DATA LOADING FUNCTIONS
# ==========================

def get_stock_id_from_ticker(ticker):
    """Get numeric ID for a ticker symbol"""
    ids = STOCK_ALTERNATIVE_NAMES.get(ticker, [])
    return ids[0] if ids else None

def standardize_columns(df):
    """Standardize column names to expected format"""
    column_mapping = {
        'close': 'Close', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'volume': 'Volume', 'adj close': 'Adj Close',
        'vix': 'VIX', 'fng': 'FNG', 'annvolatility': 'AnnVolatility',
        'momentum125': 'Momentum125', 'pricestrength': 'PriceStrength',
        'volumebreadth': 'VolumeBreadth', 'callput': 'CallPut',
        'newsscore': 'NewsScore', 'rsi': 'RSI', 'macd': 'MACD',
        'bollingerbandwidth': 'BollingerBandWidth', 'atr': 'ATR',
        'stochrsi': 'StochRSI', 'obv': 'OBV', 'cmf': 'CMF',
        'adx': 'ADX', 'williams_r': 'Williams_R', 'cci': 'CCI',
        'mfi': 'MFI'
    }

    df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]

    required = ['Close', 'Open', 'High', 'Low', 'Volume']
    for col in required:
        if col not in df.columns:
            if col == 'Volume':
                df[col] = 1000000
            else:
                df[col] = df.get('Close', 100)

    return df

def load_stock_data(ticker, csv_path=None, use_cache=True):
    """Load individual stock data with enhanced error handling and caching"""
    if csv_path is None:
        csv_path = CONFIG['STOCK_DATA_PATH']
    
    if use_cache:
        cache_key = f"stock_{ticker}_{csv_path}"
        if cache_key in DATA_CACHE:
            logger.debug(f"Using cached data for {ticker}")
            return DATA_CACHE[cache_key].copy()

    try:
        stock_id = get_stock_id_from_ticker(ticker)
        file_name = f"{stock_id}.csv" if stock_id else f"{ticker}.csv"
        file_path = os.path.join(csv_path, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Process date column
            date_col = None
            for col in ['Date', 'date', 'DATE', 'Datetime', 'datetime', 'timestamp']:
                if col in df.columns:
                    date_col = col
                    break

            if date_col:
                df['Date'] = pd.to_datetime(df[date_col])
                if date_col != 'Date':
                    df = df.drop(columns=[date_col])
            elif df.index.name and 'date' in df.index.name.lower():
                df['Date'] = pd.to_datetime(df.index)
                df = df.reset_index(drop=True)

            df = standardize_columns(df)

            # Log ALL features found
            found_features = {feat: feat in df.columns for feat in CONFIG['PROPRIETARY_FEATURES']}

            if 'Date' in df.columns:
                df = df.sort_values('Date')

                # Data sanity check
                logger.info(f"{ticker} - Date range: {df['Date'].min()} to {df['Date'].max()}")
                logger.info(f"{ticker} - Total rows: {len(df)}")
                logger.info(f"{ticker} - Proprietary features found: {[k for k,v in found_features.items() if v]}")
                logger.info(f"{ticker} - Missing proprietary features: {[k for k,v in found_features.items() if not v]}")

            return df
        else:
            logger.warning(f"No data file found for {ticker}")
            return None

    except Exception as e:
        logger.error(f"Error loading {ticker}: {e}")
        return None
    
    if use_cache and df is not None:
        cache_key = f"stock_{ticker}_{csv_path}"
        DATA_CACHE[cache_key] = df.copy()
        logger.debug(f"Cached data for {ticker}")
    
    return df

def load_all_stock_data(tickers, csv_path=None, use_cache=True):
    """Load stock data for multiple tickers with validation and batch processing"""
    stock_data = {}

    logger.info(f"Loading {len(tickers)} stocks with caching...")

    # Batch process tickers for better I/O efficiency
    for ticker in tickers:
        df = load_stock_data(ticker, csv_path, use_cache=use_cache)
        if df is not None and len(df) >= CONFIG['MIN_SAMPLES_PER_TICKER']:
            stock_data[ticker] = df
            logger.info(f"Loaded {ticker} ({len(df)} rows)")
        elif df is not None:
            logger.warning(f"Skipped {ticker} - insufficient data ({len(df)} rows, need {CONFIG['MIN_SAMPLES_PER_TICKER']})")

    logger.info(f"Successfully loaded {len(stock_data)} stocks")
    return stock_data

def load_fred_data_from_folders(use_cache=True):
    """Load FRED data from folder structure with caching"""
    fred_data = {}
    
    if use_cache:
        cache_key = f"fred_data_{CONFIG['FRED_ROOT_PATH']}"
        if cache_key in DATA_CACHE:
            logger.debug("Using cached FRED data")
            return DATA_CACHE[cache_key].copy()

    if not os.path.exists(CONFIG['FRED_ROOT_PATH']):
        logger.warning(f"FRED path not found: {CONFIG['FRED_ROOT_PATH']}")
        return fred_data

    logger.info("Loading FRED indicators...")

    fred_patterns = list(FRED_METADATA.keys()) + ['USREC', 'AMERIBOR', 'PCEPI', 'MEHOINUSA', 'OPHNFB']

    for folder in os.listdir(CONFIG['FRED_ROOT_PATH']):
        folder_path = os.path.join(CONFIG['FRED_ROOT_PATH'], folder)
        if not os.path.isdir(folder_path):
            continue

        folder_upper = folder.upper().replace('_1', '')
        is_fred = any(pattern in folder_upper for pattern in fred_patterns)

        if is_fred or folder.endswith('_1'):
            csv_names = [
                'obs._by_real-time_period.csv',
                'obs_by_real-time_period.csv',
                'obs.csv',
                'data.csv',
                f'{folder}.csv',
                f'{folder.lower()}.csv'
            ]

            for csv_name in csv_names:
                csv_path = os.path.join(folder_path, csv_name)

                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)

                        value_col = None
                        date_col = None

                        for col in df.columns:
                            col_lower = col.lower()
                            if any(d in col_lower for d in ['date', 'time', 'period']):
                                date_col = col
                                break

                        folder_base = folder.replace('_1', '')
                        if folder_base in df.columns:
                            value_col = folder_base
                        elif folder_base.upper() in df.columns:
                            value_col = folder_base.upper()
                        elif folder_base.lower() in df.columns:
                            value_col = folder_base.lower()
                        else:
                            for col in df.columns:
                                col_lower = col.lower()
                                if col != date_col and any(v in col_lower for v in ['value', 'val', 'observation']):
                                    value_col = col
                                    break

                        if value_col and date_col:
                            indicator_base = folder_base.upper()
                            if indicator_base in FRED_METADATA:
                                indicator_name = FRED_METADATA[indicator_base]['name']
                            else:
                                indicator_name = folder_base

                            fred_df = pd.DataFrame({
                                'Date': pd.to_datetime(df[date_col], errors='coerce'),
                                'Value': pd.to_numeric(df[value_col], errors='coerce')
                            }).dropna()

                            if len(fred_df) > 0:
                                fred_data[indicator_base] = fred_df
                                logger.info(f"Loaded {indicator_name} ({len(fred_df)} rows)")
                                break

                    except Exception as e:
                        continue

    logger.info(f"Loaded {len(fred_data)} FRED indicators")
    
    if use_cache and fred_data:
        cache_key = f"fred_data_{CONFIG['FRED_ROOT_PATH']}"
        DATA_CACHE[cache_key] = fred_data.copy()
        logger.debug("Cached FRED data")
    
    return fred_data

# ==========================
# MACRO DATA ALIGNMENT AND MERGE
# ==========================

def fix_macro_data_alignment(fred_data):
    """Fix macro data alignment with proper lags to prevent look-ahead bias"""
    aligned_fred = {}

    logger.info("Aligning FRED data with proper lags...")

    for indicator_name, fred_df in fred_data.items():
        if fred_df.empty:
            continue

        try:
            df = fred_df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')

            if indicator_name in FRED_METADATA:
                meta = FRED_METADATA[indicator_name]
                lag_days = meta['lag_days']
                data_type = meta['frequency']
                report_window = meta.get('report_window', lag_days)
            else:
                if len(df) > 2:
                    avg_days = df['Date'].diff().dt.days.median()

                    if avg_days > 300:
                        data_type = 'Annual'
                        lag_days = 60
                        report_window = 365
                    elif avg_days > 80:
                        data_type = 'Quarterly'
                        lag_days = 45
                        report_window = 90
                    elif avg_days > 25:
                        data_type = 'Monthly'
                        lag_days = 15
                        report_window = 30
                    elif avg_days > 5:
                        data_type = 'Weekly'
                        lag_days = 5
                        report_window = 7
                    else:
                        data_type = 'Daily'
                        lag_days = 2
                        report_window = 1
                else:
                    data_type = 'Unknown'
                    lag_days = 30
                    report_window = 30

            safety_margin = 3
            total_lag = lag_days + safety_margin
            df['Date'] = df['Date'] + pd.Timedelta(days=total_lag)

            df['lag_days'] = lag_days
            df['total_lag'] = total_lag
            df['data_type'] = data_type
            df['report_window'] = report_window
            df['original_date'] = df['Date'] - pd.Timedelta(days=total_lag)

            lag_suffix = f"_{data_type[0].lower()}{total_lag}"
            aligned_indicator_name = f"{indicator_name}{lag_suffix}"

            aligned_fred[aligned_indicator_name] = df
            logger.info(f"  {indicator_name} → {aligned_indicator_name}: {data_type} data, {total_lag}d total lag applied")

        except Exception as e:
            logger.error(f"Error aligning {indicator_name}: {e}")

    return aligned_fred

def merge_macro_with_stock(stock_data, fred_data):
    """Merge macro data with stock data using proper temporal alignment"""
    logger.info("Merging macro features with stock data...")

    all_dates = []
    for ticker, df in stock_data.items():
        if 'Date' in df.columns:
            all_dates.extend(df['Date'].tolist())

    if not all_dates:
        return stock_data, {}

    all_dates = pd.to_datetime(all_dates)
    unique_dates = sorted(list(set(all_dates)))

    macro_df = pd.DataFrame({'Date': unique_dates})
    macro_features_added = {}

    for indicator_name, fred_df in fred_data.items():
        if fred_df.empty or 'Value' not in fred_df.columns:
            continue

        col_name = f"fred_{indicator_name}"

        fred_df = fred_df.copy()
        fred_df = fred_df.rename(columns={'Value': col_name})

        # Use forward fill for macro data
        macro_df = pd.merge_asof(
            macro_df.sort_values('Date'),
            fred_df[['Date', col_name]].sort_values('Date'),
            on='Date',
            direction='backward'
        )

        base_indicator = indicator_name.split('_')[0].upper()

        if 'lag_days' in fred_df.columns:
            lag_days = fred_df['lag_days'].iloc[0]
            total_lag = fred_df['total_lag'].iloc[0] if 'total_lag' in fred_df.columns else lag_days
            data_type = fred_df['data_type'].iloc[0] if 'data_type' in fred_df.columns else 'Unknown'
            report_window = fred_df['report_window'].iloc[0] if 'report_window' in fred_df.columns else lag_days
        else:
            import re
            match = re.search(r'_([a-z])(\d+)$', indicator_name)
            if match:
                freq_char, lag = match.groups()
                total_lag = int(lag)
                lag_days = total_lag - 3

                freq_map = {'d': 'Daily', 'w': 'Weekly', 'm': 'Monthly', 'q': 'Quarterly', 'a': 'Annual'}
                data_type = freq_map.get(freq_char, 'Unknown')

                window_map = {'d': 1, 'w': 7, 'm': 30, 'q': 90, 'a': 365}
                report_window = window_map.get(freq_char, 30)
            else:
                lag_days = 30
                total_lag = 33
                data_type = 'Unknown'
                report_window = 30

        if base_indicator in FRED_METADATA:
            full_meta = FRED_METADATA[base_indicator].copy()
            full_meta.update({
                'lag_days': lag_days,
                'total_lag': total_lag,
                'actual_frequency': data_type,
                'report_window': report_window,
                'column_name': col_name,
                'indicator_with_lag': indicator_name
            })
        else:
            full_meta = {
                'name': base_indicator,
                'lag_days': lag_days,
                'total_lag': total_lag,
                'frequency': data_type,
                'actual_frequency': data_type,
                'report_window': report_window,
                'column_name': col_name,
                'indicator_with_lag': indicator_name,
                'typical_release': f'{data_type} data with {lag_days}d lag'
            }

        macro_features_added[col_name] = full_meta

    # Apply forward fill to macro data
    macro_cols = [col for col in macro_df.columns if col.startswith('fred_')]
    macro_df[macro_cols] = macro_df[macro_cols].fillna(method='ffill').fillna(method='bfill')

    merged_stock_data = {}
    for ticker, stock_df in stock_data.items():
        if 'Date' not in stock_df.columns:
            merged_stock_data[ticker] = stock_df
            continue

        stock_df = stock_df.copy()
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        merged_df = pd.merge(stock_df, macro_df, on='Date', how='left')

        # Remove USREC and other binary features after merge
        binary_features_to_remove = []
        for col in merged_df.columns:
            if col.startswith('fred_'):
                base_name = col.replace('fred_', '').split('_')[0].upper()
                if base_name in CONFIG['EXCLUDE_FROM_SHAP']:
                    binary_features_to_remove.append(col)

        if binary_features_to_remove:
            logger.info(f"Removing binary features from {ticker}: {binary_features_to_remove}")
            merged_df = merged_df.drop(columns=binary_features_to_remove)
            for feat in binary_features_to_remove:
                if feat in macro_features_added:
                    del macro_features_added[feat]

        merged_stock_data[ticker] = merged_df

        # Data sanity check after merge
        logger.info(f"{ticker} - Shape after merge: {merged_df.shape}")

    logger.info(f"Added {len(macro_features_added)} macro features (after filtering)")

    return merged_stock_data, macro_features_added

# ==========================
# FEATURE ENGINEERING
# ==========================

def create_proprietary_features(df):
    """Create or ensure ALL proprietary features exist - OPTIMIZED VERSION"""
    features = pd.DataFrame(index=df.index)

    # Ensure we have required price data
    if 'Close' not in df.columns:
        logger.error("No Close price data available!")
        return features

    close_prices = df['Close'].values
    high_prices = df.get('High', df['Close']).values
    low_prices = df.get('Low', df['Close']).values
    volume = df.get('Volume', pd.Series(1000000, index=df.index)).values

    # OPTIMIZATION 1: Pre-compute common calculations once
    returns = np.diff(close_prices) / close_prices[:-1]
    returns = np.concatenate([[0], returns])  # Add zero for first value
    
    # Cache rolling calculations
    returns_series = pd.Series(returns, index=df.index)
    vol_30 = returns_series.rolling(30).std()
    vol_20 = returns_series.rolling(20).std()
    vol_252 = returns_series.rolling(252).std()
    
    momentum_20 = pd.Series(close_prices, index=df.index).pct_change(20)
    
    # OPTIMIZATION 2: Batch feature creation with caching
    feature_cache = {}
    
    # VIX - if not present, calculate from volatility
    if 'VIX' in df.columns:
        features['VIX'] = df['VIX']
    else:
        if 'vix_calc' not in feature_cache:
            feature_cache['vix_calc'] = vol_30 * np.sqrt(252) * 100
        features['VIX'] = feature_cache['vix_calc'].fillna(20)

    # FNG (Fear & Greed) - only include if present in data
    if 'FNG' in df.columns:
        features['FNG'] = df['FNG']

    # RSI
    if 'RSI' in df.columns:
        features['RSI'] = df['RSI']
    else:
        if 'rsi_calc' not in feature_cache:
            feature_cache['rsi_calc'] = calculate_rsi(pd.Series(close_prices, index=df.index))
        features['RSI'] = feature_cache['rsi_calc'].fillna(50)

    # Annual Volatility
    if 'AnnVolatility' in df.columns:
        features['AnnVolatility'] = df['AnnVolatility']
    else:
        if 'ann_vol_calc' not in feature_cache:
            feature_cache['ann_vol_calc'] = vol_252 * np.sqrt(252) * 100
        features['AnnVolatility'] = feature_cache['ann_vol_calc'].fillna(20)

    # Momentum 125
    if 'Momentum125' in df.columns:
        features['Momentum125'] = df['Momentum125']
    else:
        if 'mom125_calc' not in feature_cache:
            feature_cache['mom125_calc'] = calculate_momentum_125(pd.Series(close_prices, index=df.index))
        features['Momentum125'] = feature_cache['mom125_calc'].fillna(0)

    # Price Strength
    if 'PriceStrength' in df.columns:
        features['PriceStrength'] = df['PriceStrength']
    else:
        if 'price_strength_calc' not in feature_cache:
            feature_cache['price_strength_calc'] = calculate_price_strength(
                pd.Series(close_prices, index=df.index), 
                pd.Series(volume, index=df.index)
            )
        features['PriceStrength'] = feature_cache['price_strength_calc'].fillna(0)

    # Volume Breadth
    if 'VolumeBreadth' in df.columns:
        features['VolumeBreadth'] = df['VolumeBreadth']
    else:
        if 'vol_breadth_calc' not in feature_cache:
            feature_cache['vol_breadth_calc'] = calculate_volume_breadth(
                pd.Series(volume, index=df.index), 
                pd.Series(close_prices, index=df.index)
            )
        features['VolumeBreadth'] = feature_cache['vol_breadth_calc'].fillna(1)

    # Call/Put Ratio - only include if present in data
    if 'CallPut' in df.columns:
        features['CallPut'] = df['CallPut']

    # News Score - only include if present in data
    if 'NewsScore' in df.columns:
        features['NewsScore'] = df['NewsScore']

    # MACD
    if 'MACD' in df.columns:
        features['MACD'] = df['MACD']
    else:
        if 'macd_calc' not in feature_cache:
            macd_data = calculate_macd(pd.Series(close_prices, index=df.index))
            feature_cache['macd_calc'] = macd_data['macd']
        features['MACD'] = feature_cache['macd_calc'].fillna(0)

    # Bollinger Band Width
    if 'BollingerBandWidth' in df.columns:
        features['BollingerBandWidth'] = df['BollingerBandWidth']
    else:
        if 'bb_calc' not in feature_cache:
            bb_data = calculate_bollinger_bands(pd.Series(close_prices, index=df.index))
            feature_cache['bb_calc'] = bb_data
        features['BollingerBandWidth'] = bb_data['bandwidth'] * 100
        features['BollingerBandWidth'] = features['BollingerBandWidth'].fillna(2)

    # ATR
    if 'ATR' in df.columns:
        features['ATR'] = df['ATR']
    else:
        features['ATR'] = calculate_atr(high_prices, low_prices, close_prices)
        features['ATR'] = features['ATR'].fillna(1)

    # Stochastic RSI
    if 'StochRSI' in df.columns:
        features['StochRSI'] = df['StochRSI']
    else:
        rsi = features['RSI']
        features['StochRSI'] = calculate_stochastic_rsi(rsi)
        features['StochRSI'] = features['StochRSI'].fillna(50)

    # Additional indicators
    # OBV
    if 'OBV' in df.columns:
        features['OBV'] = df['OBV']
    else:
        features['OBV'] = calculate_obv(close_prices, volume)
        features['OBV'] = features['OBV'].fillna(0)

    # CMF
    if 'CMF' in df.columns:
        features['CMF'] = df['CMF']
    else:
        features['CMF'] = calculate_cmf(high_prices, low_prices, close_prices, volume)
        features['CMF'] = features['CMF'].fillna(0)

    # ADX
    if 'ADX' in df.columns:
        features['ADX'] = df['ADX']
    else:
        features['ADX'] = calculate_adx(high_prices, low_prices, close_prices)
        features['ADX'] = features['ADX'].fillna(25)

    # Williams %R
    if 'Williams_R' in df.columns:
        features['Williams_R'] = df['Williams_R']
    else:
        features['Williams_R'] = calculate_williams_r(high_prices, low_prices, close_prices)
        features['Williams_R'] = features['Williams_R'].fillna(-50)

    # CCI
    if 'CCI' in df.columns:
        features['CCI'] = df['CCI']
    else:
        features['CCI'] = calculate_cci(high_prices, low_prices, close_prices)
        features['CCI'] = features['CCI'].fillna(0)

    # MFI
    if 'MFI' in df.columns:
        features['MFI'] = df['MFI']
    else:
        features['MFI'] = calculate_mfi(high_prices, low_prices, close_prices, volume)
        features['MFI'] = features['MFI'].fillna(50)

    # Fill any remaining NaN values with appropriate defaults
    features = features.fillna(method='ffill').fillna(method='bfill')

    # Final fill with defaults
    for col in features.columns:
        if features[col].isna().any():
            if col in ['VIX', 'AnnVolatility']:
                features[col] = features[col].fillna(20)
            elif col in ['FNG', 'RSI', 'StochRSI', 'MFI', 'CallPut']:
                features[col] = features[col].fillna(50)
            elif col in ['ADX']:
                features[col] = features[col].fillna(25)
            elif col in ['Williams_R']:
                features[col] = features[col].fillna(-50)
            elif col == 'NewsScore':
                features[col] = features[col].fillna(5)
            elif col == 'VolumeBreadth':
                features[col] = features[col].fillna(1)
            elif col == 'BollingerBandWidth':
                features[col] = features[col].fillna(2)
            else:
                features[col] = features[col].fillna(0)

    logger.info(f"Created {len(features.columns)} proprietary features: {list(features.columns)}")

    return features

def create_regime_features(proprietary_features):
    """Create binary regime features based on thresholds"""
    regime_features = pd.DataFrame(index=proprietary_features.index)

    for feature, thresholds in CONFIG['REGIME_THRESHOLDS'].items():
        if feature in proprietary_features.columns:
            feat_data = proprietary_features[feature]

            # Create all regime levels
            for level in ['extreme_high', 'high', 'low', 'extreme_low']:
                if level in thresholds:
                    if 'high' in level:
                        regime_features[f'{feature}_{level}'] = (
                            feat_data > thresholds[level]
                        ).astype(int)
                    else:  # low levels
                        regime_features[f'{feature}_{level}'] = (
                            feat_data < thresholds[level]
                        ).astype(int)

            # Create combined regime indicators
            if 'high' in thresholds and 'low' in thresholds:
                regime_features[f'{feature}_neutral'] = (
                    (feat_data >= thresholds['low']) &
                    (feat_data <= thresholds['high'])
                ).astype(int)

    logger.info(f"Created {len(regime_features.columns)} regime features")

    return regime_features

def create_nonlinear_transformations(features):
    """Create non-linear transformations of proprietary features - OPTIMIZED VERSION"""
    transformed_features = pd.DataFrame(index=features.index)

    # Filter proprietary features once
    prop_features = [col for col in features.columns if col in CONFIG['PROPRIETARY_FEATURES']]
    
    if not prop_features:
        logger.info("No proprietary features found for transformation")
        return transformed_features

    # OPTIMIZATION 1: Vectorized operations on all proprietary features at once
    prop_data = features[prop_features]
    
    # Pre-compute statistics for all features
    min_vals = prop_data.min()
    mean_vals = prop_data.mean()
    std_vals = prop_data.std()
    
    # OPTIMIZATION 2: Batch transformations using numpy operations
    for col in prop_features:
        feat_data = prop_data[col].values
        min_val = min_vals[col]
        mean_val = mean_vals[col]
        std_val = std_vals[col]

        # Log transformation (vectorized)
        if min_val <= 0:
            shifted_data = feat_data - min_val + 1
            transformed_features[f'{col}_log'] = np.log(shifted_data)
        else:
            transformed_features[f'{col}_log'] = np.log(feat_data + 1e-8)

        # Square transformation (vectorized)
        transformed_features[f'{col}_square'] = feat_data ** 2

        # Square root (vectorized)
        if min_val >= 0:
            transformed_features[f'{col}_sqrt'] = np.sqrt(feat_data + 1e-8)
        else:
            transformed_features[f'{col}_sqrt'] = np.sqrt(np.abs(feat_data) + 1e-8)

        # Rank transformation (optimized)
        transformed_features[f'{col}_rank'] = rankdata(feat_data) / len(feat_data)

        # Percentile transformation (using pandas for efficiency)
        transformed_features[f'{col}_pct'] = prop_data[col].rank(pct=True).values

        # Z-score normalization (vectorized)
        if std_val > 0:
            transformed_features[f'{col}_zscore'] = (feat_data - mean_val) / std_val
        else:
            transformed_features[f'{col}_zscore'] = np.zeros_like(feat_data)

    # OPTIMIZATION 3: Vectorized clipping for all columns at once
    if len(transformed_features.columns) > 0:
        transformed_features = transformed_features.clip(-10, 10)

    logger.info(f"Created {len(transformed_features.columns)} non-linear transformations (OPTIMIZED)")

    return transformed_features

def create_comprehensive_interaction_features(macro_features, proprietary_features, regime_features):
    """Create extensive interaction features between all feature types - PERFORMANCE OPTIMIZED"""
    import time
    start_time = time.time()
    interaction_features = pd.DataFrame(index=macro_features.index)

    # Get feature columns by type
    macro_cols = [col for col in macro_features.columns if 'fred_' in col]
    prop_cols = [col for col in proprietary_features.columns if col in CONFIG['PROPRIETARY_FEATURES']]
    regime_cols = regime_features.columns.tolist()

    top_macro_cols = macro_cols[:6]  # Further reduced from 10 to 6 for performance
    top_prop_cols = [col for col in prop_cols if col in ['VIX', 'FNG', 'RSI', 'Momentum125', 'AnnVolatility', 'PriceStrength']][:6]  # Top 6 proprietary
    
    logger.info(f"Creating optimized interactions: {len(top_macro_cols)} macro × {len(top_prop_cols)} proprietary (performance limited)")

    # OPTIMIZATION 1: Vectorized Macro × Proprietary interactions
    if top_macro_cols and top_prop_cols:
        macro_array = macro_features[top_macro_cols].values
        prop_array = proprietary_features[top_prop_cols].values
        
        # Vectorized outer product for all combinations
        macro_expanded = macro_array[:, :, np.newaxis]
        prop_expanded = prop_array[:, np.newaxis, :]
        
        interactions = macro_expanded * prop_expanded
        
        # Create column names and flatten
        interaction_names = [f"{macro}_X_{prop}" for macro in top_macro_cols for prop in top_prop_cols]
        interactions_flat = interactions.reshape(len(macro_features), -1)
        
        # Add to DataFrame in batch
        interaction_df = pd.DataFrame(interactions_flat, columns=interaction_names, index=macro_features.index)
        interaction_features = pd.concat([interaction_features, interaction_df], axis=1)
        
        logger.info(f"Created {len(interaction_names)} macro × proprietary interactions (vectorized)")

    # OPTIMIZATION 2: Selective Macro × Regime interactions (vectorized) - FURTHER REDUCED
    key_regime_cols = [col for col in regime_cols if any(prop in col for prop in ['VIX', 'FNG', 'RSI'])][:3]  # Top 3 only
    top_macro_regime_cols = top_macro_cols[:4]  # Further reduced to top 4
    
    if top_macro_regime_cols and key_regime_cols:
        macro_regime_array = macro_features[top_macro_regime_cols].values
        regime_array = regime_features[key_regime_cols].values
        
        # Vectorized interactions
        macro_regime_expanded = macro_regime_array[:, :, np.newaxis]
        regime_expanded = regime_array[:, np.newaxis, :]
        
        regime_interactions = macro_regime_expanded * regime_expanded
        regime_names = [f"{macro}_X_{regime}" for macro in top_macro_regime_cols for regime in key_regime_cols]
        regime_flat = regime_interactions.reshape(len(macro_features), -1)
        
        regime_df = pd.DataFrame(regime_flat, columns=regime_names, index=macro_features.index)
        interaction_features = pd.concat([interaction_features, regime_df], axis=1)
        
        logger.info(f"Created {len(regime_names)} macro × regime interactions (vectorized)")

    # OPTIMIZATION 3: Pre-computed triple interactions (cached approach)
    important_interactions = [
        ('CPIAUCSL', 'VIX', 'VIX_high'),
        ('DGS10', 'AnnVolatility', 'VIX_extreme_high'),
        ('UNRATE', 'FNG', 'FNG_low'),
        ('GDP', 'Momentum125', 'Momentum125_high'),
        ('FEDFUNDS', 'PriceStrength', 'RSI_extreme_high'),
        ('AMERIBOR', 'VolumeBreadth', 'VIX_low'),
        ('PAYEMS', 'RSI', 'FNG_extreme_low'),
        ('RETAILSL', 'MACD', 'Momentum125_low')
    ]

    # Batch process triple interactions
    triple_data = {}
    for macro_base, prop, regime in important_interactions:
        macro_col = next((col for col in macro_cols if macro_base in col), None)
        
        if macro_col and prop in proprietary_features.columns and regime in regime_features.columns:
            interaction_name = f"{macro_col}_X_{prop}_X_{regime}"
            triple_data[interaction_name] = (
                macro_features[macro_col].values *
                proprietary_features[prop].values *
                regime_features[regime].values
            )
    
    if triple_data:
        triple_df = pd.DataFrame(triple_data, index=macro_features.index)
        interaction_features = pd.concat([interaction_features, triple_df], axis=1)
        logger.info(f"Created {len(triple_data)} triple interactions (batch processed)")

    # OPTIMIZATION 4: Vectorized Proprietary × Proprietary interactions
    prop_interactions = [
        ('VIX', 'FNG'), ('VIX', 'RSI'), ('VIX', 'Momentum125'),
        ('FNG', 'RSI'), ('Momentum125', 'PriceStrength'),
        ('AnnVolatility', 'VolumeBreadth'), ('MACD', 'RSI'), ('ATR', 'ADX')
    ]

    prop_data = {}
    for prop1, prop2 in prop_interactions:
        if prop1 in proprietary_features.columns and prop2 in proprietary_features.columns:
            interaction_name = f"{prop1}_X_{prop2}"
            prop_data[interaction_name] = (
                proprietary_features[prop1].values * proprietary_features[prop2].values
            )
    
    if prop_data:
        prop_df = pd.DataFrame(prop_data, index=proprietary_features.index)
        interaction_features = pd.concat([interaction_features, prop_df], axis=1)
        logger.info(f"Created {len(prop_data)} proprietary × proprietary interactions (vectorized)")

    # OPTIMIZATION 5: Vectorized normalization
    if len(interaction_features.columns) > 0:
        # Clip extreme values (vectorized)
        interaction_features = interaction_features.clip(-1000, 1000)
        
        # Vectorized standardization
        means = interaction_features.mean()
        stds = interaction_features.std()
        
        # Only standardize columns with non-zero std
        valid_cols = stds[stds > 0].index
        if len(valid_cols) > 0:
            interaction_features[valid_cols] = (interaction_features[valid_cols] - means[valid_cols]) / stds[valid_cols]
            interaction_features[valid_cols] = interaction_features[valid_cols].clip(-5, 5)

    total_interactions = len(interaction_features.columns)
    logger.info(f"Created {total_interactions} total interaction features (OPTIMIZED)")

    return interaction_features

def create_technical_features(df):
    """Create comprehensive technical features"""
    features = pd.DataFrame(index=df.index)

    # Price-based features
    features['returns_1d'] = df['Close'].pct_change()
    features['returns_5d'] = df['Close'].pct_change(5)
    features['returns_20d'] = df['Close'].pct_change(20)
    features['returns_60d'] = df['Close'].pct_change(60)

    # Log returns
    close_shifted = df['Close'].shift(1)
    mask = (df['Close'] > 0) & (close_shifted > 0)
    features['log_returns_1d'] = 0.0
    features.loc[mask, 'log_returns_1d'] = np.log(df.loc[mask, 'Close'] / close_shifted[mask])

    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        if len(df) >= period:
            sma = df['Close'].rolling(window=period).mean()
            ema = df['Close'].ewm(span=period).mean()

            features[f'price_to_sma_{period}'] = 1.0
            features[f'price_to_ema_{period}'] = 1.0

            mask_sma = sma > 0
            mask_ema = ema > 0

            features.loc[mask_sma, f'price_to_sma_{period}'] = df.loc[mask_sma, 'Close'] / sma[mask_sma]
            features.loc[mask_ema, f'price_to_ema_{period}'] = df.loc[mask_ema, 'Close'] / ema[mask_ema]

            features[f'sma_{period}_slope'] = sma.pct_change(5)

    # Support/Resistance
    high_20d = df['High'].rolling(20).max()
    low_20d = df['Low'].rolling(20).min()
    high_52w = df['High'].rolling(252).max()
    low_52w = df['Low'].rolling(252).min()

    features['dist_from_high_20d'] = 1.0
    features['dist_from_low_20d'] = 1.0
    features['dist_from_high_52w'] = 1.0
    features['dist_from_low_52w'] = 1.0

    mask_h20 = high_20d > 0
    mask_l20 = low_20d > 0
    mask_h52 = high_52w > 0
    mask_l52 = low_52w > 0

    features.loc[mask_h20, 'dist_from_high_20d'] = df.loc[mask_h20, 'Close'] / high_20d[mask_h20]
    features.loc[mask_l20, 'dist_from_low_20d'] = df.loc[mask_l20, 'Close'] / low_20d[mask_l20]
    features.loc[mask_h52, 'dist_from_high_52w'] = df.loc[mask_h52, 'Close'] / high_52w[mask_h52]
    features.loc[mask_l52, 'dist_from_low_52w'] = df.loc[mask_l52, 'Close'] / low_52w[mask_l52]

    # Trend strength
    features['trend_strength_10d'] = calculate_trend_strength(df['Close'], 10)
    features['trend_strength_20d'] = calculate_trend_strength(df['Close'], 20)
    features['trend_strength_60d'] = calculate_trend_strength(df['Close'], 60)

    # Pattern recognition
    features['higher_highs'] = (df['High'] > df['High'].shift(1)).astype(int).rolling(20).sum()
    features['lower_lows'] = (df['Low'] < df['Low'].shift(1)).astype(int).rolling(20).sum()

    # Final safety check
    for col in features.columns:
        if features[col].dtype in ['float64', 'float32']:
            features[col] = features[col].clip(lower=-1000, upper=1000)

    return features

def create_macro_features(df, macro_metadata, debug=False):
    """Create macro features - preserving all features"""
    features = pd.DataFrame(index=df.index)

    macro_cols = [c for c in df.columns if c.startswith('fred_')]

    if debug:
        logger.info(f"Creating macro features from {len(macro_cols)} columns...")

    if not macro_cols:
        if debug:
            logger.warning("No macro columns (fred_*) found in DataFrame!")
        return features

    # Process ALL macro columns without filtering
    for col in macro_cols:
        # Skip if too many NaN
        if df[col].notna().sum() < len(df) * 0.1:
            continue

        # Get the base data
        col_data = df[col].copy()

        # Fill forward then backward to handle NaN
        col_data = col_data.fillna(method='ffill').fillna(method='bfill')

        # Keep the raw feature
        features[col] = col_data

        # Normalized version (z-score)
        rolling_mean = col_data.rolling(252, min_periods=60).mean()
        rolling_std = col_data.rolling(252, min_periods=60).std()

        zscore_col = f'{col}_zscore'
        features[zscore_col] = 0.0

        valid_mask = (rolling_std > 1e-6) & rolling_std.notna() & rolling_mean.notna()
        if valid_mask.any():
            features.loc[valid_mask, zscore_col] = (
                (col_data[valid_mask] - rolling_mean[valid_mask]) / rolling_std[valid_mask]
            )

        features[zscore_col] = features[zscore_col].clip(-5, 5)

        # Rate of change
        features[f'{col}_roc_5d'] = col_data.pct_change(5).clip(-2, 2)
        features[f'{col}_roc_20d'] = col_data.pct_change(20).clip(-2, 2)

        # Trend
        trend_col = f'{col}_trend'
        trend_values = calculate_trend_strength(col_data, 60)
        features[trend_col] = trend_values.clip(-5, 5)

        # Momentum
        momentum_col = f'{col}_momentum'
        momentum_values = col_data - col_data.shift(20)

        col_scale = col_data.abs().rolling(60, min_periods=20).mean()
        features[momentum_col] = 0.0
        scale_mask = col_scale > 1e-6
        if scale_mask.any():
            features.loc[scale_mask, momentum_col] = (
                momentum_values[scale_mask] / col_scale[scale_mask]
            ).clip(-5, 5)

    # Final safety check
    for col in features.columns:
        features[col] = features[col].replace([np.inf, -np.inf], 0)
        features[col] = features[col].fillna(0)
        features[col] = features[col].clip(-10, 10)

    if debug:
        logger.info(f"Created {len(features.columns)} macro features")

    return features

def create_all_features(df, macro_metadata=None, use_cache=True):
    """Create all features: technical, proprietary, macro, regime, transformations, and interactions with caching"""
    import time
    start_time = time.time()
    
    logger.info("Creating comprehensive feature set with ALL proprietary features...")
    
    if use_cache:
        # Create horizon-agnostic cache key for cross-horizon reuse
        ticker_hash = hash(str(df.index.tolist() + df.columns.tolist()))
        data_hash = hash(str(df.values.tobytes()))
        macro_hash = hash(str(macro_metadata)) if macro_metadata else 0
        cache_key = f"features_cross_horizon_{ticker_hash}_{data_hash}_{macro_hash}"
        if cache_key in FEATURE_CACHE:
            logger.info(f"Using cached features (cross-horizon optimization) - saved {time.time() - start_time:.2f}s")
            return FEATURE_CACHE[cache_key].copy()

    # 1. Technical features
    tech_features = create_technical_features(df)
    logger.info(f"Created {len(tech_features.columns)} technical features")

    # 2. Proprietary features - FORCE CREATION OF ALL
    proprietary_features = create_proprietary_features(df)
    logger.info(f"Created/verified {len(proprietary_features.columns)} proprietary features")

    # 3. Macro features
    macro_features = create_macro_features(df, macro_metadata)
    logger.info(f"Created {len(macro_features.columns)} macro features")

    # 4. Regime features
    regime_features = create_regime_features(proprietary_features)
    logger.info(f"Created {len(regime_features.columns)} regime features")

    # 5. Non-linear transformations
    transformed_features = create_nonlinear_transformations(proprietary_features)
    logger.info(f"Created {len(transformed_features.columns)} non-linear transformations")

    # 6. Comprehensive interaction features
    interaction_features = create_comprehensive_interaction_features(
        macro_features, proprietary_features, regime_features
    )
    logger.info(f"Created {len(interaction_features.columns)} interaction features")

    # Combine all features
    all_features = pd.concat([
        tech_features,
        proprietary_features,
        macro_features,
        regime_features,
        transformed_features,
        interaction_features
    ], axis=1)

    # DO NOT remove any features based on variance - keep ALL proprietary features

    # Clean the features
    all_features = all_features.replace([np.inf, -np.inf], np.nan)

    # Cap extreme values
    for col in all_features.columns:
        if all_features[col].dtype in ['float64', 'float32']:
            valid_values = all_features[col].dropna()
            if len(valid_values) > 0:
                q001 = valid_values.quantile(0.001)
                q999 = valid_values.quantile(0.999)

                if np.isfinite(q001) and np.isfinite(q999):
                    all_features[col] = all_features[col].clip(lower=q001, upper=q999)

    # Enhanced missing data handling
    all_features = all_features.fillna(method='ffill', limit=5)
    all_features = all_features.fillna(method='bfill', limit=5)
    all_features = all_features.fillna(0)

    # Log feature type distribution
    feature_types = {
        'Technical': len([c for c in all_features.columns if c in tech_features.columns]),
        'Proprietary': len([c for c in all_features.columns if c in proprietary_features.columns]),
        'Macro': len([c for c in all_features.columns if c in macro_features.columns]),
        'Regime': len([c for c in all_features.columns if c in regime_features.columns]),
        'Transformed': len([c for c in all_features.columns if c in transformed_features.columns]),
        'Interaction': len([c for c in all_features.columns if c in interaction_features.columns])
    }

    logger.info(f"\nFeature distribution:")
    for feat_type, count in feature_types.items():
        logger.info(f"  {feat_type}: {count} features")
    logger.info(f"  TOTAL: {len(all_features.columns)} features")

    # Verify all proprietary features are present
    missing_proprietary = []
    for feat in CONFIG['PROPRIETARY_FEATURES']:
        if feat not in all_features.columns:
            missing_proprietary.append(feat)

    if missing_proprietary:
        logger.warning(f"Missing proprietary features in final set: {missing_proprietary}")
    else:
        logger.info("✓ All proprietary features successfully included!")

    # Cache the result for cross-horizon reuse
    if use_cache:
        FEATURE_CACHE[cache_key] = all_features.copy()
        logger.info(f"Cached features for cross-horizon reuse - total time: {time.time() - start_time:.2f}s")

    logger.info(f"Total features created: {len(all_features.columns)} in {time.time() - start_time:.2f}s")
    return all_features

# ==========================
# ENHANCED ML MODEL WITH SHAP
# ==========================

class EnhancedTradingModel:
    """Enhanced trading model with comprehensive SHAP analysis and multi-horizon support"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.feature_metadata = {}
        self.shap_explainers = {}
        self.training_diagnostics = {}
        self.performance_metrics = {}
        self.signal_statistics = defaultdict(lambda: defaultdict(int))
        self.per_stock_metrics = {}
        self.feature_importance_matrix = {}
        self.feature_presence_matrix = {}  # Track which features appear in top 5

    def prepare_training_data(self, df, features, prediction_days):
        """Prepare training data with proper future alignment"""
        # Calculate future returns
        future_price = df['Close'].shift(-prediction_days)
        current_price = df['Close']
        future_returns = (future_price - current_price) / current_price

        # Binary target: 1 for up, 0 for down
        target = (future_returns > 0).astype(int)

        # Store actual returns for performance calculation
        actual_returns = future_returns

        # Remove last rows without future data
        features = features.iloc[:-prediction_days]
        target = target.iloc[:-prediction_days]
        actual_returns = actual_returns.iloc[:-prediction_days]

        # Remove any NaN
        valid_idx = ~(features.isna().any(axis=1) | target.isna())

        X = features[valid_idx].copy()
        y = target[valid_idx].copy()
        returns = actual_returns[valid_idx].copy()

        # Check class balance
        if len(y) > 0:
            up_count = (y == 1).sum()
            down_count = (y == 0).sum()
            logger.info(f"Target balance for {prediction_days}d: UP={up_count} ({up_count/len(y)*100:.1f}%), DOWN={down_count} ({down_count/len(y)*100:.1f}%)")

        return X, y, returns

    def categorize_features(self, feature_names):
        """Categorize features by type"""
        categories = {
            'macro': [],
            'proprietary': [],
            'technical': [],
            'regime': [],
            'transformed': [],
            'interaction': []
        }

        for feat in feature_names:
            if '_X_' in feat:
                categories['interaction'].append(feat)
            elif feat in CONFIG['PROPRIETARY_FEATURES']:
                categories['proprietary'].append(feat)
            elif any(f'{prop}_' in feat for prop in CONFIG['PROPRIETARY_FEATURES']):
                if any(transform in feat for transform in ['_log', '_square', '_sqrt', '_rank', '_pct', '_zscore']):
                    categories['transformed'].append(feat)
                elif any(regime in feat for regime in ['_high', '_low', '_extreme', '_neutral']):
                    categories['regime'].append(feat)
                else:
                    categories['technical'].append(feat)
            elif 'fred_' in feat:
                categories['macro'].append(feat)
            else:
                categories['technical'].append(feat)

        return categories

    def get_top_features_by_type(self, feature_names, shap_values, n_per_type=5):
        """Get top N features from each category"""
        # Calculate absolute SHAP importance
        shap_importance = np.abs(shap_values).mean(axis=0) if len(shap_values.shape) > 1 else np.abs(shap_values)

        # Categorize features
        categories = self.categorize_features(feature_names)

        # Get top features by category
        top_features = {}

        for category, feat_list in categories.items():
            category_features = []
            for i, feat_name in enumerate(feature_names):
                if feat_name in feat_list:
                    category_features.append((feat_name, shap_importance[i], i))

            # Sort by importance
            category_features.sort(key=lambda x: x[1], reverse=True)
            top_features[category] = category_features[:n_per_type]

        # Also get overall top 5
        all_features = [(feat_name, shap_importance[i], i) for i, feat_name in enumerate(feature_names)]
        all_features.sort(key=lambda x: x[1], reverse=True)
        top_features['overall_top_5'] = all_features[:5]

        return top_features

    def calculate_per_stock_metrics(self, stock_data, prediction_days):
        """Calculate test metrics and feature importance for each individual stock"""
        if prediction_days not in self.models:
            return {}

        model = self.models[prediction_days]
        scaler = self.scalers[prediction_days]
        feature_cols = self.feature_columns[prediction_days]

        per_stock_metrics = {}
        per_stock_feature_importance = {}
        feature_presence = defaultdict(lambda: defaultdict(int))

        for ticker, df in stock_data.items():
            try:
                # Create features
                features = create_all_features(df, self.feature_metadata.get(prediction_days, {}), use_cache=True)

                # Prepare data
                X, y, returns = self.prepare_training_data(df, features, prediction_days)

                if len(X) < 50:  # Need sufficient test data
                    continue

                # Align features
                X_aligned = pd.DataFrame(index=X.index, columns=feature_cols)
                for col in feature_cols:
                    if col in X.columns:
                        X_aligned[col] = X[col]
                    else:
                        X_aligned[col] = 0

                X_aligned = X_aligned.fillna(0)

                # Split data for this stock (use last 20% as test)
                test_size = int(len(X_aligned) * 0.2)
                if test_size < 10:
                    continue

                X_test = X_aligned.iloc[-test_size:]
                y_test = y.iloc[-test_size:]
                returns_test = returns.iloc[-test_size:]

                # Calculate CAGR
                test_start_idx = len(df) - test_size - prediction_days
                test_end_idx = len(df) - prediction_days

                if test_start_idx >= 0 and test_end_idx < len(df):
                    starting_price = df['Close'].iloc[test_start_idx]
                    ending_price = df['Close'].iloc[test_end_idx]

                    if starting_price > 0 and ending_price > 0:
                        num_days = test_size
                        if num_days > 0:
                            total_return = ending_price / starting_price
                            years = num_days / 252.0
                            if years > 0:
                                cagr = (total_return ** (1 / years) - 1) * 100
                                cagr = max(-100, min(200, cagr))
                            else:
                                cagr = 0
                        else:
                            cagr = 0
                    else:
                        cagr = 0
                else:
                    cagr = 0

                # Scale and predict
                X_test_scaled = scaler.transform(X_test.values)
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                metrics = calculate_performance_metrics(y_test, y_pred, returns_test, prediction_days)
                metrics['cagr'] = cagr
                metrics['test_samples'] = len(y_test)

                per_stock_metrics[ticker] = metrics

                # Calculate SHAP values for this stock's test data
                if prediction_days in self.shap_explainers:
                    try:
                        # Get SHAP values for a sample of test data with caching
                        sample_size = min(5, len(X_test_scaled))  # Aggressively reduced from 10 to 5
                        X_sample = X_test_scaled[:sample_size]
                        
                        shap_key = f"shap_stock_{ticker}_{prediction_days}_{hash(str(X_sample.tobytes()))}"
                        
                        if shap_key in SHAP_CACHE:
                            shap_values = SHAP_CACHE[shap_key]
                        else:
                            shap_values = self.shap_explainers[prediction_days].shap_values(X_sample)

                            # Handle binary classification
                            if isinstance(shap_values, list) and len(shap_values) == 2:
                                shap_values = shap_values[1]
                            elif len(shap_values.shape) == 3:
                                shap_values = shap_values[:, :, 1]

                            SHAP_CACHE[shap_key] = shap_values

                        # Get feature importance by type
                        feature_importance = self.get_top_features_by_type(
                            feature_cols, shap_values, n_per_type=5
                        )
                        per_stock_feature_importance[ticker] = feature_importance

                        # Track feature presence in top 5
                        for feat_name, _, _ in feature_importance['overall_top_5']:
                            categories = self.categorize_features([feat_name])
                            for cat, feats in categories.items():
                                if feats:
                                    feature_presence[ticker][cat] += 1

                    except Exception as e:
                        logger.debug(f"Could not calculate SHAP for {ticker}: {e}")

            except Exception as e:
                logger.debug(f"Could not calculate metrics for {ticker}: {e}")
                continue

        # Store feature importance and presence matrices
        if per_stock_feature_importance:
            self.feature_importance_matrix[prediction_days] = per_stock_feature_importance
            self.feature_presence_matrix[prediction_days] = dict(feature_presence)

        return per_stock_metrics

    def train_model(self, stock_data, macro_metadata, prediction_days=30):
        """Train model with enhanced feature diversity"""
        logger.info(f"\nTraining Enhanced ML Model for {prediction_days}-day predictions...")

        all_X = []
        all_y = []
        all_returns = []
        all_tickers = []

        # Store macro metadata
        self.feature_metadata[prediction_days] = macro_metadata

        # Process each stock with pre-generated features
        for ticker, df in stock_data.items():
            if df.empty or len(df) < CONFIG['MIN_SAMPLES_PER_TICKER']:
                continue

            if hasattr(self, '_pregenerated_features') and ticker in self._pregenerated_features:
                features = self._pregenerated_features[ticker]
                logger.debug(f"Using pre-generated features for {ticker}")
            else:
                features = create_all_features(df, macro_metadata, use_cache=True)

            # Prepare training data
            X, y, returns = self.prepare_training_data(df, features, prediction_days)

            if len(X) >= CONFIG['MIN_SAMPLES_FOR_TRAINING']:
                all_X.append(X)
                all_y.append(y)
                all_returns.append(returns)
                all_tickers.extend([ticker] * len(X))

        if not all_X:
            logger.error("Insufficient data for training")
            return None

        # Combine all data
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        returns_combined = pd.concat(all_returns, ignore_index=True)

        logger.info(f"Combined data shape: {X_combined.shape}")

        # DIAGNOSTIC: Verify proprietary features are present
        logger.info(f"\n=== VERIFYING PROPRIETARY FEATURES in training data ===")
        prop_features_present = []
        prop_features_missing = []

        for feat in CONFIG['PROPRIETARY_FEATURES']:
            if feat in X_combined.columns:
                prop_features_present.append(feat)
                non_zero = (X_combined[feat] != 0).sum()
                unique_vals = X_combined[feat].nunique()
                mean_val = X_combined[feat].mean()
                logger.info(f"  ✓ {feat}: non-zero={non_zero}, unique={unique_vals}, mean={mean_val:.2f}")
            else:
                prop_features_missing.append(feat)
                logger.warning(f"  ✗ {feat}: MISSING from features!")

        logger.info(f"\nProprietary features present: {len(prop_features_present)}/{len(CONFIG['PROPRIETARY_FEATURES'])}")

        # Categorize all features
        feature_categories = self.categorize_features(X_combined.columns.tolist())

        logger.info(f"\n[Feature Distribution in Training Data]")
        for category, features in feature_categories.items():
            logger.info(f"  {category}: {len(features)} features")
        logger.info(f"  TOTAL: {len(X_combined.columns)} features")

        # Remove truly constant features (but keep low-variance proprietary features)
        constant_cols = []
        for col in X_combined.columns:
            if col not in CONFIG['PROPRIETARY_FEATURES'] and X_combined[col].nunique() <= 1:
                constant_cols.append(col)

        if constant_cols:
            X_combined = X_combined.drop(columns=constant_cols)
            logger.info(f"Removed {len(constant_cols)} constant features (excluding proprietary)")

        self.feature_columns[prediction_days] = X_combined.columns.tolist()

        # Unified scaling for all features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        self.scalers[prediction_days] = scaler

        # Split data
        X_train, X_test, y_train, y_test, returns_train, returns_test = train_test_split(
            X_scaled, y_combined, returns_combined,
            test_size=CONFIG['TEST_SIZE'],
            random_state=CONFIG['RANDOM_STATE'],
            stratify=y_combined
        )

        # Train Random Forest with updated parameters for better feature exploration
        rf_model = RandomForestClassifier(
            n_estimators=CONFIG['N_ESTIMATORS'],
            max_depth=CONFIG['MAX_DEPTH'],
            min_samples_split=CONFIG['MIN_SAMPLES_SPLIT'],
            min_samples_leaf=CONFIG['MIN_SAMPLES_LEAF'],
            max_features=CONFIG['MAX_FEATURES'],  # Use 50% of features at each split
            random_state=CONFIG['RANDOM_STATE'],
            n_jobs=-1,
            class_weight='balanced',
            max_samples=0.8
        )

        logger.info("Training Random Forest with enhanced parameters...")
        rf_model.fit(X_train, y_train)

        # Get predictions
        train_pred = rf_model.predict(X_train)
        test_pred = rf_model.predict(X_test)

        # Calculate performance metrics
        train_metrics = calculate_performance_metrics(y_train, train_pred, returns_train, prediction_days)
        test_metrics = calculate_performance_metrics(y_test, test_pred, returns_test, prediction_days)

        # Store metrics
        self.performance_metrics[prediction_days] = {
            'train': train_metrics,
            'test': test_metrics
        }

        # Print results
        logger.info(f"\nTraining Performance ({prediction_days}d):")
        for metric, value in train_metrics.items():
            logger.info(f"  {metric}: {value:.2f}")

        logger.info(f"\nTest Performance ({prediction_days}d):")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.2f}")

        # Initialize SHAP explainer with caching
        logger.info("\nInitializing SHAP explainer...")
        try:
            explainer_key = f"explainer_{prediction_days}_{hash(str(rf_model.get_params()))}"
            
            if explainer_key in EXPLAINER_CACHE:
                self.shap_explainers[prediction_days] = EXPLAINER_CACHE[explainer_key]
                logger.debug("Using cached SHAP explainer")
            else:
                self.shap_explainers[prediction_days] = shap.TreeExplainer(rf_model)
                EXPLAINER_CACHE[explainer_key] = self.shap_explainers[prediction_days]

            # Calculate sample SHAP values with batching
            sample_size = min(20, len(X_test))  # Aggressively reduced from 50 to 20
            X_test_sample = X_test[:sample_size]

            shap_key = f"shap_{prediction_days}_{hash(str(X_test_sample.tobytes()))}"
            
            if shap_key in SHAP_CACHE:
                shap_values = SHAP_CACHE[shap_key]
                logger.debug("Using cached SHAP values")
            else:
                # Get SHAP values with batch processing
                try:
                    shap_values = self.shap_explainers[prediction_days].shap_values(X_test_sample)
                except:
                    explanation = self.shap_explainers[prediction_days](X_test_sample)
                    shap_values = explanation.values

                # For binary classification, use positive class
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                elif len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]

                SHAP_CACHE[shap_key] = shap_values

            # Get feature importance by type
            feature_importance_by_type = self.get_top_features_by_type(
                self.feature_columns[prediction_days],
                shap_values,
                n_per_type=10
            )

            # DIAGNOSTIC: Print top features by type
            logger.info(f"\n=== TOP FEATURES BY TYPE for {prediction_days}d ===")

            for category in ['macro', 'proprietary', 'technical', 'interaction', 'regime', 'transformed']:
                logger.info(f"\nTop {category.upper()} Features:")
                if category in feature_importance_by_type:
                    for i, (feat_name, importance, idx) in enumerate(feature_importance_by_type[category][:5], 1):
                        logger.info(f"  {i}. {feat_name}: {importance:.4f}")
                    if not feature_importance_by_type[category]:
                        logger.warning(f"  ⚠️  NO {category} features in top rankings!")
                else:
                    logger.warning(f"  ⚠️  NO {category} features found!")

            # Check overall top 5
            logger.info(f"\nOVERALL TOP 5 FEATURES:")
            overall_categories = defaultdict(int)
            for feat_name, importance, idx in feature_importance_by_type['overall_top_5']:
                cats = self.categorize_features([feat_name])
                cat_name = next(iter([k for k, v in cats.items() if v]), 'unknown')
                overall_categories[cat_name] += 1
                logger.info(f"  [{cat_name.upper()}] {feat_name}: {importance:.4f}")

            # Warning if proprietary features are missing from top 5
            if overall_categories['proprietary'] == 0:
                logger.warning("\n⚠️  WARNING: No proprietary features in overall top 5!")

            # Calculate overall feature importance
            shap_importance = np.abs(shap_values).mean(axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns[prediction_days],
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)

            # Store diagnostics
            self.training_diagnostics[prediction_days] = {
                'feature_importance': feature_importance_df,
                'feature_importance_by_type': feature_importance_by_type,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'all_features': self.feature_columns[prediction_days],
                'feature_categories': feature_categories
            }

            logger.info("SHAP explainer initialized successfully")

        except Exception as e:
            logger.error(f"SHAP initialization failed: {e}")
            import traceback
            traceback.print_exc()

        self.models[prediction_days] = rf_model

        # Calculate per-stock metrics
        logger.info("\nCalculating per-stock test metrics and feature importance...")
        self.per_stock_metrics[prediction_days] = self.calculate_per_stock_metrics(stock_data, prediction_days)
        logger.info(f"Calculated metrics for {len(self.per_stock_metrics[prediction_days])} stocks")

        return rf_model

    def predict_proba(self, features, prediction_days):
        """Make prediction with proper feature alignment"""
        if prediction_days not in self.models:
            return None

        try:
            model = self.models[prediction_days]
            scaler = self.scalers[prediction_days]
            feature_cols = self.feature_columns[prediction_days]

            if not isinstance(features, pd.DataFrame):
                return None

            # Create a new DataFrame with all required features
            features_aligned = pd.DataFrame(index=features.index, columns=feature_cols)

            # Fill in available features
            for col in feature_cols:
                if col in features.columns:
                    features_aligned[col] = features[col]
                else:
                    features_aligned[col] = 0

            # Fill any remaining NaN
            features_aligned = features_aligned.fillna(0)

            # Convert to numpy array for scaling
            features_array = features_aligned.values

            # Scale features
            features_scaled = scaler.transform(features_array)

            # Get prediction
            proba = model.predict_proba(features_scaled)[0]
            return proba

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None

    def get_signal_shap_explanation(self, features, prediction_days):
        """Get enhanced SHAP explanation showing all feature types"""
        if prediction_days not in self.models or prediction_days not in self.shap_explainers:
            return None

        try:
            # Prepare features
            feature_cols = self.feature_columns[prediction_days]
            scaler = self.scalers[prediction_days]

            if not isinstance(features, pd.DataFrame):
                return None

            # Create aligned features DataFrame
            features_aligned = pd.DataFrame(index=features.index, columns=feature_cols)

            # Fill in available features
            for col in feature_cols:
                if col in features.columns:
                    features_aligned[col] = features[col]
                else:
                    features_aligned[col] = 0

            # Fill NaN and convert to array
            features_aligned = features_aligned.fillna(0)
            features_array = features_aligned.values

            if features_array.shape[0] != 1:
                features_array = features_array[0:1]

            # Scale features
            features_scaled = scaler.transform(features_array)

            # Get prediction
            proba = self.models[prediction_days].predict_proba(features_scaled)[0]

            # Get SHAP values with caching
            shap_key = f"shap_signal_{prediction_days}_{hash(str(features_scaled.tobytes()))}"
            
            if shap_key in SHAP_CACHE:
                shap_values = SHAP_CACHE[shap_key]
            else:
                try:
                    shap_values = self.shap_explainers[prediction_days].shap_values(features_scaled)
                except:
                    explanation = self.shap_explainers[prediction_days](features_scaled)
                    shap_values = explanation.values

                # Handle binary classification
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                elif len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]

                # Ensure we have a 1D array
                if shap_values.shape[0] == 1:
                    shap_values = shap_values[0]

                SHAP_CACHE[shap_key] = shap_values

            # Get top features by type
            top_features_by_type = self.get_top_features_by_type(
                feature_cols, shap_values, n_per_type=5
            )

            # Create explanation object
            explanation = {
                'prob_up': proba[1],
                'shap_values': shap_values,
                'feature_names': feature_cols,
                'feature_values': features_array[0],
                'top_features_by_type': top_features_by_type
            }

            return explanation

        except Exception as e:
            logger.error(f"Error getting SHAP explanation: {e}")
            return None

    def create_feature_presence_heatmap(self, output_path=None):
        """Create heatmap showing feature type presence in top 5 for each stock"""
        if not self.feature_presence_matrix:
            logger.warning("No feature presence data available for heatmap")
            return

        for horizon in CONFIG['HORIZONS']:
            if horizon not in self.feature_presence_matrix:
                continue

            presence_data = self.feature_presence_matrix[horizon]

            # Create presence matrix
            stocks = list(presence_data.keys())
            feature_types = ['macro', 'proprietary', 'technical', 'interaction', 'regime', 'transformed']

            presence_matrix = pd.DataFrame(
                index=stocks,
                columns=feature_types,
                data=0
            )

            for stock, type_counts in presence_data.items():
                for feat_type, count in type_counts.items():
                    if feat_type in presence_matrix.columns:
                        presence_matrix.loc[stock, feat_type] = count

            # Create heatmap
            plt.figure(figsize=(10, max(8, len(stocks) * 0.3)))

            # Create color map - 0 is red, >0 is green scale
            colors = ['red', 'yellow', 'lightgreen', 'green', 'darkgreen']
            n_colors = len(colors)
            cmap = plt.cm.colors.ListedColormap(colors)
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

            sns.heatmap(
                presence_matrix.astype(float),
                cmap=cmap,
                norm=norm,
                annot=True,
                fmt='g',
                cbar_kws={'label': 'Features in Top 5', 'ticks': [0, 1, 2, 3, 4]},
                linewidths=0.5,
                linecolor='gray'
            )

            plt.title(f'Feature Type Presence in Top 5 - {horizon}d Horizon\n(Red = 0 features, Green = multiple features)')
            plt.xlabel('Feature Type')
            plt.ylabel('Stock')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if output_path:
                filename = f"{output_path}_presence_{horizon}d.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Saved feature presence heatmap to {filename}")
            else:
                plt.show()

            plt.close()

            # Print summary statistics
            logger.info(f"\n=== Feature Type Presence Summary for {horizon}d ===")
            for feat_type in feature_types:
                stocks_with_type = (presence_matrix[feat_type] > 0).sum()
                avg_count = presence_matrix[feat_type].mean()
                logger.info(f"{feat_type}: {stocks_with_type}/{len(stocks)} stocks ({stocks_with_type/len(stocks)*100:.1f}%), avg {avg_count:.2f} features")

            # Identify stocks missing proprietary features
            stocks_missing_prop = presence_matrix[presence_matrix['proprietary'] == 0].index.tolist()
            if stocks_missing_prop:
                logger.warning(f"\nStocks with NO proprietary features in top 5: {stocks_missing_prop}")

# ==========================
# SIGNAL GENERATION
# ==========================

def generate_signals_with_shap(stock_data, ml_model, macro_metadata, timeframe=30):
    """Generate trading signals with comprehensive SHAP explanations"""
    signals = []

    # Track statistics
    feature_type_counts = defaultdict(int)
    successful_signals = 0
    failed_signals = 0

    # Calculate market metrics
    market_metrics = {}
    for ticker, df in stock_data.items():
        if len(df) < 60:
            continue

        try:
            current_price = float(df['Close'].iloc[-1])
            sma20_series = df['Close'].rolling(20).mean()
            sma20 = float(sma20_series.iloc[-1]) if not pd.isna(sma20_series.iloc[-1]) else current_price

            # Calculate momentum
            price_20d_ago = float(df['Close'].iloc[-20])
            momentum_20d = (current_price / price_20d_ago - 1) * 100 if price_20d_ago > 0 else 0.0

            # Calculate RSI
            rsi_series = calculate_rsi(df['Close'])
            rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

            market_metrics[ticker] = {
                'momentum_20d': momentum_20d,
                'rsi': rsi_val,
                'above_sma20': current_price > sma20,
                'df': df,
                'current_date': df['Date'].iloc[-1] if 'Date' in df.columns else pd.Timestamp.now()
            }
        except Exception as e:
            logger.error(f"Error calculating metrics for {ticker}: {e}")
            continue

    if not market_metrics:
        logger.warning("No valid market metrics calculated")
        return signals

    # Calculate market regime
    stocks_above_sma20 = sum(1 for m in market_metrics.values() if m['above_sma20'])
    market_breadth = stocks_above_sma20 / len(market_metrics)

    if market_breadth > 0.65:
        regime = 'BULL'
    elif market_breadth < 0.35:
        regime = 'BEAR'
    else:
        regime = 'NEUTRAL'

    logger.info(f"\nGenerating Signals for {timeframe}-day horizon")
    logger.info(f"  Market breadth: {market_breadth:.1%}")
    logger.info(f"  Market regime: {regime}")
    logger.info(f"  Processing {len(market_metrics)} stocks...")

    # Generate signal for each stock
    for ticker, metrics in market_metrics.items():
        df = metrics['df']
        signal_date = metrics['current_date']

        try:
            if hasattr(ml_model, '_pregenerated_features') and ticker in ml_model._pregenerated_features:
                features = ml_model._pregenerated_features[ticker]
                logger.debug(f"Using pre-generated features for {ticker} in signal generation")
            else:
                features = create_all_features(df, macro_metadata, use_cache=True)
            if features.empty or len(features) == 0:
                failed_signals += 1
                continue

            # Get the last row of features
            last_features = features.iloc[-1:].copy()

            # Get ML prediction
            proba = ml_model.predict_proba(last_features, timeframe)
            if proba is None:
                failed_signals += 1
                continue

            prob_up = float(proba[1])

            # Get SHAP explanation with all feature types
            shap_explanation = ml_model.get_signal_shap_explanation(last_features, timeframe)

            # Calculate ALL indicators
            indicators = calculate_indicators(df)

            # Process SHAP explanation
            shap_features = []
            shap_display = 'N/A'
            top_features_by_type = None
            feature_presence = defaultdict(int)
            driver_type = 'Unknown'

            if shap_explanation:
                try:
                    top_features_by_type = shap_explanation.get('top_features_by_type', {})

                    # Get top 5 overall features
                    overall_top_5 = top_features_by_type.get('overall_top_5', [])

                    # Format features for display
                    formatted_features = []

                    for feat_name, importance, idx in overall_top_5:
                        # Determine category
                        categories = ml_model.categorize_features([feat_name])
                        category = next(iter([k for k, v in categories.items() if v]), 'unknown')

                        # Track feature presence
                        feature_presence[category] += 1

                        feat_value = shap_explanation['feature_values'][idx]
                        shap_val = shap_explanation['shap_values'][idx]

                        formatted = format_shap_feature_complete(feat_name, shap_val, feat_value, category)
                        formatted_features.append(formatted)

                        shap_features.append({
                            'feature': feat_name,
                            'shap_value': float(shap_val),
                            'feature_type': category,
                            'actual_value': float(feat_value),
                            'rank': len(shap_features) + 1
                        })

                    shap_display = ' | '.join(formatted_features)

                    # Determine driver type based on feature presence
                    if feature_presence['proprietary'] >= 2:
                        driver_type = 'Proprietary-driven'
                    elif feature_presence['macro'] >= 2:
                        driver_type = 'Macro-driven'
                    elif feature_presence['interaction'] >= 2:
                        driver_type = 'Interaction-driven'
                    elif len(feature_presence) >= 3:
                        driver_type = 'Mixed-diverse'
                    else:
                        driver_type = list(feature_presence.keys())[0] + '-driven' if feature_presence else 'Unknown'

                    feature_type_counts[driver_type] += 1

                except Exception as e:
                    logger.error(f"Error processing SHAP for {ticker}: {e}")

            # Calculate combination-based confidence
            confidence = calculate_combination_confidence(prob_up, indicators, regime, top_features_by_type)

            # Get per-stock performance metrics
            if timeframe in ml_model.per_stock_metrics and ticker in ml_model.per_stock_metrics[timeframe]:
                stock_metrics = ml_model.per_stock_metrics[timeframe][ticker]
                accuracy = stock_metrics['accuracy']
                sharpe = stock_metrics['sharpe_ratio']
                win_rate = stock_metrics['win_rate']
                max_drawdown = stock_metrics['max_drawdown']
                cagr = stock_metrics.get('cagr', 0)
            else:
                # Fallback to overall model metrics
                if timeframe in ml_model.performance_metrics:
                    test_metrics = ml_model.performance_metrics[timeframe]['test']
                    accuracy = test_metrics['accuracy']
                    sharpe = test_metrics['sharpe_ratio']
                    win_rate = test_metrics['win_rate']
                    max_drawdown = test_metrics['max_drawdown']
                    cagr = 0
                else:
                    accuracy = 50.0
                    sharpe = 0.0
                    win_rate = 50.0
                    max_drawdown = 0.0
                    cagr = 0.0

            # Determine signal using combination logic
            signal = determine_combination_signal(prob_up, confidence, indicators, top_features_by_type)

            # Determine direction
            direction = 'UP' if prob_up > 0.5 else 'DOWN'

            # Build comprehensive signal data
            signal_data = {
                'ticker': ticker,
                'Stock': ticker,
                'horizon': f'{timeframe}d',
                'signal': signal,
                'Signal': signal,
                'direction': direction,
                'Price_Direction': direction.title(),
                'confidence': float(confidence),
                'Confidence': float(confidence),
                'prob_up': float(prob_up),
                'accuracy': float(accuracy),
                'Accuracy': float(accuracy),
                'sharpe_ratio': float(sharpe),
                'Sharpe': float(sharpe),
                'win_rate': float(win_rate),
                'max_drawdown': float(max_drawdown),
                'Drawdown': float(max_drawdown),
                'shap_top_5': shap_display,
                'SHAP': shap_display,
                'driver_type': driver_type,
                'indicators': indicators,
                'regime': regime,
                'signal_date': signal_date,
                'shap_features': shap_features,
                'top_features_by_type': top_features_by_type,
                'feature_presence': dict(feature_presence),

                # ALL fields including proprietary
                'CAGR': float(cagr),
                'VIX': float(indicators.get('VIX', 0)),
                'FNG': float(indicators.get('FNG', 0)),
                'RSI': float(indicators.get('RSI', 50)),
                'AnnVolatility': float(indicators.get('AnnVolatility', 30)),
                'Momentum125': float(indicators.get('Momentum125', 0)),
                'PriceStrength': float(indicators.get('PriceStrength', 0)),
                'VolumeBreadth': float(indicators.get('VolumeBreadth', 1)),
                'CallPut': float(indicators.get('CallPut', 50)),
                'NewsScore': float(indicators.get('NewsScore', 5)),
                'MACD': float(indicators.get('MACD', 0)),
                'BollingerBandWidth': float(indicators.get('BollingerBandWidth', 2)),
                'ATR': float(indicators.get('ATR', 1)),
                'StochRSI': float(indicators.get('StochRSI', 50)),
                'OBV': float(indicators.get('OBV', 0)),
                'CMF': float(indicators.get('CMF', 0)),
                'ADX': float(indicators.get('ADX', 25)),
                'Williams_R': float(indicators.get('Williams_R', -50)),
                'CCI': float(indicators.get('CCI', 0)),
                'MFI': float(indicators.get('MFI', 50)),

                # Price data
                'Current_Price': float(df['Close'].iloc[-1]),
                'SMA20': float(df['Close'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(df['Close'].iloc[-1]),
                'Vol_Breadth': float(indicators.get('VolumeBreadth', 1.0)),
                'BL20': 0.0,
                'BH20': 0.0,
            }

            # Calculate Bollinger Bands
            if len(df) >= 20:
                bb_data = calculate_bollinger_bands(df['Close'])
                signal_data['BL20'] = float(bb_data['lower'].iloc[-1])
                signal_data['BH20'] = float(bb_data['upper'].iloc[-1])

            # Create IF/THEN logic
            if_then_logic = create_if_then_logic_complete(
                ticker,
                timeframe,
                direction.title(),
                signal,
                accuracy,
                shap_features,
                indicators,
                sharpe,
                float(df['Close'].iloc[-1]),
                signal_data['BL20'],
                signal_data['BH20'],
                feature_presence
            )
            signal_data['IF_THEN'] = if_then_logic

            signals.append(signal_data)
            successful_signals += 1

            # Update model statistics
            ml_model.signal_statistics[timeframe]['total'] += 1
            ml_model.signal_statistics[timeframe][driver_type.lower()] += 1
            ml_model.signal_statistics[timeframe][signal.lower().replace(' ', '_')] += 1

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)[:100]}")
            failed_signals += 1
            continue

    # Print summary
    logger.info(f"\nSignal Generation Summary ({timeframe}d):")
    logger.info(f"  Successful: {successful_signals}")
    logger.info(f"  Failed: {failed_signals}")

    if len(signals) > 0:
        # Signal distribution
        signal_counts = defaultdict(int)
        for s in signals:
            signal_counts[s['signal']] += 1

        logger.info(f"\nSignal Distribution:")
        for signal_type, count in signal_counts.items():
            logger.info(f"  {signal_type}: {count} ({count/len(signals)*100:.1f}%)")

        # Driver type distribution
        logger.info(f"\nDriver Type Distribution:")
        total_counts = sum(feature_type_counts.values())
        if total_counts > 0:
            for driver_type, count in sorted(feature_type_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {driver_type}: {count} ({count/total_counts*100:.1f}%)")

        # Feature presence summary
        feature_presence_summary = defaultdict(int)
        for signal in signals:
            if 'feature_presence' in signal:
                for feat_type, count in signal['feature_presence'].items():
                    if count > 0:
                        feature_presence_summary[feat_type] += 1

        logger.info(f"\nFeature Types in Top 5 (across all stocks):")
        for feat_type, count in sorted(feature_presence_summary.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {feat_type}: {count}/{len(signals)} stocks ({count/len(signals)*100:.1f}%)")

    return signals

# ==========================
# DISPLAY FUNCTIONS
# ==========================

def get_performance_indicator(value: float, metric_type: str) -> str:
    """Return a colored emoji representing the value."""
    if pd.isna(value) or value is None:
        return ""

    if metric_type == "accuracy":
        return "🟢" if value >= 65 else "🟡" if value >= 55 else "🔴"
    elif metric_type == "sharpe":
        return "🟢" if value >= 1.0 else "🟡" if value >= 0.5 else "🔴"
    elif metric_type == "cagr":
        return "🟢" if value >= 20 else "🟡" if value >= 10 else "🔴"
    elif metric_type == "vix":
        if value == 0: return ""
        return "🔴" if value > 35 else "🟡" if value > 25 else "🟢"
    elif metric_type == "call_put":
        if value == 0: return ""
        return "🟢" if value > 60 else "🔴" if value < 40 else "🟡" if value > 55 or value < 45 else "⚪"
    elif metric_type == "fng":
        if value == 0: return ""
        return "🟢" if value > 75 else "🟢" if value > 60 else "🔴" if value < 25 else "🔴" if value < 40 else "⚪"
    elif metric_type == "news":
        if value == 0: return ""
        return "🟢" if value >= 7 else "🟡" if value >= 5 else "🔴"
    elif metric_type == "volatility":
        if value == 0: return ""
        return "🔴" if value > 40 else "🟡" if value > 30 else "🟢"
    elif metric_type == "momentum":
        return "🟢" if value > 20 else "🟡" if value > 10 else "🔴" if value < -10 else "⚪"
    return ""

def create_complete_playbook_tables(df, horizon):
    """Create complete tables showing ALL data including missing values"""
    if len(df) == 0:
        return

    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)

    print(f"\n{'='*150}")
    print(f"<span style='font-size:24px;font-weight:bold'>📊 {horizon}-DAY COMPLETE ANALYSIS</span>")
    print(f"{'='*150}")

    # Prepare COMPLETE data table with ALL proprietary features
    complete_rows = []
    for _, row in df.iterrows():
        # Get all proprietary values
        prop_values = {}
        for feat in CONFIG['PROPRIETARY_FEATURES']:
            value = row.get(feat, np.nan)
            prop_values[feat] = display_value(value)

        complete_rows.append({
            "Ticker": row['Stock'],
            "Signal": row['Signal'],
            "Accuracy": f"{get_performance_indicator(row['Accuracy'], 'accuracy')}{row['Accuracy']:.1f}%",
            "Sharpe": f"{get_performance_indicator(row['Sharpe'], 'sharpe')}{row['Sharpe']:.2f}",
            "CAGR": f"{get_performance_indicator(row['CAGR'], 'cagr')}{display_value(row['CAGR'])}%",
            "VIX": f"{get_performance_indicator(row['VIX'], 'vix')}{prop_values['VIX']}",
            "FNG": f"{get_performance_indicator(row['FNG'], 'fng')}{prop_values['FNG']}",
            "RSI": prop_values['RSI'],
            "AnnVol": f"{get_performance_indicator(row['AnnVolatility'], 'volatility')}{prop_values['AnnVolatility']}%",
            "Mom125": f"{get_performance_indicator(row['Momentum125'], 'momentum')}{prop_values['Momentum125']}%",
            "PriceStr": prop_values['PriceStrength'],
            "VolBreadth": prop_values['VolumeBreadth'],
            "MACD": prop_values['MACD'],
            "ATR": prop_values['ATR'],
            "ADX": prop_values['ADX'],
            "StochRSI": prop_values['StochRSI'],
            "CCI": prop_values['CCI'],
            "MFI": prop_values['MFI'],
            "News": prop_values['NewsScore'],
            "CallPut": prop_values['CallPut']
        })

    complete_df = pd.DataFrame(complete_rows)

    # Print COMPLETE proprietary features table
    print(f"\n📈 **{horizon}-DAY COMPLETE PROPRIETARY FEATURES TABLE**")
    print("```")
    print(complete_df.to_markdown(index=False))
    print("```")

    # Prepare SHAP analysis table
    shap_rows = []
    for _, row in df.iterrows():
        # Parse feature presence
        feature_presence = row.get('feature_presence', {})
        presence_str = ", ".join([f"{k}:{v}" for k, v in feature_presence.items() if v > 0])

        shap_rows.append({
            "Ticker": row['Stock'],
            "Top 5 SHAP Features": row.get('SHAP', 'N/A'),
            "Feature Types Present": presence_str or "None",
            "Driver": row.get('driver_type', 'Unknown')
        })

    shap_df = pd.DataFrame(shap_rows)

    # Print SHAP analysis table
    print(f"\n🔍 **{horizon}-DAY SHAP FEATURE ANALYSIS**")
    print("```")
    print(shap_df.to_markdown(index=False))
    print("```")

    # Summary statistics
    print(f"\n📊 **{horizon}-DAY SUMMARY STATISTICS**")

    # Calculate statistics for ALL proprietary features
    prop_stats = {}
    for feat in CONFIG['PROPRIETARY_FEATURES']:
        if feat in df.columns:
            values = df[feat].dropna()
            if len(values) > 0:
                prop_stats[feat] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'coverage': len(values) / len(df) * 100
                }

    # Print proprietary feature statistics
    print("\n**Proprietary Feature Statistics:**")
    print(f"{'Feature':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Coverage':>10}")
    print("-" * 70)

    for feat, stats in prop_stats.items():
        print(f"{feat:<15} {stats['mean']:>10.1f} {stats['std']:>10.1f} "
              f"{stats['min']:>10.1f} {stats['max']:>10.1f} {stats['coverage']:>9.1f}%")

    # Feature type presence in top 5
    print("\n**Feature Type Presence in Top 5 SHAP:**")
    type_counts = defaultdict(int)
    for _, row in df.iterrows():
        if 'feature_presence' in row:
            for feat_type, count in row['feature_presence'].items():
                if count > 0:
                    type_counts[feat_type] += 1

    for feat_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(df) * 100
        print(f"  {feat_type}: {count}/{len(df)} stocks ({pct:.1f}%)")

    # Warning if proprietary features are underrepresented
    if type_counts.get('proprietary', 0) < len(df) * 0.3:
        print("\n⚠️  WARNING: Proprietary features appear in less than 30% of top 5 SHAP features!")

    # IF/THEN examples
    print(f"\n💡 **{horizon}-DAY IF/THEN LOGIC EXAMPLES**")

    example_count = 0
    for _, row in df.iterrows():
        if example_count >= 5:
            break

        if_then = row.get('IF_THEN', '')
        if if_then and if_then != 'N/A':
            print(f"\n{if_then}")
            example_count += 1

def analyze_feature_diversity_complete(all_signals, ml_model):
    """Complete analysis of feature diversity with detailed breakdown"""

    print("\n" + "="*150)
    print("**COMPLETE FEATURE DIVERSITY ANALYSIS**")
    print("="*150)

    # Detailed tracking
    feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    feature_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    stocks_with_feature = defaultdict(lambda: defaultdict(set))
    total_stocks_per_horizon = defaultdict(int)

    # Track proprietary feature coverage
    proprietary_coverage = defaultdict(lambda: defaultdict(int))

    for signal in all_signals:
        horizon = int(signal['horizon'].replace('d', ''))
        ticker = signal['ticker']
        total_stocks_per_horizon[horizon] += 1

        # Track proprietary feature values
        for feat in CONFIG['PROPRIETARY_FEATURES']:
            if feat in signal and signal[feat] != 0:
                proprietary_coverage[horizon][feat] += 1

        # Track SHAP features
        if 'shap_features' in signal and signal['shap_features']:
            for feat_info in signal['shap_features']:
                feat_type = feat_info.get('feature_type', 'unknown')
                feat_name = feat_info['feature']

                feature_counts[horizon][feat_type][feat_name] += 1
                feature_values[horizon][feat_type][feat_name].append(abs(feat_info['shap_value']))
                stocks_with_feature[horizon][feat_type].add(ticker)

    # Print proprietary feature coverage
    print("\n📊 **PROPRIETARY FEATURE DATA COVERAGE**")
    print("\n{:<20} {:>15} {:>15} {:>15}".format("Feature", "30d Coverage", "45d Coverage", "60d Coverage"))
    print("-" * 70)

    for feat in CONFIG['PROPRIETARY_FEATURES']:
        coverages = []
        for h in CONFIG['HORIZONS']:
            count = proprietary_coverage[h].get(feat, 0)
            total = total_stocks_per_horizon[h]
            pct = (count / total * 100) if total > 0 else 0
            coverages.append(f"{count}/{total} ({pct:.1f}%)")

        print("{:<20} {:>15} {:>15} {:>15}".format(feat, *coverages))

    # Detailed feature type analysis
    print("\n📊 **FEATURE TYPE REPRESENTATION IN TOP 5 SHAP**")

    for feat_type in ['proprietary', 'macro', 'technical', 'interaction', 'regime', 'transformed']:
        print(f"\n--- {feat_type.upper()} Features ---")
        print("{:<40} {:>12} {:>12} {:>12}".format("Feature", "30d Count", "45d Count", "60d Count"))
        print("-" * 80)

        # Get all features of this type
        all_features = set()
        for horizon in CONFIG['HORIZONS']:
            all_features.update(feature_counts[horizon][feat_type].keys())

        if not all_features:
            print(f"  NO {feat_type} features found in top 5 SHAP!")
            continue

        # Sort by total frequency
        feature_totals = {}
        for feature in all_features:
            total = sum(feature_counts[h][feat_type].get(feature, 0) for h in CONFIG['HORIZONS'])
            feature_totals[feature] = total

        # Print top features
        for feature in sorted(all_features, key=lambda x: feature_totals[x], reverse=True)[:10]:
            counts = [feature_counts[h][feat_type].get(feature, 0) for h in CONFIG['HORIZONS']]
            print("{:<40} {:>12} {:>12} {:>12}".format(
                feature[:40], counts[0], counts[1], counts[2]
            ))

    # Stock-level analysis
    print("\n📊 **STOCK-LEVEL FEATURE TYPE COVERAGE**")

    for horizon in CONFIG['HORIZONS']:
        print(f"\n{horizon}-day horizon:")
        total_stocks = total_stocks_per_horizon[horizon]

        for feat_type in ['proprietary', 'macro', 'technical', 'interaction']:
            stocks_with_type = len(stocks_with_feature[horizon][feat_type])
            pct = (stocks_with_type / total_stocks * 100) if total_stocks > 0 else 0
            print(f"  {feat_type}: {stocks_with_type}/{total_stocks} stocks ({pct:.1f}%)")

    # Critical warnings
    print("\n⚠️  **CRITICAL WARNINGS**")

    warnings_found = False

    # Check proprietary feature representation
    for horizon in CONFIG['HORIZONS']:
        prop_stocks = len(stocks_with_feature[horizon]['proprietary'])
        total_stocks = total_stocks_per_horizon[horizon]

        if total_stocks > 0 and prop_stocks / total_stocks < 0.5:
            print(f"\n⚠️  {horizon}d: Only {prop_stocks}/{total_stocks} stocks "
                  f"({prop_stocks/total_stocks*100:.1f}%) have proprietary features in top 5!")
            warnings_found = True

    # Check if specific proprietary features are missing
    missing_proprietary = set()
    for feat in CONFIG['PROPRIETARY_FEATURES']:
        found = False
        for horizon in CONFIG['HORIZONS']:
            if any(feat in fname for fname in feature_counts[horizon]['proprietary'].keys()):
                found = True
                break
        if not found:
            missing_proprietary.add(feat)

    if missing_proprietary:
        print(f"\n⚠️  These proprietary features NEVER appear in top 5 SHAP: {missing_proprietary}")
        warnings_found = True

    if not warnings_found:
        print("\n✅ No critical warnings - feature diversity appears healthy")

    # Recommendations
    print("\n💡 **SPECIFIC RECOMMENDATIONS**")

    print("\n1. **To Improve Proprietary Feature Visibility:**")
    print("   - Increase interaction terms between low-visibility proprietary features and high-impact macro features")
    print("   - Add polynomial features for key proprietary indicators (VIX², FNG×RSI, etc.)")
    print("   - Consider feature engineering specific to regime changes")

    print("\n2. **Model Architecture Adjustments:**")
    print("   - Set min_samples_leaf lower (try 5) to capture proprietary feature nuances")
    print("   - Use feature_importances_ to pre-select diverse features")
    print("   - Consider ensemble with proprietary-focused and macro-focused models")

    print("\n3. **Data Quality Improvements:**")
    print("   - Ensure proprietary features have sufficient variation")
    print("   - Check for multicollinearity between similar indicators")
    print("   - Validate proprietary feature calculations")

# ==========================
# MAIN EXECUTION
# ==========================

def load_data():
    """Optimized data loading with caching and batch processing"""
    print("\n1. Loading stock data with ALL proprietary features...")
    
    # Initialize enhanced_fred_data to None for scope
    enhanced_fred_data = None
    
    # Get all CSV files from directory
    all_csv_files = []
    if os.path.exists(CONFIG['STOCK_DATA_PATH']):
        all_csv_files = [f for f in os.listdir(CONFIG['STOCK_DATA_PATH'])
                        if f.endswith('.csv') and not f.endswith('.gsheet.csv')]

    # Get tickers for ONLY 5 CSV files
    all_tickers = []
    for filename in all_csv_files[:CONFIG['MAX_STOCKS']]:  # Process only 5 stocks
        stock_id = filename.replace('.csv', '')
        if stock_id in STOCK_ID_TO_NAME:
            ticker = STOCK_ID_TO_NAME[stock_id]
        else:
            ticker = None
            for tick, ids in STOCK_ALTERNATIVE_NAMES.items():
                if stock_id == tick:
                    ticker = tick
                    break
            if not ticker:
                ticker = stock_id
        all_tickers.append(ticker)

    print(f"\nProcessing {len(all_tickers)} stocks: {', '.join(all_tickers)}")

    # Load stock data with caching
    stock_data = load_all_stock_data(all_tickers, use_cache=True)

    if not stock_data:
        logger.info("No stock data loaded with ticker-based approach, trying enhanced directory scanning...")
        enhanced_stock_data, enhanced_fred_data = load_all_stock_and_fred_data_enhanced(
            CONFIG['STOCK_DATA_PATH'], 
            CONFIG['FRED_ROOT_PATH'], 
            use_cache=True
        )
        if enhanced_stock_data:
            stock_data = enhanced_stock_data
            logger.info(f"✅ Enhanced loader found {len(stock_data)} stocks")
            if enhanced_fred_data:
                logger.info(f"✅ Enhanced loader found {len(enhanced_fred_data)} FRED indicators")

    if not stock_data:
        print("ERROR: No stock data loaded. Check file paths and stock IDs.")
        return None, None, None

    print(f"   Loaded {len(stock_data)} stocks")

    # Complete proprietary feature coverage analysis
    print("\n📊 PROPRIETARY FEATURE COVERAGE IN RAW DATA:")
    proprietary_feature_coverage = defaultdict(int)
    for ticker, df in stock_data.items():
        for feat in CONFIG['PROPRIETARY_FEATURES']:
            if feat in df.columns:
                proprietary_feature_coverage[feat] += 1

    print(f"\n{'Feature':<20} {'Coverage':>20} {'Percentage':>15}")
    print("-" * 60)
    for feat in CONFIG['PROPRIETARY_FEATURES']:
        count = proprietary_feature_coverage[feat]
        pct = count / len(stock_data) * 100
        status = "✅" if count > 0 else "❌"
        print(f"{feat:<20} {count}/{len(stock_data):>20} {pct:>14.1f}% {status}")

    # Load FRED data with caching - use enhanced_fred_data if available
    print("\n2. Loading FRED economic indicators...")
    if enhanced_fred_data:
        logger.info("Using enhanced FRED data from directory scanning...")
        fred_data_raw = enhanced_fred_data
        print(f"   Using {len(fred_data_raw)} enhanced FRED indicators")
    else:
        fred_data_raw = load_fred_data_from_folders(use_cache=True)

    if not fred_data_raw:
        print("WARNING: No FRED data loaded. Continuing with technical analysis only.")
        aligned_fred_data = {}
    else:
        print(f"   Loaded {len(fred_data_raw)} raw indicators")

        # Align FRED data
        print("\n3. Aligning FRED data with proper lags...")
        aligned_fred_data = fix_macro_data_alignment(fred_data_raw)
        print(f"   Created {len(aligned_fred_data)} aligned indicators")

    # Merge data
    print("\n4. Merging macro data with stock data...")
    merged_stock_data, macro_metadata = merge_macro_with_stock(stock_data, aligned_fred_data)

    return merged_stock_data, macro_metadata, stock_data

def generate_features(merged_stock_data, macro_metadata, use_cache=True):
    """Generate features once and cache for all horizons"""
    import time
    start_time = time.time()
    print("\nGenerating features with aggressive cross-horizon optimization...")
    
    features_by_stock = {}
    
    for ticker, df in merged_stock_data.items():
        # Generate features once per stock, reuse across horizons
        ticker_start = time.time()
        features = create_all_features(df, macro_metadata, use_cache=use_cache)
        features_by_stock[ticker] = features
        logger.info(f"Generated features for {ticker} in {time.time() - ticker_start:.2f}s (will reuse across horizons)")
        
    total_time = time.time() - start_time
    print(f"Generated features for {len(features_by_stock)} stocks in {total_time:.2f}s (cross-horizon cached)")
    return features_by_stock

def train_model(merged_stock_data, macro_metadata, horizons):
    """Aggressively optimized model training with batch processing"""
    import time
    start_time = time.time()
    print("\nTraining models with aggressive optimization...")
    
    # Initialize model
    ml_model = EnhancedTradingModel()
    
    # Store all signals
    all_signals = []

    # Train models and generate signals for each horizon
    for i, horizon in enumerate(horizons):
        horizon_start = time.time()
        print(f"\n{'='*60}")
        print(f"PROCESSING {horizon}-DAY HORIZON ({i+1}/{len(horizons)})")
        print(f"{'='*60}")

        # Train model
        print(f"\n5. Training ML model for {horizon}-day predictions...")
        model_start = time.time()
        model = ml_model.train_model(merged_stock_data, macro_metadata, horizon)
        print(f"   Model training completed in {time.time() - model_start:.2f}s")

        if model is None:
            print(f"ERROR: Model training failed for {horizon}-day horizon")
            continue

        # Generate signals
        print(f"\n6. Generating trading signals for {horizon}-day horizon...")
        signal_start = time.time()
        signals = generate_signals_with_shap(merged_stock_data, ml_model, macro_metadata, horizon)
        print(f"   Signal generation completed in {time.time() - signal_start:.2f}s")

        # Add to all signals
        all_signals.extend(signals)

        import gc
        del model
        gc.collect()
        horizon_time = time.time() - horizon_start
        logger.info(f"Completed {horizon}-day horizon in {horizon_time:.2f}s with memory cleanup")

    total_time = time.time() - start_time
    print(f"\nTotal model training completed in {total_time:.2f}s")
    return ml_model, all_signals

def run_shap(ml_model, all_signals, batch_size=50):
    """Optimized SHAP analysis with batching"""
    print("\n7. Analyzing complete feature diversity with SHAP optimization...")
    analyze_feature_diversity_complete(all_signals, ml_model)
    return all_signals

def format_outputs(all_signals, ml_model):
    """Optimized output formatting preserving playbook-style output"""
    print("\n8. Creating feature presence heatmaps...")
    if IN_COLAB:
        heatmap_prefix = '/content/drive/MyDrive/feature_presence'
    else:
        heatmap_prefix = 'feature_presence'
    ml_model.create_feature_presence_heatmap(heatmap_prefix)

    print("\n" + "="*150)
    print("**COMPREHENSIVE TRADING SIGNAL ANALYSIS WITH ALL FEATURES**")
    print("="*150)

    # Display complete results for each horizon
    for horizon in CONFIG['HORIZONS']:
        horizon_signals = [s for s in all_signals if s['horizon'] == f'{horizon}d']
        if horizon_signals:
            df_horizon = pd.DataFrame(horizon_signals)
            create_complete_playbook_tables(df_horizon, horizon)

    # Executive Summary
    print("\n" + "="*150)
    print("📊 **EXECUTIVE SUMMARY - COMPLETE ANALYSIS**")
    print("="*150)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Total Stocks Analyzed: {len(set([s['Stock'] for s in all_signals]))}")
    print(f"Total Signals Generated: {len(all_signals)}")

    # Filter strong signals
    strong_signals = [s for s in all_signals
                     if s['Accuracy'] >= 60 and s['Signal'] in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']]

    if strong_signals:
        print(f"\n📊 Strong Signals Summary:")
        print(f"- Total Strong Signals: {len(strong_signals)}")

        # Signals with proprietary features in top 5
        prop_signals = [s for s in strong_signals
                       if s.get('feature_presence', {}).get('proprietary', 0) > 0]
        print(f"- Signals with Proprietary Features in Top 5: {len(prop_signals)} "
              f"({len(prop_signals)/len(strong_signals)*100:.1f}%)")

        # Top opportunities
        buy_signals = [s for s in strong_signals if s['Signal'] in ['BUY', 'STRONG BUY']]
        sell_signals = [s for s in strong_signals if s['Signal'] in ['SELL', 'STRONG SELL']]

        if buy_signals:
            print("\n📈 **TOP BUY OPPORTUNITIES (sorted by VIX/Momentum combo):**")

            buy_df = pd.DataFrame(buy_signals)
            buy_df['combo_score'] = buy_df['Momentum125'] - buy_df['VIX'] + buy_df['FNG']/2
            top_buys = buy_df.nlargest(min(10, len(buy_df)), 'combo_score')

            print("\n{:<8} {:<8} {:<6} {:<7} {:<4} {:<4} {:<4} {:<7} {:<8} {:<15}".format(
                "Stock", "Horizon", "Acc%", "Sharpe", "VIX", "FNG", "RSI", "Mom125%", "AnnVol%", "Signal"
            ))
            print("-" * 100)

            for _, signal in top_buys.iterrows():
                signal_type = "Strong Buy" if "STRONG" in signal['Signal'] else "Buy"
                print(f"{signal['Stock']:<8} {signal['horizon']:<8} {signal['Accuracy']:>5.1f} "
                      f"{signal['Sharpe']:>7.2f} {display_value(signal['VIX']):>4} "
                      f"{display_value(signal['FNG']):>4} {display_value(signal['RSI']):>4} "
                      f"{display_value(signal['Momentum125']):>7} "
                      f"{display_value(signal['AnnVolatility']):>8} {signal_type:<15}")

    # Save results
    if IN_COLAB:
        output_prefix = '/content/drive/MyDrive/complete_proprietary_signals'
    else:
        output_prefix = 'complete_proprietary_signals'

    save_complete_analysis_results(all_signals, ml_model, output_prefix)

    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    # Print complete legend
    print("\n" + "="*60)
    print("📚 **COMPLETE LEGEND AND EXPLANATIONS**")
    print("="*60)

    print("\n**Signal Types:**")
    print("📈 = Buy/Up  📉 = Sell/Down")
    print("STRONG BUY/SELL = High confidence signal with multiple confirmations")
    print("BUY/SELL = Standard signal meeting criteria")
    print("NEUTRAL = Insufficient confidence or conflicting signals")

    print("\n**Performance Indicators:**")
    print("🟢 = Good  🟡 = Moderate  🔴 = Poor  ⚪ = Neutral")

    print("\n**Metric Thresholds:**")
    print("Accuracy: 🟢 ≥65%  🟡 55-65%  🔴 <55%")
    print("Sharpe: 🟢 ≥1.0  🟡 0.5-1.0  🔴 <0.5")
    print("CAGR: 🟢 ≥20%  🟡 10-20%  🔴 <10%")

    print("\n**Proprietary Indicators:**")
    print("VIX: 🟢 <15  🟡 15-30  🔴 >30 (Market volatility)")
    print("FNG: 🔴 <25 (ExFear)  🔴 25-40 (Fear)  ⚪ 40-60 (Neutral)  🟢 60-75 (Greed)  🟢 >75 (ExGreed)")
    print("RSI: <30 = Oversold, 30-70 = Normal, >70 = Overbought")
    print("Momentum125: 🔴 <-10%  ⚪ -10% to 10%  🟡 10-20%  🟢 >20%")
    print("AnnVolatility: 🟢 <20%  🟡 20-40%  🔴 >40%")

    print("\n**Feature Type Codes:**")
    print("[M] = Macro  [P] = Proprietary  [T] = Technical")
    print("[I] = Interaction  [R] = Regime  [X] = Transformed")

    print("\n**Missing Data:**")
    print("— = Data not available or could not be calculated")

    print("\n" + "-"*60)
    print("End of Analysis")
    print("-"*60)

def main():
    """Main execution function with aggressive optimization"""
    import time
    total_start = time.time()
    global IN_COLAB
    
    print("🚀 ENHANCED TRADING MODEL - AGGRESSIVELY OPTIMIZED EXECUTION")
    print("="*60)

    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab")

        # Mount Google Drive if not already mounted
        from google.colab import drive
        import os
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
    except:
        IN_COLAB = False
        print("Not running in Google Colab")

    import warnings
    warnings.filterwarnings('ignore')

    try:
        print("\n1. Loading and processing data...")
        data_start = time.time()
        merged_stock_data, macro_metadata, stock_data = load_data()
        if merged_stock_data is None:
            return
        print(f"   Data loading completed in {time.time() - data_start:.2f}s")
        
        print("\n2. Pre-generating features for cross-horizon reuse...")
        feature_start = time.time()
        features_by_stock = generate_features(merged_stock_data, macro_metadata)
        print(f"   Feature generation completed in {time.time() - feature_start:.2f}s")
        
        # 3. Train models and generate signals with pre-generated features
        print("\n3. Training models with pre-generated features...")
        train_start = time.time()
        ml_model, all_signals = train_model(merged_stock_data, macro_metadata, CONFIG['HORIZONS'])
        
        ml_model._pregenerated_features = features_by_stock
        print(f"   Model training completed in {time.time() - train_start:.2f}s")

        all_signals = run_shap(ml_model, all_signals)

        print("\n4. Formatting and displaying results...")
        output_start = time.time()
        format_outputs(all_signals, ml_model)
        print(f"   Output formatting completed in {time.time() - output_start:.2f}s")
        
        total_time = time.time() - total_start
        print(f"\n✅ ENHANCED TRADING MODEL EXECUTION COMPLETE IN {total_time:.2f}s!")
        print("="*60)
        
        if total_time < 120:  # 2 minutes
            print(f"🎯 SUCCESS: Execution time {total_time:.2f}s is under 2-minute target!")
        else:
            print(f"⚠️  WARNING: Execution time {total_time:.2f}s exceeds 2-minute target")
        
        gc.collect()

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

def save_complete_analysis_results(signals, ml_model, output_prefix):
    """Save complete analysis results with all features"""
    if not signals:
        logger.warning("No signals to save")
        return

    # Convert to DataFrame
    df = pd.DataFrame(signals)

    # Save complete results to CSV
    csv_filename = f"{output_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_filename, index=False)
    logger.info(f"Saved complete results to {csv_filename}")

    # Save enhanced summary to JSON
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_signals': len(signals),
        'horizons': list(set([s['horizon'] for s in signals])),
        'stocks_analyzed': list(set([s['ticker'] for s in signals])),
        'signal_distribution': dict(pd.Series([s['signal'] for s in signals]).value_counts()),
        'driver_distribution': dict(pd.Series([s['driver_type'] for s in signals]).value_counts()),
        'proprietary_feature_statistics': {},
        'feature_type_coverage': {}
    }

    # Add proprietary feature statistics
    for feat in CONFIG['PROPRIETARY_FEATURES']:
        if feat in df.columns:
            values = df[feat].dropna()
            if len(values) > 0:
                summary['proprietary_feature_statistics'][feat] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'coverage': float(len(values) / len(df) * 100)
                }

    # Add feature type coverage
    feature_type_counts = defaultdict(int)
    for signal in signals:
        if 'feature_presence' in signal:
            for feat_type, count in signal['feature_presence'].items():
                if count > 0:
                    feature_type_counts[feat_type] += 1

    for feat_type, count in feature_type_counts.items():
        summary['feature_type_coverage'][feat_type] = {
            'count': count,
            'percentage': float(count / len(signals) * 100)
        }

    json_filename = f"{output_prefix}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, 'w') as f:
        json.dump(summary, f, indent=2, default=convert_np)
    logger.info(f"Saved complete summary to {json_filename}")

def calculate_indicators(stock_df):
    """Calculate ALL technical indicators for signal generation"""
    indicators = {}

    try:
        # Get ALL proprietary features - either from data or calculate
        prop_features = create_proprietary_features(stock_df)

        # Add all proprietary features to indicators
        for feat in CONFIG['PROPRIETARY_FEATURES']:
            if feat in prop_features.columns:
                value = prop_features[feat].iloc[-1]
                indicators[feat] = float(value) if not pd.isna(value) else 0.0
            else:
                indicators[feat] = 0.0

        # Add any additional calculated indicators
        current_price = float(stock_df['Close'].iloc[-1])

        # Moving averages
        if len(stock_df) >= 50:
            sma50 = stock_df['Close'].rolling(window=50).mean().iloc[-1]
            indicators['price_sma50'] = float(current_price / sma50) if sma50 > 0 and not pd.isna(sma50) else 1.0
        else:
            indicators['price_sma50'] = 1.0

        if len(stock_df) >= 200:
            sma200 = stock_df['Close'].rolling(window=200).mean().iloc[-1]
            indicators['price_sma200'] = float(current_price / sma200) if sma200 > 0 and not pd.isna(sma200) else 1.0
        else:
            indicators['price_sma200'] = 1.0

        # Momentum
        if len(stock_df) >= 20:
            price_20d_ago = stock_df['Close'].iloc[-20]
            indicators['momentum_20d'] = float((current_price / price_20d_ago - 1)) if price_20d_ago > 0 else 0.0
        else:
            indicators['momentum_20d'] = 0.0

        # Bollinger Bands
        bb_data = calculate_bollinger_bands(stock_df['Close'])
        bb_percent_val = bb_data['percent_b'].iloc[-1]
        indicators['bb_percent'] = float(bb_percent_val) if not pd.isna(bb_percent_val) else 0.5

        # MACD components
        macd_data = calculate_macd(stock_df['Close'])
        indicators['macd_histogram'] = float(macd_data['histogram'].iloc[-1]) if not pd.isna(macd_data['histogram'].iloc[-1]) else 0.0
        indicators['macd_bullish_cross'] = bool(macd_data['bullish_cross'])
        indicators['macd_bearish_cross'] = bool(macd_data['bearish_cross'])

        # Ensure all values are Python scalars
        for key, value in indicators.items():
            if hasattr(value, 'item'):
                indicators[key] = value.item()
            elif isinstance(value, (np.ndarray, pd.Series)):
                if len(value) > 0:
                    indicators[key] = float(value[0]) if not isinstance(value[0], bool) else bool(value[0])
                else:
                    indicators[key] = 0.0 if key not in ['macd_bullish_cross', 'macd_bearish_cross'] else False

        logger.debug(f"Calculated {len(indicators)} indicators including all proprietary features")

        return indicators

    except Exception as e:
        logger.error(f"Error in calculate_indicators: {str(e)}")
        # Return default values for all proprietary features
        default_indicators = {feat: 0.0 for feat in CONFIG['PROPRIETARY_FEATURES']}
        default_indicators.update({
            'price_sma50': 1.0, 'price_sma200': 1.0, 'momentum_20d': 0.0,
            'bb_percent': 0.5, 'macd_histogram': 0.0,
            'macd_bullish_cross': False, 'macd_bearish_cross': False
        })
        return default_indicators

def calculate_combination_confidence(prob_up, indicators, regime, top_features_by_type):
    """Calculate confidence based on feature combinations"""
    confidence = 40.0

    # 1. Model probability contribution (20 points max)
    prob_up = ensure_scalar(prob_up)
    prob_confidence = abs(prob_up - 0.5) * 40
    confidence += prob_confidence * 0.5

    # 2. Feature diversity scoring (40 points max)
    diversity_score = 0

    if top_features_by_type:
        # Check how many different feature types are in top 5
        feature_types_present = set()
        for category in ['macro', 'proprietary', 'technical', 'interaction']:
            if category in top_features_by_type and top_features_by_type[category]:
                feature_types_present.add(category)

        # More diverse features = higher confidence
        diversity_score = len(feature_types_present) * 10
        confidence += diversity_score

    # 3. Critical combinations (40 points max)
    combination_score = 0
    combination_count = 0

    # VIX-based combinations
    vix = indicators.get('VIX', 20)
    if vix > 30:
        if prob_up < 0.45:
            combination_score += 3  # High VIX + bearish prediction
        else:
            combination_score -= 1  # High VIX + bullish prediction (contrarian)
        combination_count += 1
    elif vix < 15:
        if prob_up > 0.55:
            combination_score += 3  # Low VIX + bullish prediction
        combination_count += 1

    # FNG-based combinations
    fng = indicators.get('FNG', 50)
    if fng < 25:
        if indicators.get('RSI', 50) < 30:
            combination_score += 2  # Fear + oversold
        combination_count += 1
    elif fng > 75:
        if indicators.get('RSI', 50) > 70:
            combination_score += 2  # Greed + overbought
        combination_count += 1

    # Momentum combinations
    momentum = indicators.get('Momentum125', 0)
    if momentum > 30:
        if indicators.get('AnnVolatility', 30) < 25:
            combination_score += 2  # Strong momentum + low volatility
        combination_count += 1
    elif momentum < -30:
        if vix > 35:
            combination_score += 2  # Negative momentum + high fear
        combination_count += 1

    if combination_count > 0:
        confidence += (combination_score / combination_count) * 40

    # Cap confidence
    confidence = max(20, min(90, confidence))

    return confidence

def determine_combination_signal(prob_up, confidence, indicators, top_features_by_type):
    """Determine signal based on feature combinations and diversity"""
    # Define thresholds
    BUY_THRESHOLD = 0.55
    SELL_THRESHOLD = 0.45
    MIN_CONFIDENCE = 50

    # Check confidence first
    if confidence < MIN_CONFIDENCE:
        return 'NEUTRAL'

    # Extract key indicators
    vix = indicators.get('VIX', 20)
    fng = indicators.get('FNG', 50)
    rsi = indicators.get('RSI', 50)
    momentum = indicators.get('Momentum125', 0)
    volatility = indicators.get('AnnVolatility', 30)
    price_strength = indicators.get('PriceStrength', 0)
    volume_breadth = indicators.get('VolumeBreadth', 1)

    # Check feature diversity
    has_proprietary = False
    if top_features_by_type and 'proprietary' in top_features_by_type:
        has_proprietary = len(top_features_by_type['proprietary']) > 0

    # Strong combination signals
    # 1. Extreme VIX conditions
    if vix > 40 and prob_up < 0.4 and rsi > 60:
        return 'STRONG SELL'
    elif vix < 12 and momentum > 40 and prob_up > 0.65:
        return 'STRONG BUY'

    # 2. Fear/Greed extremes
    elif fng < 20 and rsi < 30 and prob_up > 0.55:
        return 'BUY'  # Contrarian buy
    elif fng > 85 and rsi > 70 and prob_up < 0.45:
        return 'SELL'  # Contrarian sell

    # 3. Momentum + Volatility combinations
    elif momentum > 30 and volatility < 20 and prob_up > 0.6:
        return 'STRONG BUY'
    elif momentum < -30 and volatility > 40 and prob_up < 0.4:
        return 'STRONG SELL'

    # 4. Volume breadth signals
    elif volume_breadth > 2.0 and price_strength > 50 and prob_up > 0.55:
        return 'BUY'
    elif volume_breadth < 0.5 and price_strength < -25 and prob_up < 0.45:
        return 'SELL'

    # Regular signals with feature diversity check
    elif prob_up > BUY_THRESHOLD and confidence > 60:
        if has_proprietary or confidence > 70:
            # Check supporting conditions
            bullish_conditions = 0
            if vix < 25: bullish_conditions += 1
            if momentum > 10: bullish_conditions += 1
            if rsi < 70: bullish_conditions += 1
            if volatility < 35: bullish_conditions += 1
            if volume_breadth > 1.0: bullish_conditions += 1

            if bullish_conditions >= 3:
                return 'STRONG BUY'
            elif bullish_conditions >= 2:
                return 'BUY'
            else:
                return 'NEUTRAL'
        else:
            return 'NEUTRAL'  # Need proprietary features for buy signal

    elif prob_up < SELL_THRESHOLD and confidence > 60:
        if has_proprietary or confidence > 70:
            # Check supporting conditions
            bearish_conditions = 0
            if vix > 25: bearish_conditions += 1
            if momentum < -10: bearish_conditions += 1
            if rsi > 30: bearish_conditions += 1
            if volatility > 35: bearish_conditions += 1
            if volume_breadth < 1.0: bearish_conditions += 1

            if bearish_conditions >= 3:
                return 'STRONG SELL'
            elif bearish_conditions >= 2:
                return 'SELL'
            else:
                return 'NEUTRAL'
        else:
            return 'NEUTRAL'  # Need proprietary features for sell signal

    # Default to neutral
    else:
        return 'NEUTRAL'

def format_shap_feature_complete(feat_name, shap_val, feat_value, category):
    """Format SHAP feature with complete information"""
    # Determine display name based on category and feature
    if category == 'macro':
        base_name = feat_name.replace('fred_', '').split('_')[0].upper()
        if base_name in FRED_METADATA:
            display_name = FRED_METADATA[base_name]['name']
            if len(display_name) > 20:
                display_name = base_name
        else:
            display_name = base_name
    elif category == 'proprietary':
        display_name = feat_name
    elif category == 'interaction':
        parts = feat_name.split('_X_')
        if len(parts) >= 2:
            macro_part = parts[0].replace('fred_', '').split('_')[0]
            other_part = parts[1].split('_')[0]
            display_name = f"{macro_part}×{other_part}"
        else:
            display_name = feat_name[:20]
    elif category == 'transformed':
        base_feat = feat_name.split('_')[0]
        transform = feat_name.split('_')[-1]
        display_name = f"{base_feat}_{transform}"
    elif category == 'regime':
        display_name = feat_name
    else:  # technical
        if 'sma' in feat_name:
            display_name = feat_name.upper()
        elif 'returns' in feat_name:
            period = feat_name.split('_')[-1]
            display_name = f"Ret_{period}"
        else:
            display_name = feat_name

    # Format value
    if isinstance(feat_value, (int, float)):
        if abs(feat_value) >= 1000:
            value_str = f"{feat_value:.0f}"
        elif abs(feat_value) >= 10:
            value_str = f"{feat_value:.1f}"
        elif abs(feat_value) >= 1:
            value_str = f"{feat_value:.2f}"
        else:
            value_str = f"{feat_value:.3f}"
    else:
        value_str = str(feat_value)[:10]

    # Direction
    direction = "↑" if shap_val > 0 else "↓"

    # Category abbreviation
    cat_abbrev = {
        'macro': 'M',
        'proprietary': 'P',
        'technical': 'T',
        'interaction': 'I',
        'regime': 'R',
        'transformed': 'X'
    }.get(category, 'U')

    return f"[{cat_abbrev}] {display_name}={value_str} ({shap_val:+.3f}{direction})"

def label_fng(val: int) -> str:
    """Get textual label for FNG"""
    if val >= 85:
        return "ExGreed"
    elif val >= 75:
        return "Greed"
    elif val >= 60:
        return "Greed+"
    elif val >= 40:
        return "Neutral"
    elif val >= 25:
        return "Fear"
    elif val >= 15:
        return "Fear+"
    else:
        return "ExFear"

def create_if_then_logic_complete(stock_name, horizon, direction, signal, accuracy,
                                 shap_features, indicators, sharpe, current_price,
                                 bl20, bh20, feature_presence):
    """Create IF/THEN logic with complete feature information"""

    # Initialize the logic components
    if_conditions = []

    # Add top SHAP features with their types
    feature_conditions = []
    for feat in shap_features[:3]:  # Top 3 features
        feat_name = feat['feature']
        shap_val = feat['shap_value']
        actual_val = feat.get('actual_value', 0)
        feat_type = feat.get('feature_type', 'unknown')

        # Format based on feature type
        type_label = feat_type[0].upper()  # First letter of type

        if feat_type == 'proprietary' and feat_name in ['VIX', 'FNG', 'RSI', 'Momentum125']:
            if feat_name == 'VIX':
                condition = f"{feat_name}={actual_val:.0f}"
            elif feat_name == 'FNG':
                condition = f"{feat_name}={actual_val:.0f}({label_fng(actual_val)})"
            elif feat_name == 'Momentum125':
                condition = f"Mom125={actual_val:.0f}%"
            else:
                condition = f"{feat_name}={actual_val:.1f}"
        else:
            condition = f"{feat_name[:15]}={display_value(actual_val)}"

        feature_conditions.append(f"[{type_label}]{condition}({shap_val:+.2f})")

    if feature_conditions:
        if_conditions.append("Features: " + ", ".join(feature_conditions))

    # Add critical combinations
    critical_combos = []

    vix = indicators.get('VIX', 20)
    fng = indicators.get('FNG', 50)
    rsi = indicators.get('RSI', 50)
    momentum = indicators.get('Momentum125', 0)
    volatility = indicators.get('AnnVolatility', 30)

    # VIX-based combinations
    if vix > 30:
        if rsi > 70:
            critical_combos.append(f"VIX>{30} & RSI>{70}")
        elif momentum < -20:
            critical_combos.append(f"VIX>{30} & Mom<-20%")
    elif vix < 15:
        if momentum > 20:
            critical_combos.append(f"VIX<{15} & Mom>{20}%")

    # FNG-based combinations
    if fng < 25 and rsi < 30:
        critical_combos.append(f"Fear({fng}) & RSI<{30}")
    elif fng > 75 and rsi > 70:
        critical_combos.append(f"Greed({fng}) & RSI>{70}")

    if critical_combos:
        if_conditions.append("Combos: " + " | ".join(critical_combos))

    # Add feature diversity information
    if feature_presence:
        diversity_str = "Types: " + ", ".join([f"{k}({v})" for k, v in feature_presence.items() if v > 0])
        if_conditions.append(diversity_str)

    # Create the IF/THEN statement
    if_part = " AND ".join(if_conditions) if if_conditions else "No significant features"

    # Determine confidence level
    if accuracy >= 70 and len(feature_presence) >= 3:
        confidence_text = f"{accuracy:.0f}% very high confidence (diverse signals)"
    elif accuracy >= 65:
        confidence_text = f"{accuracy:.0f}% high confidence"
    elif accuracy >= 55:
        confidence_text = f"{accuracy:.0f}% moderate confidence"
    else:
        confidence_text = f"{accuracy:.0f}% low confidence"

    # Build the complete logic string
    logic = f"{stock_name} ({horizon}d) {'↑' if direction == 'Up' else '↓'} "
    logic += f"IF {if_part} "
    logic += f"THEN {signal} ({confidence_text}). "

    # Add signal quality indicator
    has_proprietary = feature_presence.get('proprietary', 0) > 0
    has_macro = feature_presence.get('macro', 0) > 0
    has_interaction = feature_presence.get('interaction', 0) > 0

    if has_proprietary and (has_macro or has_interaction):
        logic += "High-quality mixed signal. ✅✅"
    elif has_proprietary:
        logic += "Proprietary-driven signal. ✅"
    elif len(feature_presence) >= 3:
        logic += "Diverse feature signal. ✅"
    else:
        logic += "Limited feature diversity. ⚠️"

    return logic

# Run main if executed directly
if __name__ == "__main__":
    main()
