import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List


def sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period, min_periods=period).mean()


def ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, min_periods=period).mean()


def wma(prices: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return prices.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def bollinger_bands(price: pd.Series, 
                   period: int = 20, 
                   std_dev: float = 2.0) -> Dict[str, pd.Series]:
    middle = price.rolling(window=period, min_periods=period).mean()
    std = price.rolling(window=period, min_periods=period).std()
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    width = upper - lower
    
    middle_safe = middle.replace(0, np.nan)
    width_pct = width / middle_safe * 100
    
    return {
        'BB_UPPER': upper.rename('BB_UPPER'),
        'BB_MIDDLE': middle.rename('BB_MIDDLE'),
        'BB_LOWER': lower.rename('BB_LOWER'),
        'BB_WIDTH': width.rename('BB_WIDTH'),
        'BB_WIDTH_PCT': width_pct.rename('BB_WIDTH_PCT')
    }


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
        signal: int = 9) -> Dict[str, pd.Series]:
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    return prices - prices.shift(period)


def roc(prices: pd.Series, period: int = 10) -> pd.Series:
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def stddev(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period, min_periods=period).std()


def mabw(prices: pd.Series, fast_period: int = 10, slow_period: int = 50,
        multiplier: float = 1.0) -> Dict[str, pd.Series]:
    ma_slow = ema(prices, slow_period)
    ma_fast = ema(prices, fast_period)
    
    dst = ma_slow - ma_fast
    dv = dst.pow(2).rolling(fast_period, min_periods=fast_period).mean().apply(
        lambda x: x ** 0.5 if not pd.isna(x) else np.nan
    )
    dev = dv * multiplier
    
    upper = ma_slow + dev
    lower = ma_slow - dev
    width = (upper - lower) / ma_slow * 100
    
    llv = width.rolling(slow_period, min_periods=slow_period).min()
    
    return {
        'upper': upper,
        'middle': ma_slow,
        'lower': lower,
        'width': width,
        'llv': llv
    }


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df['High'] + df['Low'] + df['Close']) / 3


def vpn(df: pd.DataFrame, period: int = 30, smooth: int = 3, 
        ma_period: int = 30) -> Dict[str, pd.Series]:
    tp = typical_price(df)
    tp_prev = tp.shift(1)
    
    atr_val = atr(df['High'], df['Low'], df['Close'], period)
    threshold = 0.1 * atr_val
    
    volume = df['Volume']
    mav = volume.rolling(window=period, min_periods=period).mean()
    
    price_change = tp - tp_prev
    positive_cond = price_change > threshold
    negative_cond = price_change < -threshold
    
    vp = np.where(positive_cond, volume, 0)
    vn = np.where(negative_cond, volume, 0)
    
    vp_sum = pd.Series(vp, index=df.index).rolling(window=period, min_periods=period).sum()
    vn_sum = pd.Series(vn, index=df.index).rolling(window=period, min_periods=period).sum()
    
    mav_safe = mav.replace(0, 1)
    vpn_raw = (vp_sum - vn_sum) / mav_safe / period * 100
    
    vpn_ema = vpn_raw.ewm(span=smooth, min_periods=smooth).mean()
    vpn_ma = vpn_ema.rolling(window=ma_period, min_periods=ma_period).mean()
    
    return {
        'vpn': vpn_raw,
        'vpn_smoothed': vpn_ema,
        'vpn_ma': vpn_ma
    }


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.where(df['Close'] > df['Close'].shift(1), 1,
                        np.where(df['Close'] < df['Close'].shift(1), -1, 0))
    return (direction * df['Volume']).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = typical_price(df)
    return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()


def adx(df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    atr_val = atr(high, low, close, period)
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period).sum()
    
    plus_di = 100 * (plus_dm_smooth / atr_val)
    minus_di = 100 * (minus_dm_smooth / atr_val)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(window=period).mean()
    
    return {
        'adx': adx_val,
        'plus_di': plus_di,
        'minus_di': minus_di
    }


def pivot_points(df: pd.DataFrame) -> Dict[str, pd.Series]:
    high = df['High'].shift(1)
    low = df['Low'].shift(1)
    close = df['Close'].shift(1)
    
    pivot = (high + low + close) / 3
    
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }


def add_indicators_to_df(df: pd.DataFrame, indicators_config: Dict) -> pd.DataFrame:
    result = df.copy()
    
    for indicator_name, params in indicators_config.items():
        if indicator_name == 'sma':
            for period in params.get('periods', []):
                result[f'SMA_{period}'] = sma(df['Close'], period)
        
        elif indicator_name == 'ema':
            for period in params.get('periods', []):
                result[f'EMA_{period}'] = ema(df['Close'], period)
        
        elif indicator_name == 'atr':
            period = params.get('period', 14)
            result['ATR'] = atr(df['High'], df['Low'], df['Close'], period)
        
        elif indicator_name == 'rsi':
            period = params.get('period', 14)
            result['RSI'] = rsi(df['Close'], period)
        
        elif indicator_name == 'bollinger':
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2.0)
            bb = bollinger_bands(df['Close'], period, std_dev)
            result['BB_UPPER'] = bb['upper']
            result['BB_MIDDLE'] = bb['middle']
            result['BB_LOWER'] = bb['lower']
        
        elif indicator_name == 'mabw':
            fast = params.get('fast_period', 10)
            slow = params.get('slow_period', 50)
            mult = params.get('multiplier', 1.0)
            mabw_result = mabw(df['Close'], fast, slow, mult)
            result['MAB_UPPER'] = mabw_result['upper']
            result['MAB_MIDDLE'] = mabw_result['middle']
            result['MAB_LOWER'] = mabw_result['lower']
            result['MAB_WIDTH'] = mabw_result['width']
            result['MAB_LLV'] = mabw_result['llv']
        
        elif indicator_name == 'macd':
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            signal = params.get('signal', 9)
            macd_result = macd(df['Close'], fast, slow, signal)
            result['MACD'] = macd_result['macd']
            result['MACD_SIGNAL'] = macd_result['signal']
            result['MACD_HIST'] = macd_result['histogram']
    
    return result


def ama(ohlc: pd.DataFrame, 
        period: int = 10, 
        fast_period: int = 2, 
        slow_period: int = 30) -> pd.Series:
    data = ohlc.copy()
    data.columns = [c.lower() for c in data.columns]
    
    high = data['high']
    low = data['low']
    close = data['close']
    
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.nan)
    
    numerator = abs((close - lowest_low) - (highest_high - close))
    efficiency_ratio = numerator / price_range
    
    ssc = efficiency_ratio * (fast_sc - slow_sc) + slow_sc
    cst = ssc ** 2
    
    ama_values = np.zeros(len(data))
    
    for i in range(len(data)):
        if i == 0:
            ama_values[i] = close.iloc[i]
        elif i < period or pd.isna(cst.iloc[i]):
            ama_values[i] = close.iloc[i]
        else:
            ama_values[i] = ama_values[i-1] + cst.iloc[i] * (close.iloc[i] - ama_values[i-1])
    
    return pd.Series(ama_values, index=data.index, name='AMA')


def efficiency_ratio(price: pd.Series, period: int = 10) -> pd.Series:
    change = price.diff(period).abs()
    volatility = price.diff().abs().rolling(window=period).sum()
    
    volatility = volatility.replace(0, np.nan)
    er = change / volatility
    er = er.replace([np.inf, -np.inf], np.nan)
    
    return er.rename(f'ER{period}')


def kama(price: pd.Series, 
         er_period: int = 10, 
         ema_fast: int = 2, 
         ema_slow: int = 30, 
         period: int = 20) -> pd.Series:
    er = efficiency_ratio(price, period=er_period)
    
    fast_alpha = 2.0 / (ema_fast + 1)
    slow_alpha = 2.0 / (ema_slow + 1)
    
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    
    sma_values = price.rolling(window=period).mean()
    kama_values = sma_values.copy()
    
    for i in range(period, len(price)):
        if pd.notna(kama_values.iloc[i-1]) and pd.notna(sc.iloc[i]):
            kama_values.iloc[i] = (
                kama_values.iloc[i-1] + 
                sc.iloc[i] * (price.iloc[i] - kama_values.iloc[i-1])
            )
    
    return kama_values.rename('KAMA')


def rolling_window(series: pd.Series, window: int) -> List[pd.Series]:
    return [series.iloc[i:i+window] for i in range(len(series) - window + 1)]


def normalize(series: pd.Series, method: str = 'zscore') -> pd.Series:
    if method == 'zscore':
        return (series - series.mean()) / series.std()
    elif method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'percentrank':
        return series.rank(pct=True)
    return series


def bullish_engulfing(open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    current_bullish = close_price > open_price
    
    prev_bearish = close_price.shift(1) < open_price.shift(1)
    
    close_above_prev_open = close_price > open_price.shift(1)
    
    open_below_prev_close = open_price < close_price.shift(1)
    
    current_body = abs(close_price - open_price)
    prev_body = abs(close_price.shift(1) - open_price.shift(1))
    body_engulfing = current_body > prev_body
    
    bull_engulfing = (
        current_bullish & 
        prev_bearish & 
        close_above_prev_open & 
        open_below_prev_close & 
        body_engulfing
    )
    
    return bull_engulfing.rename('BULL_ENGULF')


def bearish_engulfing(open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    current_bearish = close_price < open_price
    
    prev_bullish = close_price.shift(1) > open_price.shift(1)
    
    close_below_prev_open = close_price < open_price.shift(1)
    
    open_above_prev_close = open_price > close_price.shift(1)
    
    current_body = abs(close_price - open_price)
    prev_body = abs(close_price.shift(1) - open_price.shift(1))
    body_engulfing = current_body > prev_body
    
    bear_engulfing = (
        current_bearish & 
        prev_bullish & 
        close_below_prev_open & 
        open_above_prev_close & 
        body_engulfing
    )
    
    return bear_engulfing.rename('BEAR_ENGULF')


def hammer(open_price: pd.Series, 
           high_price: pd.Series,
           low_price: pd.Series, 
           close_price: pd.Series) -> pd.Series:
    body_size = abs(close_price - open_price)
    upper_shadow = high_price - pd.concat([open_price, close_price], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_price, close_price], axis=1).min(axis=1) - low_price
    
    long_lower_shadow = lower_shadow >= (2 * body_size)
    
    small_upper_shadow = upper_shadow <= (0.3 * body_size)
    
    total_range = high_price - low_price
    body_in_upper = (pd.concat([open_price, close_price], axis=1).min(axis=1) - low_price) >= (0.6 * total_range)
    
    return (long_lower_shadow & small_upper_shadow & body_in_upper).rename('HAMMER')


def bbands_squeeze(bb_width: pd.Series, period: int = 125) -> pd.Series:
    lowest_width = bb_width.rolling(window=period, min_periods=period).min()
    squeeze = bb_width <= lowest_width
    
    return squeeze.rename('BB_SQUEEZE')


def bbands_bounce(close: pd.Series, 
                 low: pd.Series,
                 bb_lower: pd.Series) -> pd.Series:
    prev_close_below = close.shift(1) < bb_lower.shift(1)
    
    current_close_above = close > bb_lower
    
    current_low_below = low <= bb_lower
    
    bounce = prev_close_below & current_close_above & current_low_below
    
    return bounce.rename('BBANDS_BOUNCE')


def bbands_breakout(high: pd.Series, bb_upper: pd.Series) -> pd.Series:
    prev_below = high.shift(1) < bb_upper.shift(1)
    
    current_above = high > bb_upper
    
    breakout = prev_below & current_above
    
    return breakout.rename('BBANDS_BREAKOUT')


def stochastic_macd(ohlc: pd.DataFrame,
                   period: int = 45,
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9,
                   adjust: bool = True) -> Dict[str, pd.Series]:
    data = ohlc.copy()
    data.columns = [c.lower() for c in data.columns]
    
    highest_high = data['high'].rolling(window=period, min_periods=period).max()
    lowest_low = data['low'].rolling(window=period, min_periods=period).min()
    
    ema_fast = data['close'].ewm(span=fast_period, adjust=adjust, min_periods=fast_period).mean()
    ema_slow = data['close'].ewm(span=slow_period, adjust=adjust, min_periods=slow_period).mean()
    
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.nan)
    
    stoch_fast = (ema_fast - lowest_low) / price_range
    stoch_slow = (ema_slow - lowest_low) / price_range
    
    stmacd_line = (stoch_fast - stoch_slow) * 100
    
    signal_line = stmacd_line.ewm(span=signal_period, adjust=adjust, min_periods=signal_period).mean()
    
    histogram = stmacd_line - signal_line
    
    return {
        'stmacd': stmacd_line.rename('STMACD'),
        'signal': signal_line.rename('STMACD_SIGNAL'),
        'histogram': histogram.rename('STMACD_HISTOGRAM')
    }


def rsema(price: pd.Series,
         ema_period: int = 50,
         rs_period: int = 50,
         multiplier: float = 10.0,
         adjust: bool = True) -> pd.Series:
    mltp1 = 2.0 / (ema_period + 1.0)
    
    delta = price.diff()
    delta.iloc[0] = 0
    
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    down = down.abs()
    
    gain_ema = up.ewm(span=rs_period, adjust=adjust, min_periods=rs_period).mean()
    loss_ema = down.ewm(span=rs_period, adjust=adjust, min_periods=rs_period).mean()
    
    rs = (gain_ema - loss_ema).abs() / (gain_ema + loss_ema + 0.00001)
    rs = rs * multiplier
    
    rate = mltp1 * (1.0 + rs)
    
    rsema_values = np.zeros(len(price))
    rsema_values[0] = price.iloc[0]
    
    for i in range(1, len(price)):
        rsema_values[i] = rate.iloc[i] * price.iloc[i] + (1.0 - rate.iloc[i]) * rsema_values[i-1]
    
    return pd.Series(rsema_values, index=price.index, name='RSEMA')


def smi(ohlc: pd.DataFrame,
        k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    data = ohlc.copy()
    data.columns = [c.lower() for c in data.columns]
    
    highest_high =  data['high'].rolling(window=k_period, min_periods=k_period).max()
    lowest_low = data['low'].rolling(window=k_period, min_periods=k_period).min()
    close = data['close']

    midpoint = (highest_high + lowest_low) / 2
    rel_diff = close - midpoint
    price_range = highest_high - lowest_low

    smooth_rel_1 = rel_diff.ewm(span=d_period, adjust=False).mean()
    smooth_diff_1 = price_range.ewm(span=d_period, adjust=False).mean()

    avg_rel = smooth_rel_1.ewm(span=d_period, adjust=False).mean()
    avg_diff = smooth_diff_1.ewm(span=d_period, adjust=False).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        smi_val = avg_rel / (avg_diff / 2) * 100

    smi_val = smi_val.replace([np.inf, -np.inf], 0).fillna(0)

    smi_signal = smi_val.ewm(span=d_period, adjust=False).mean()

    return {
        'SMI': smi_val,
        'SMI_SIGNAL': smi_signal
    }


def tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def cama(ohlc: pd.DataFrame, period: int = 20) -> pd.Series:
    data = ohlc.copy()
    data.columns = [c.lower() for c in data.columns]
    
    highest_high =  data['high'].rolling(window=period, min_periods=period).max()
    lowest_low = data['low'].rolling(window=period, min_periods=period).min()
    close =  data['close']
    true_range = tr(data['high'], data['low'], close)
    
    high_low_range = highest_high - lowest_low
    effort = true_range.rolling(window=period).sum().fillna(0)
    
    alpha = pd.Series(0.0, index=close.index)
    
    valid_mask = (effort != 0)
    alpha[valid_mask] = high_low_range[valid_mask] / effort[valid_mask]
    
    price_values = close.values
    alpha_values = alpha.values
    cama_values = np.zeros_like(price_values)
    
    cama_values[0] = price_values[0]
    
    for i in range(1, len(price_values)):
        a = alpha_values[i]
        cama_values[i] = a * price_values[i] + (1 - a) * cama_values[i-1]
        
    return pd.Series(cama_values, index=close.index, name=f"CAMA_{period}")