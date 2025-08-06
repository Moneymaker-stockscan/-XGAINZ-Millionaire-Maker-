def compute_universal_rank_confluence(df, anchor_date, anchor_time, float_shares=120_000_000, atr_length=14, risk1=1.5, risk2=2.5):
    import pandas as pd
    import numpy as np

    df = df.copy()
    df['datetime'] = df.index
    df['date'] = df['datetime'].dt.strftime('%Y%m%d').astype(int)
    df['seconds'] = df['datetime'].dt.hour * 3600 + df['datetime'].dt.minute * 60 + df['datetime'].dt.second
    anchor_seconds = int(str(anchor_time)[:2]) * 3600 + int(str(anchor_time)[2:]) * 60

    is_anchor = (df['date'] == anchor_date) & (df['seconds'] == anchor_seconds)
    df['cum_pv'] = np.where(is_anchor, df['close'] * df['volume'], 0).cumsum()
    df['cum_v'] = np.where(is_anchor, df['volume'], 0).cumsum()
    df['anchor_avwap'] = df['cum_pv'] / df['cum_v']

    df['cum_pv2'] = np.where(is_anchor, df['close']**2 * df['volume'], 0).cumsum()
    df['anchor_dev'] = np.sqrt(np.maximum(df['cum_pv2'] / df['cum_v'] - df['anchor_avwap']**2, 0))
    df['upper_band'] = df['anchor_avwap'] + 2 * df['anchor_dev']
    df['lower_band'] = df['anchor_avwap'] - 2 * df['anchor_dev']

    ema9 = df['close'].ewm(span=9).mean()
    ema21 = df['close'].ewm(span=21).mean()
    df['trend_up'] = ema9 > ema21

    df['rsi'] = df['close'].diff().rolling(14).mean()
    df['rsi_stack'] = (df['rsi'] > 55)

    macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['alpha_wave'] = ((df['rsi'] > df['rsi'].shift(1)) & (macd > macd.shift(1)) & (df['volume'] > 2 * df['volume'].rolling(50).mean())).astype(int) * 2

    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_slope_down'] = df['obv'] < df['obv'].shift(1)

    avg_vol = df['volume'].rolling(50).mean()
    df['rel_vol'] = df['volume'] / avg_vol
    df['float_rotation'] = df['volume'] / float_shares

    df['light_green'] = (df['close'] > df['open']) & (df['rel_vol'] > 2)
    df['magenta'] = (df['rel_vol'] > 2) & (df['close'] < df['open'])

    df['momentum_score'] = (
        (df['rsi'] > df['rsi'].shift(1)).astype(int) * 25 +
        (~df['obv_slope_down']).astype(int) * 25 +
        (df['close'] > df['anchor_avwap']).astype(int) * 25 +
