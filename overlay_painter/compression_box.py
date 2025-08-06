def compute_compression_zone(df, lookback=10, compression_factor=0.75):
    import pandas as pd
    df = df.copy()
    high_range = df['high'].rolling(lookback).max()
    low_range = df['low'].rolling(lookback).min()
    box_range = high_range - low_range
    avg_box_range = box_range.rolling(20).mean()
    is_compressed = box_range < (avg_box_range * compression_factor)
    box_high = high_range.where(is_compressed)
    box_low = low_range.where(is_compressed)
    return box_high, box_low
