import pandas as pd
from arrow_logic.universal_rank_confluence import compute_universal_rank_confluence

# === Load sample OHLCV data ===
df = pd.read_csv("sample_data.csv", parse_dates=["timestamp"], index_col="timestamp")

# === Set anchor date/time (must match Thinkorswim AVWAP) ===
anchor_date = 20250404  # Format: YYYYMMDD
anchor_time = 936       # Format: HHMM (e.g., 0936)

# === Run signal computation ===
result = compute_universal_rank_confluence(df, anchor_date, anchor_time)

# === Filter to signal triggers only ===
signals = result[(result['entry_cyan']) | (result['entry_lime']) | (result['entry_gold'])]

# === View tactical outputs ===
print(signals[['entry_cyan', 'entry_lime', 'entry_gold', 'momentum_score', 'sl', 'tp1', 'tp2']])
