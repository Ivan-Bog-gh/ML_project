# validation/check_labels.py

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger("check_labels")

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

def check_label_distribution(
    symbol: str, 
    timeframe: str
):
    symbol          = symbol.upper()
    timeframe       = timeframe.lower()
    processed_path  = PROCESSED_DIR / f"{symbol}_{timeframe}_labels.parquet"
    
    if not processed_path.is_file():
        logger.error(f"Файл не найден: {processed_path}")
        return
        
    labels = pd.read_parquet(processed_path)
    dist = labels["label"].value_counts(normalize=True).sort_index()
    counts = labels["label"].value_counts().sort_index()
    
    assert dist.get(0, 0) > 0.6, "Too many trades — likely noisy labeling"
    assert dist.get(1, 0) > 0.01, "Too few positive events"
    assert dist.get(-1, 0) > 0.01, "Too few negative events"
    

    print("Counts:\n", counts)
    print("Distribution:\n", dist)
    return True


# ─── CLI - runs from the command line ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Инкрементальная загрузка OHLCV с Binance")
    parser.add_argument("--symbol",     default="BTCUSDT",      help="Торговая пара")
    parser.add_argument("--timeframe",  default="5m",           help="Таймфрейм")
    parser.add_argument("--no-args",    action="store_true",    help="Использовать значения по умолчанию без argparse")

    args = parser.parse_args()

    # Если явно попросили игнорировать аргументы → используем старый удобный пример
    if args.no_args:
        symbol      = "BTCUSDT"
        timeframe   = "5m"
    else:
        symbol      = args.symbol
        timeframe   = args.timeframe

    check_label_distribution(symbol=symbol, timeframe=timeframe)