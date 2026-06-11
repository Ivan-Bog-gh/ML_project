# preprocessing/pipeline.py
import argparse
from pathlib import Path

import sys
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from preprocessing.clean_ohlcv import clean_ohlcv
from preprocessing.align_time import align_time_controlled
from preprocessing.validate_data import run_all_validations 
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ─── CONFIG ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("preprocessing.pipeline")

RAW_DIR      = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

def process_raw_to_interim(symbol: str, timeframe: str):
    symbol = symbol.upper()
    timeframe = timeframe.lower()
    raw_path        = RAW_DIR     / f"{symbol}_{timeframe}.parquet"
    interim_path    = INTERIM_DIR / f"{symbol}_{timeframe}.parquet"
    
    if not raw_path.is_file():
        logger.error(f"Файл не найден: {raw_path}")
        return

    logger.info(f"Обрабатываем {symbol} {timeframe}")
    logger.info(f"  raw    → {raw_path}")
    logger.info(f"  interim → {interim_path}")

    # 1. Чтение
    try:
        df = pd.read_parquet(raw_path)
        logger.info(f"Прочитано {len(df):,} строк")
        if "timestamp" in df.columns:   # На случай взятия из другого источника, где может быть такое название (DIB)
            df.rename(columns={"timestamp": "close_time"}, inplace=True)  # На всякий случай убираем пробелы в названиях
            df["open_time"] = df["close_time"].shift().fillna(df.iloc[0]["close_time"]-1) + 1  # На всякий случай убираем пробелы в названиях
            # print(f"0/-1: {pd.to_datetime(df.iloc[0]['close_time'], unit='ms')}, {pd.to_datetime(df.iloc[-1]['close_time'], unit='ms')}")
            
            mask_close = df["close_time"] < 5e12  # Если в микросекундах, а не миллисекундах
            df["close_time"] = pd.to_datetime(np.where(mask_close, df["close_time"], df["close_time"] / 1000), unit='ms')
            
            mask_open = df["open_time"] < 5e12  # Если в микросекундах, а не миллисекундах
            df["open_time"] = pd.to_datetime(np.where(mask_open, df["open_time"], df["open_time"] / 1000), unit='ms')
    except Exception as e:
        logger.exception(f"Ошибка чтения parquet: {e}")
        return
    logger.info("Чтение завершено")
    logger.info("Начинаем очистку данных")

    df = clean_ohlcv(df)
    if timeframe.find("ib") == -1: # Если не Imbalance Bar
        df = align_time_controlled(df, timeframe)
    success = run_all_validations(df, timeframe)
    
    if success:
        df = df.set_index("open_time") # Проверка в run_all_validations()
        df.index = pd.to_datetime(df.index)
        try:
            df.to_parquet(
                interim_path,
                index=True,
                compression="zstd",
                engine="pyarrow"
            )
            logger.info(f"Успешно сохранено → {interim_path}")
            logger.info(f"Итоговое количество строк: {len(df):,}")
        except Exception as e:
            logger.exception(f"Ошибка сохранения: {e}")
    else:
        logger.error("Валидация НЕ пройдена → файл НЕ сохранён")
        

# ─── CLI - runs from the command line ──────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Обработка сырых OHLCV → interim")
    parser.add_argument("symbol",    type=str,    help="Торговая пара (BTCUSDT, ETHUSDT и т.д.)")
    parser.add_argument("timeframe", type=str,    help="Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d ...)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод (DEBUG)")
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    process_raw_to_interim(args.symbol, args.timeframe)