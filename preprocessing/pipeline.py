# preprocessing/pipeline.py
import argparse

from .clean_ohlcv import clean_ohlcv
from .align_time import align_time_controlled
from .validate_data import run_all_validations 
from pathlib import Path
import pandas as pd
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
    except Exception as e:
        logger.exception(f"Ошибка чтения parquet: {e}")
        return
    logger.info("Чтение завершено")
    logger.info("Начинаем очистку данных")

    df = clean_ohlcv(df)
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