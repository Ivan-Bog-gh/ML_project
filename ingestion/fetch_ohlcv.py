# ingestion/fetch_ohlcv.py
"""
    Инкрементальная загрузка OHLCV с Binance
    Сохраняет данные в data/raw/{symbol}_{timeframe}.parquet
    Поддерживает дозагрузку только новых свечей
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException


# ─── CONFIG ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Binance weight limit ~1200 в минуту → стараемся не превышать
REQUEST_LIMIT_PER_MIN = 1100
MS_IN_MINUTE = 60_000

# Маппинг удобных названий → binance формат
TIMEFRAME_TO_BINANCE = {
    "1m":  "1m",
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "8h":  "8h",
    "12h": "12h",
    "1d":  "1d",
    "3d":  "3d",
    "1w":  "1w",
}

logger = logging.getLogger("binance_fetcher")


# ─── HELPERS ───────────────────────────────────────────────────────────────

def get_parquet_path(symbol: str, timeframe: str) -> Path:
    """data/raw/{symbol}_{timeframe}.parquet"""
    safe_symbol = symbol.replace("/", "").upper()
    safe_tf = timeframe.lower()
    return RAW_DATA_DIR / f"{safe_symbol}_{safe_tf}.parquet"


def load_existing_data(symbol: str, timeframe: str) -> pd.DataFrame:
    path = get_parquet_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
        if df.empty:
            return df

        # Проверяем наличие нужных колонок
        required = {"open_time", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.warning("Неполные колонки в файле → перезаписываем с нуля")
            return pd.DataFrame()

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.sort_values("open_time")
        return df
    except Exception as e:
        logger.exception(f"Ошибка чтения {path}: {e}")
        return pd.DataFrame()


def save_data(df: pd.DataFrame, symbol: str, timeframe: str):
    if df.empty:
        return

    path = get_parquet_path(symbol, timeframe)
    df.to_parquet(
        path,
        index=False,
        compression="zstd",          # или "gzip", "snappy"
        engine="pyarrow",
    )
    logger.info(f"Сохранено {len(df):,} строк → {path.name}")


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: Optional[datetime | int | str] = None,
    end: Optional[datetime | int | str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    limit_per_request: int = 1000,
) -> pd.DataFrame:
    """
    Инкрементальная загрузка OHLCV с Binance

    Args:
        symbol:       BTCUSDT, ETHUSDT, ...
        timeframe:    1m, 5m, 1h, 4h, 1d, ...
        start:        начало периода (если None → с самой первой свечи)
        end:          конец периода (если None → до текущего момента)
        api_key, api_secret: если не переданы — берутся из переменных окружения

    Returns:
        pd.DataFrame с колонками:
            open_time, open, high, low, close, volume, close_time, ...
    """
    if timeframe not in TIMEFRAME_TO_BINANCE:
        raise ValueError(f"Неподдерживаемый таймфрейм: {timeframe}")

    binance_tf = TIMEFRAME_TO_BINANCE[timeframe]

    client = Client(api_key=api_key, api_secret=api_secret)

    # 1. Загружаем существующие данные
    existing = load_existing_data(symbol, timeframe)
    last_ts_ms = None

    if not existing.empty:
        last_ts_ms = int(existing["open_time"].max().timestamp() * 1000)
        logger.info(f"Найдено {len(existing):,} свечей. Последняя: {existing['open_time'].max()}")

    # 2. Определяем реальное начало загрузки
    if start is None:
        if last_ts_ms is not None:
            start_ms = last_ts_ms + 1          # начинаем со следующей свечи
        else:
            # самую первую свечу по инструменту
            start_ms = None
    else:
        if isinstance(start, datetime):
            start_ms = int(start.timestamp() * 1000)
        elif isinstance(start, str):
            start_ms = int(pd.to_datetime(start).timestamp() * 1000)
        else:
            start_ms = int(start)

        # Если уже есть данные → берём более позднюю границу
        if last_ts_ms is not None:
            start_ms = max(start_ms, last_ts_ms + 1)

    # 3. Конец периода
    if end is None:
        end_ms = None
    else:
        if isinstance(end, datetime):
            end_ms = int(end.timestamp() * 1000)
        elif isinstance(end, str):
            end_ms = int(pd.to_datetime(end).timestamp() * 1000)
        else:
            end_ms = int(end)

    # 4. Собираем новые свечи
    new_candles = []
    current_start = start_ms

    while True:
        try:
            klines = client.get_klines(
                symbol=symbol.upper(),
                interval=binance_tf,
                startTime=current_start,
                endTime=end_ms,
                limit=limit_per_request,
            )

            if not klines:
                break

            new_candles.extend(klines)

            # последняя свеча в ответе
            last_open_time = int(klines[-1][0])
            current_start = last_open_time + 1

            if len(klines) < limit_per_request:
                break  # дошли до конца

            logger.debug(f"Загружено {len(klines)} свечей, следующая с {pd.to_datetime(current_start, unit='ms')}")

        except BinanceAPIException as e:
            logger.error(f"Binance API ошибка: {e}")
            logger.error(f"Код: {e.code}, сообщение: {e.message}")

            if e.code in (-1003, -1007, -1000):  # rate limit / server issues
                logger.info("Достигли лимита или проблема на стороне Binance → ждём минуту")
                import time
                time.sleep(65)
                continue

            raise
        except BinanceRequestException as e:
            logger.error(f"Ошибка запроса: {e}")
            raise

    if not new_candles:
        logger.info("Новых данных нет")
        return existing

    # 5. Преобразуем в DataFrame
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades_count",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df_new = pd.DataFrame(new_candles, columns=columns, dtype=str)# Приводим числовые поля к float/int безопасно
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote", "trades_count"]:
        df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

    # Время оставляем как есть (оно приходит строкой или int)
    df_new["open_time"]  = pd.to_numeric(df_new["open_time"],  errors="coerce")
    df_new["close_time"] = pd.to_numeric(df_new["close_time"], errors="coerce")

    # Убираем битые строки
    df_new = df_new.dropna(subset=["open_time", "close_time"])

    # Теперь конвертируем в datetime
    df_new["open_time"]  = pd.to_datetime(df_new["open_time"],  unit="ms", errors="coerce")
    df_new["close_time"] = pd.to_datetime(df_new["close_time"], unit="ms", errors="coerce")

    df_new = df_new.dropna(subset=["open_time", "close_time"])

    # 6. Простая дедупликация + сортировка
    df_new = df_new.drop_duplicates(subset=["open_time"], keep="last")
    df_new = df_new.sort_values("open_time")

    # 7. Проверка непрерывности (опционально, но полезно)
    if len(df_new) > 1:
        diffs = df_new["open_time"].diff().dt.total_seconds().dropna()
        expected = pd.Timedelta(timeframe).total_seconds()
        gaps = diffs[diffs > expected * 1.1]
        if not gaps.empty:
            logger.warning(f"Обнаружено {len(gaps)} разрывов в данных!")

    # 8. Объединяем со старыми данными
    if existing.empty:
        final_df = df_new
    else:
        final_df = pd.concat([existing, df_new], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["open_time"], keep="last")
        final_df = final_df.sort_values("open_time")

    # 9. Сохраняем
    save_data(final_df, symbol, timeframe)

    logger.info(f"Итого свечей: {len(final_df):,}")
    return final_df

# ─── CLI - runs from the command line ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Инкрементальная загрузка OHLCV с Binance")
    parser.add_argument("--symbol",    default="BTCUSDT",     help="Торговая пара")
    parser.add_argument("--timeframe", default="5m",          help="Таймфрейм")
    parser.add_argument("--start",     default="2022-01-01",  help="Начало периода (YYYY-MM-DD или YYYY-MM-DD HH:MM)")
    parser.add_argument("--end",       default=None,          help="Конец периода")
    parser.add_argument("--no-args",   action="store_true",   help="Использовать значения по умолчанию без argparse")

    args = parser.parse_args()

    # Если явно попросили игнорировать аргументы → используем старый удобный пример
    if args.no_args:
        symbol    = "BTCUSDT"
        timeframe = "5m"
        start     = "2022-01-01"
        end       = None
    else:
        symbol    = args.symbol
        timeframe = args.timeframe
        start     = args.start
        end       = args.end

    print(f"Запуск: {symbol} {timeframe}  start={start}  end={end}")

    df = fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )

    if not df.empty:
        print("\nПоследние 8 строк:")
        print(df.tail(8))
    else:
        print("Данные не загружены")