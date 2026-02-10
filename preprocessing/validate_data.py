# preprocessing/validate_data.py
"""
    Валидатор parquet-файлов с OHLCV данными (Binance)
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ------------------------------------------------------------------------------------
#  Logging settings
# ------------------------------------------------------------------------------------

logger = logging.getLogger("ohlcv_validator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ------------------------------------------------------------------------------------
#  Main validations
# ------------------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> bool:
    """Проверяет наличие обязательных колонок и их типы + сортировку по времени"""
    required_columns = {
        "open_time":  ["datetime64[ns]", "int64"],
        "open":       ["float64"],
        "high":       ["float64"],
        "low":        ["float64"],
        "close":      ["float64"],
        "volume":     ["float64"],
    }

    ok = True

    # Проверка наличия колонок
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"Отсутствуют обязательные колонки: {', '.join(missing)}")
        ok = False

    # Проверка типов (хотя бы один из допустимых)
    for col, allowed_dtypes in required_columns.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            if actual not in allowed_dtypes:
                logger.warning(f"Неправильный тип колонки {col}: ожидается один из {allowed_dtypes}, получено {actual}")

    # Проверка, что open_time отсортирован
    if "open_time" in df.columns and not df["open_time"].is_monotonic_increasing:
        logger.error("Столбец open_time НЕ отсортирован по возрастанию")
        ok = False

    return ok


def validate_no_duplicates(df: pd.DataFrame) -> bool:
    """Проверяет дубликаты по open_time"""
    if "open_time" not in df.columns:
        return True  # уже ругались в схеме

    dups = df["open_time"].duplicated(keep=False)
    if dups.any():
        dup_count = dups.sum()
        dup_times = df.loc[dups, "open_time"].head(3).tolist()
        logger.error(f"Найдено {dup_count} дубликатов по open_time (примеры: {dup_times})")
        return False
    return True


def validate_time_continuity(df: pd.DataFrame, timeframe: str) -> bool:
    """Проверяет непрерывность временного ряда"""
    if "open_time" not in df.columns or df.empty:
        return True

    # Преобразуем timeframe в timedelta
    try:
        delta = pd.Timedelta(timeframe)
    except ValueError:
        logger.error(f"Невозможно распознать timeframe: {timeframe}")
        return False

    # Разница между соседними свечами
    diffs = df["open_time"].diff().dropna()

    gaps = diffs[diffs > pd.Timedelta(seconds=delta.total_seconds() * 1.1)]
    if not gaps.empty:
        logger.warning(f"Обнаружено {len(gaps)} разрывов в данных (больше {delta * 1.1})")
        logger.info(f"Самый большой разрыв: {gaps.max()}")
        logger.info(f"Первые 3 разрыва:\n{gaps.head(3)}")
        return False

    # Проверяем, что нет наложений / отрицательных разниц
    if (diffs <= pd.Timedelta(0)).any():
        logger.error("Найдены отрицательные или нулевые разницы между свечами (нарушение порядка)")
        return False

    return True


def validate_ohlcv_consistency(df: pd.DataFrame) -> bool:
    """Бизнес-логика свечей: high ≥ max(o,c), low ≤ min(o,c), volume ≥ 0"""
    if not all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
        return True  # уже ругались в схеме

    violations = []

    # high должен быть максимумом среди open/close/high
    bad_high = df["high"] < df[["open", "close"]].max(axis=1)
    if bad_high.any():
        violations.append(f"{bad_high.sum()} свечей, где high < max(open, close)")

    # low должен быть минимумом
    bad_low = df["low"] > df[["open", "close"]].min(axis=1)
    if bad_low.any():
        violations.append(f"{bad_low.sum()} свечей, где low > min(open, close)")

    # volume
    bad_volume = df["volume"] < 0
    if bad_volume.any():
        violations.append(f"{bad_volume.sum()} свечей с отрицательным volume")

    if violations:
        logger.error("Нарушена консистентность OHLCV:")
        for v in violations:
            logger.error("  • " + v)
        return False

    return True


def validate_missing_values(df: pd.DataFrame) -> bool:
    """Пропуски в ключевых полях"""
    critical_cols = ["open_time", "open", "high", "low", "close", "volume"]

    missing = df[critical_cols].isna().sum() # Суммирование по колонке
    missing = missing[missing > 0]

    if not missing.empty:
        logger.error("Найдены пропущенные значения в критических колонках:")
        for col, cnt in missing.items():
            logger.error(f"  • {col}: {cnt} пропусков")
        return False

    return True


def validate_extreme_values(df: pd.DataFrame) -> bool:
    """Очень странные / нереалистичные значения"""
    if not all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
        return True

    bad = []

    # Отрицательные или нулевые цены (кроме некоторых стейблкоинов, но для большинства пар — аномалия)
    zero_or_neg = df[["open", "high", "low", "close"]] <= 0
    if zero_or_neg.any().any():
        cnt = zero_or_neg.sum().sum()
        bad.append(f"{cnt} значений ≤ 0 в OHLC")

    # Слишком большой спред (примерно)
    spread_ratio = (df["high"] - df["low"]) / df["close"].replace(0, 1e-10)
    extreme_spread = spread_ratio > 10  # >1000% спред — почти всегда ошибка
    if extreme_spread.any():
        bad.append(f"{extreme_spread.sum()} свечей с экстремальным спредом (>1000%)")

    if bad:
        logger.warning("Обнаружены экстремальные / подозрительные значения:")
        for msg in bad:
            logger.warning("  • " + msg)
        return False

    return True


# ------------------------------------------------------------------------------------
#  Главная функция-оркестратор
# ------------------------------------------------------------------------------------

def run_all_validations(df: pd.DataFrame, timeframe: str) -> bool:
    """
    Запускает все проверки подряд.
    Возвращает True, если все проверки прошли успешно.
    """
    checks = [
        ("Схема и порядок",          validate_schema),
        ("Дубликаты",                validate_no_duplicates),
        ("Непрерывность времени",    lambda df: validate_time_continuity(df, timeframe)),
        ("Консистентность OHLCV",    validate_ohlcv_consistency),
        ("Пропущенные значения",     validate_missing_values),
        ("Экстремальные значения",   validate_extreme_values),
    ]

    all_passed = True

    logger.info(f"Запуск валидации для {len(df)} свечей, timeframe = {timeframe}")

    for name, func in checks:
        logger.info(f"→ {name} ...")
        passed = func(df)
        if not passed:
            all_passed = False
        # можно добавить короткий статус в одну строку, если хочется

    if all_passed:
        logger.info("Все проверки пройдены успешно ✓")
    else:
        logger.error("Валидация НЕ пройдена ✗")

    return all_passed


# ------------------------------------------------------------------------------------
#  CLI-интерфейс
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path",         type=str,               help="Путь к .parquet файлу")
    parser.add_argument("timeframe",    type=str,               help="Таймфрейм (1m, 5m, 1h, 4h, 1d и т.д.)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Показывать DEBUG логи")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    path = Path(args.path)
    if not path.is_file():
        logger.error(f"Файл не найден: {path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(path)
        logger.info(f"Прочитано {len(df):,} строк из {path.name}")
    except Exception as e:
        logger.exception(f"Не удалось прочитать parquet: {e}")
        sys.exit(1)

    success = run_all_validations(df, args.timeframe)

    sys.exit(0 if success else 1)