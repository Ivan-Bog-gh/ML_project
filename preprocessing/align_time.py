# preprocessing/align_time.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def align_time_controlled(df: pd.DataFrame, timeframe: str, max_ffill_hours=4):
    freq = pd.Timedelta(timeframe)
    max_ffill_bars = int(pd.Timedelta(hours=max_ffill_hours) / freq)

    df = df.set_index('open_time').sort_index()
    aligned = df.asfreq(freq)

    # Флаги
    aligned['is_missing'] = aligned['close'].isna()
    aligned['gap_group'] = (aligned['is_missing'] != aligned['is_missing'].shift()).cumsum()
    gap_sizes = aligned['is_missing'].groupby(aligned['gap_group']).sum()

    # Заполняем короткие гэпы
    short_gaps = gap_sizes[(gap_sizes <= max_ffill_bars) & (gap_sizes > 0)].index
    for g in short_gaps:
        mask = aligned['gap_group'] == g

        idx = aligned[mask].index[0] - freq # До гэпа
        for col in ['open','high','low','close']:
            aligned.loc[mask, col] = aligned.loc[idx, col]
        aligned.loc[mask, ['volume','quote_volume','trades_count']] = 0

    # Длинные гэпы оставляем как есть (NaN) или можно fail здесь
    long_gaps = gap_sizes[gap_sizes > max_ffill_bars]
    if not long_gaps.empty:
        logger.warning(f"Оставлены как NaN {long_gaps.sum()} баров в гэпах > {max_ffill_hours} ч")

    # Дополнительный флаг "подозрительная заморозка"
    aligned['is_suspended'] = (
        (aligned['volume'] == 0) &
        (aligned['open'] == aligned['high']) &
        (aligned['high'] == aligned['low']) &
        (aligned['low'] == aligned['close'])
    )

    aligned = aligned.reset_index()
    return aligned