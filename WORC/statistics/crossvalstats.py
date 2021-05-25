from contextlib import contextmanager
import pandas as pd
import numpy as np
import logging


class CrossvalStats(object):
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self.reset()

    def reset(self):
        self._iter = {}

    def cviter(self, i, n):
        self._i = i
        self._n = n
        self._eta_iter = None

        for j in range(i, n):
            self._i = j
            self._iter[j] = {'timeit_start': pd.to_datetime('now')}

            if self._eta_iter:
                self._logger.info(f'CrossValidation {j+1} / {n} started at {self._iter[j]["timeit_start"]} ETA: {self._eta_iter}')
            else:
                self._logger.info(f'CrossValidation {j + 1} / {n} started at {self._iter[j]["timeit_start"]}')

            yield i

            self._iter[j]['timeit_end'] = pd.to_datetime('now')
            self._iter[j]['timeit_seconds'] = (self._iter[j]['timeit_end'] - self._iter[j]['timeit_start']).total_seconds()

            self._dostats()
            self._logger.info(f'CrossValidation {j+1} / {n} ended at {self._iter[j]["timeit_end"]}')
            self._logger.info(str(self))

    def _dostats(self):
        df = pd.DataFrame.from_dict(self._iter, orient='index')
        df['timeit_minutes'] = np.ceil((df['timeit_seconds'] / 60)).astype(int)

        df_stats = df['timeit_minutes'].describe().to_frame().T
        self._df_stats = df_stats

        total_minutes = df['timeit_minutes'].sum()
        avg_iter_minutes_upper = total_minutes / (self._i+1) + df_stats.fillna(0).at['timeit_minutes', 'std'] * 1.96
        minutes_remaining = int((total_minutes / (self._i + 1) * self._n - total_minutes)) + df_stats.fillna(0).at['timeit_minutes', 'std'] * 1.96

        eta = pd.to_datetime('now') + pd.Timedelta(minutes=minutes_remaining)
        self._eta = eta
        eta_iter = pd.to_datetime('now') + pd.Timedelta(minutes=avg_iter_minutes_upper)
        self._eta_iter = eta_iter

        self._logger.debug(f'\n{str(df_stats)}')

    def __str__(self):
        return f'ETA Until all iters completion: {self._eta} (local time is now {pd.to_datetime("now")})'