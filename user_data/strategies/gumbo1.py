"""
https://github.com/raph92?tab=repositories
"""
# --- Do not remove these libs ---
import sys
from datetime import datetime, timedelta
from functools import reduce
from numbers import Number
from pathlib import Path
from pprint import pprint
from typing import Optional, Union, Tuple

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import pandas_ta
import talib.abstract as ta
from finta import TA
from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IntParameter,
    DecimalParameter,
    merge_informative_pair,
    CategoricalParameter,
)
from freqtrade.strategy.interface import IStrategy
from numpy import number
from pandas import DataFrame
from pandas_ta import ema
import logging

sys.path.append(str(Path(__file__).parent))
import custom_indicators as ci

logger = logging.getLogger(__name__)


class Gumbo1(IStrategy):
    # region Parameters
    ewo_low = DecimalParameter(-20.0, 1, default=0, space="buy", optimize=True)
    t3_periods = IntParameter(5, 20, default=5, space="buy", optimize=True)

    stoch_high = IntParameter(60, 100, default=80, space="sell", optimize=True)
    stock_periods = IntParameter(70, 90, default=80, space="sell", optimize=True)

    # endregion
    # region Params
    minimal_roi = {"0": 0.10, "20": 0.05, "64": 0.03, "168": 0}
    stoploss = -0.25
    # endregion
    timeframe = '5m'
    use_custom_stoploss = False
    inf_timeframe = '1h'
    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True
    startup_candle_count = 200

    def informative_pairs(self) -> ListPairsWithTimeframes:
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_informative_indicators(self, dataframe: DataFrame, metadata):
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.inf_timeframe
        )
        # t3 from custom_indicators
        informative['T3'] = ci.T3(informative)
        # bollinger bands
        bbands = ta.BBANDS(informative, timeperiod=20)
        informative['bb_lowerband'] = bbands['lowerband']
        informative['bb_middleband'] = bbands['middleband']
        informative['bb_upperband'] = bbands['upperband']

        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, self.inf_timeframe
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ewo
        dataframe['EWO'] = ci.EWO(dataframe)
        # ema
        dataframe['EMA'] = ta.EMA(dataframe)
        # t3
        for i in self.t3_periods.range:
            dataframe[f'T3_{i}'] = ci.T3(dataframe, i)
        # bollinger bands 40
        bbands = ta.BBANDS(dataframe, timeperiod=40)
        dataframe['bb_lowerband_40'] = bbands['lowerband']
        dataframe['bb_middleband_40'] = bbands['middleband']
        dataframe['bb_upperband_40'] = bbands['upperband']
        # stochastic
        # stochastic windows
        for i in self.stock_periods.range:
            dataframe[f'stoch_{i}'] = ci.stoch_sma(dataframe, window=i)
        dataframe = self.populate_informative_indicators(dataframe, metadata)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # ewo < 0
        conditions.append(dataframe['EWO'] < self.ewo_low.value)
        # middleband 1h >= t3 1h
        conditions.append(dataframe['bb_middleband_1h'] >= dataframe['T3_1h'])
        # t3 <= ema
        conditions.append(dataframe[f'T3_{self.t3_periods.value}'] <= dataframe['EMA'])
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # stoch > 80
        conditions.append(
            dataframe[f'stoch_{self.stock_periods.value}'] > self.stoch_high.value
        )
        # t3 >= middleband_40
        conditions.append(
            dataframe[f'T3_{self.t3_periods.value}'] >= dataframe['bb_middleband_40']
        )
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'sell'] = 1
        return dataframe
