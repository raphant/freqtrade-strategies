"""
https://github.com/raph92/freqtrade-strategies/
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
from indicator_opt import (
    IndicatorOptHelper,
    indicators,
    Comparison,
    InvalidSeriesError,
    CombinationTester,
)

sys.path.append(str(Path(__file__).parent))


logger = logging.getLogger(__name__)

# Buy hyperspace params:
buy_params = {
    "buy_comparison_series_1": "close",
    "buy_comparison_series_2": "T3Average_1h",
    "buy_comparison_series_3": "bb_upperband_1h",
    "buy_comparison_series_4": "EMA",
    "buy_comparison_series_5": "T3Average_1h",
    "buy_comparison_series_6": "bb_lowerband_1h",
    "buy_operator_1": "<=",
    "buy_operator_2": ">=",
    "buy_operator_3": "<=",
    "buy_operator_4": ">=",
    "buy_operator_5": "<",
    "buy_operator_6": ">",
    "buy_series_1": "rsi",
    "buy_series_2": "rsi_1h",
    "buy_series_3": "bb_middleband_1h",
    "buy_series_4": "rvi_1h",
    "buy_series_5": "bb_upperband_1h",
    "buy_series_6": "bb_lowerband",
}


# Sell hyperspace params:
sell_params = {
    "sell_comparison_series_1": "sar_1h",
    "sell_comparison_series_2": "bb_middleband_40",
    "sell_operator_1": ">",
    "sell_operator_2": ">=",
    "sell_series_1": "stoch80_sma10",
    "sell_series_2": "T3Average",
}

ct = CombinationTester(buy_params, sell_params)
iopt = ct.iopt


class IMTest(IStrategy):
    # region Parameters
    ct.update_local_parameters(locals())
    # endregion
    # region Params
    minimal_roi = {"0": 0.10, "20": 0.05, "64": 0.03, "168": 0}
    stoploss = -0.25
    # endregion
    timeframe = '5m'
    use_custom_stoploss = False

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True
    startup_candle_count = 200

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.ct = ct

    def informative_pairs(self) -> ListPairsWithTimeframes:
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, iopt.inf_timeframes) for pair in pairs]
        return informative_pairs

    def populate_informative_indicators(self, dataframe: DataFrame, metadata):
        inf_dfs = {}
        for timeframe in iopt.inf_timeframes:
            inf_dfs[timeframe] = self.dp.get_pair_dataframe(
                pair=metadata['pair'], timeframe=timeframe
            )
        for indicator in indicators.values():
            if not indicator.is_informative:
                continue
            inf_dfs[indicator.timeframe] = indicator.populate(
                inf_dfs[indicator.timeframe]
            )
        for tf, df in inf_dfs.items():
            dataframe = merge_informative_pair(dataframe, df, self.timeframe, tf)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for indicator in indicators.values():
            if indicator.is_informative:
                continue
            dataframe = indicator.populate(dataframe)
        dataframe = self.populate_informative_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        for c in ct.buy_comparisons:
            parameter_name = self.ct.get_parameter_name(c.series1.series_name)
            parameter = getattr(self, parameter_name, None)
            conditions.append(
                self.ct.compare(
                    data=dataframe,
                    comparison=c,
                    bs='buy',
                    optimized_parameter=parameter,
                )
            )
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        for c in ct.sell_comparisons:
            parameter_name = self.ct.get_parameter_name(c.series1.series_name)
            parameter = getattr(self, parameter_name, None)
            conditions.append(
                self.ct.compare(
                    data=dataframe,
                    comparison=c,
                    bs='sell',
                    optimized_parameter=parameter,
                )
            )
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'sell'] = 1
        return dataframe


class IMTestOpt(IStrategy):
    # region Parameters
    # ct
    ct.update_local_parameters(locals())

    # sell
    _, sell_parameters = ct.iopt.create_local_parameters(locals(), num_sell=4)
    # endregion
    # region Params
    # ROI table:
    minimal_roi = {"0": 0.101, "22": 0.077, "57": 0.01, "155": 0}

    stoploss = -0.25
    # Buy hyperspace params:
    buy_params = {
        "ewo__ewo__buy_value": 1.764,
        "stoch_sma__stoch80_sma10__buy_value": 44,
    }

    # endregion
    timeframe = '15m'
    use_custom_stoploss = False

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True
    startup_candle_count = 200

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.ct = ct

    def informative_pairs(self) -> ListPairsWithTimeframes:
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, iopt.inf_timeframes) for pair in pairs]
        return informative_pairs

    def populate_informative_indicators(self, dataframe: DataFrame, metadata):
        inf_dfs = {}
        for timeframe in iopt.inf_timeframes:
            inf_dfs[timeframe] = self.dp.get_pair_dataframe(
                pair=metadata['pair'], timeframe=timeframe
            )
        for indicator in indicators.values():
            if not indicator.is_informative:
                continue
            inf_dfs[indicator.timeframe] = indicator.populate(
                inf_dfs[indicator.timeframe]
            )
        for tf, df in inf_dfs.items():
            dataframe = merge_informative_pair(dataframe, df, self.timeframe, tf)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for indicator in indicators.values():
            if indicator.is_informative:
                continue
            dataframe = indicator.populate(dataframe)
        dataframe = self.populate_informative_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # conditions = iopt.create_conditions(dataframe, self.buy_parameters, self, 'buy')
        conditions = ct.get_conditions(dataframe, self, 'buy')

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # conditions = ct.get_conditions(dataframe, self, 'sell')
        conditions = iopt.create_conditions(
            dataframe, self.sell_parameters, self, 'sell', 2
        )
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'sell'] = 1
        return dataframe
