"""all indicators will be defined here"""
import sys
from collections import namedtuple
from pathlib import Path
from typing import Union

import freqtrade.vendor.qtpylib.indicators as qta
import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta
import technical.indicators as tita
from freqtrade.strategy import DecimalParameter, CategoricalParameter, IntParameter

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
# noinspection PyUnresolvedReferences
from helpers import custom_indicators as ci

import entities.indicator as ind

OptimizeField = namedtuple(
    'OptimizeField',
    [
        'column_name',
        'buy_or_sell',
        'parameter_type',
        'default',
        'low',
        'high',
    ],
)
IP = IntParameter
DP = DecimalParameter
CP = CategoricalParameter

# region Value Indicators
VALUE_INDICATORS = {
    "rsi": ind.ValueIndicator(
        func=ta.RSI,
        columns=['rsi'],
        optimize_func__timeperiod=IP(10, 20, default=14, space='buy'),
        optimize_value__rsi__buy=IP(0, 50, default=30, space='buy'),
        optimize_value__rsi__sell=IP(50, 100, default=70, space='sell'),
    ),
    "rsi_fast": ind.ValueIndicator(
        func=ta.RSI,
        columns=['rsi'],
        optimize_func__timeperiod=IP(1, 10, default=4, space='buy'),
        optimize_value__rsi__buy=IP(0, 50, default=35, space='buy'),
        optimize_value__rsi__sell=IP(50, 100, default=65, space='sell'),
    ),
    "rsi_slow": ind.ValueIndicator(
        func=ta.RSI,
        columns=['rsi'],
        optimize_func__timeperiod=IP(15, 25, default=20, space='buy'),
        optimize_value__rsi__buy=IP(0, 50, default=35, space='buy'),
        optimize_value__rsi__sell=IP(50, 100, default=65, space='sell'),
    ),
    "rvi": ind.ValueIndicator(
        func=lambda dataframe, timeperiod: ci.rvi(dataframe, periods=timeperiod),
        columns=['rvi'],
        optimize_func__timeperiod=IP(10, 20, default=14, space='buy'),
        optimize_value__rvi__buy=IP(50, 100, default=60, space='buy'),
        optimize_value__rvi__sell=IP(0, 50, default=40, space='sell'),
    ),
    "rmi": ind.ValueIndicator(
        func=lambda dataframe, timeperiod: ci.RMI(dataframe, length=timeperiod),
        columns=['rmi'],
        optimize_func__timeperiod=IP(15, 30, default=20, space='buy'),
        optimize_value__rmi__buy=IP(0, 50, default=30, space='buy'),
        optimize_value__rmi__sell=IP(50, 100, default=70, space='sell'),
    ),
    "ewo": ind.ValueIndicator(
        func=ci.EWO,
        columns=['ewo'],
        optimize_func__ema_length=IP(45, 60, default=50, space='buy', optimize=False),
        optimize_func__ema2_length=IP(20, 40, default=200, space='buy', optimize=False),
        optimize_value__ewo__buy=DP(-20, -8, default=-14, space='buy'),
        optimize_value__ewo__sell=DP(-2, 2, default=0, space='sell'),
    ),
    "ewo_high": ind.ValueIndicator(
        func=ci.EWO,
        columns=['ewo'],
        optimize_func__ema_length=IP(45, 60, default=50, space='buy', optimize=False),
        optimize_func__ema2_length=IP(20, 40, default=200, space='buy', optimize=False),
        optimize_value__ewo__buy=DP(2.0, 12.0, default=2.4, space='buy'),
        optimize_value__ewo__sell=DP(-2, 2, default=0, space='sell'),
    ),
    "ewo_high2": ind.ValueIndicator(
        func=ci.EWO,
        columns=['ewo'],
        optimize_func__ema_length=IP(45, 60, default=50, space='buy', optimize=False),
        optimize_func__ema2_length=IP(20, 40, default=200, space='buy', optimize=False),
        optimize_value__ewo__buy=DP(-6.0, 12.0, default=-5.5, space='buy'),
        optimize_value__ewo__sell=DP(-2, 2, default=0, space='sell'),
    ),
    "cci": ind.ValueIndicator(
        func=lambda df, timeperiod: pta.cci(
            **df[['high', 'low', 'close']], length=timeperiod
        ),
        columns=['cci'],
        values=[
            OptimizeField('cci', 'buy', int, 100, 50, 150),
            OptimizeField('cci', 'sell', int, -100, -150, -50),
        ],
        optimize_func__timeperiod=IP(10, 25, default=14, space='buy'),
        optimize_value__cci__buy=IP(50, 150, default=100, space='buy'),
        optimize_value__cci__sell=IP(-150, -50, default=-100, space='sell'),
    ),
    "stoch_sma": ind.ValueIndicator(
        func=lambda dataframe, timeperiod, sma_window: ci.stoch_sma(
            dataframe, window=timeperiod, sma_window=sma_window
        ),
        columns=['stoch_sma'],
        optimize_func__timeperiod=IP(60, 100, default=80, space='buy'),
        optimize_func__sma_window=IP(5, 20, default=10, space='buy', optimize=False),
        optimize_value__stoch_sma__buy=IP(0, 50, default=20, space='buy'),
        optimize_value__stoch_sma__sell=IP(60, 100, default=100, space='sell'),
        inf_timeframes=['1h', '30m'],
    ),
    # "awesome_oscillator": ValueIndicator(
    #     func=lambda df, fast, slow: qta.awesome_oscillator(df, fast=fast, slow=slow),
    #     columns=['ao'],
    #     values=[
    #         OptimizeField('ao', 'buy', 'decimal', 0, -2.0, 2.0),
    #         OptimizeField('ao', 'sell', 'decimal', 0, -2.0, 2.0),
    #     ],
    #     optimize_func__fast=IP(1, 14, default=5, space='buy'),
    #     optimize_func__slow=IP(14, 45, default=34, space='buy'),
    #     optimize_value__stoch_sma__buy=DP(0, 50, default=20, space='buy'),
    #     optimize_value__stoch_sma__sell=DP(60, 100, default=100, space='sell'),
    # ),
    "adx": ind.ValueIndicator(
        func=ta.ADX,
        func_columns=['high', 'low', 'close'],
        columns=['adx'],
        optimize_func__timeperiod=IP(5, 20, default=14, space='buy'),
        optimize_value__adx__buy=IP(25, 100, default=50, space='buy'),
        optimize_value__adx__sell=IP(0, 25, default=25, space='sell'),
        inf_timeframes=['1h', '30m'],
    ),
}
# endregion

# region SeriesIndicators
SERIES_INDICATORS = {
    "psar": ind.SeriesIndicator(
        func=ta.SAR,
        func_columns=['high', 'low'],
        columns=['sar'],
        inf_timeframes=['1h', '30m'],
    ),
    "bb_fast": ind.SeriesIndicator(
        func=ci.bollinger_bands,
        columns=[
            "bb_lowerband",
            "bb_middleband",
            "bb_upperband",
        ],
        inf_timeframes=['1h', '30m'],
        optimize_func__timeperiod=IP(10, 30, default=20, space='buy'),
    ),
    "bb_slow": ind.SeriesIndicator(
        func=ci.bollinger_bands,
        columns=[
            "bb_lowerband",
            "bb_middleband",
            "bb_upperband",
        ],
        inf_timeframes=['1h', '30m'],
        optimize_func__timeperiod=IP(40, 60, default=50, space='buy'),
    ),
    "tema_fast": ind.SeriesIndicator(
        func=ta.TEMA,
        columns=["TEMA"],
        inf_timeframes=['1h', '30m'],
        optimize_timeperiod=True,
        optimize_func__timeperiod=IP(5, 15, default=9, space='buy'),
    ),
    "tema_slow": ind.SeriesIndicator(
        func=ta.TEMA,
        columns=["TEMA"],
        inf_timeframes=['1h', '30m'],
        optimize_func__timeperiod=IP(80, 120, default=100, space='buy'),
    ),
    "hema_slow": ind.SeriesIndicator(
        func=lambda dataframe, timeperiod: qta.hull_moving_average(
            dataframe, window=timeperiod
        ),
        func_columns=['close'],
        columns=["hma"],
        inf_timeframes=['1h', '30m'],
        optimize_timeperiod=True,
        optimize_func__timeperiod=IP(180, 210, default=200, space='buy'),
    ),
    "hema_fast": ind.SeriesIndicator(
        func=lambda dataframe, timeperiod: qta.hull_moving_average(
            dataframe, window=timeperiod
        ),
        func_columns=['close'],
        columns=["hma"],
        inf_timeframes=['1h', '30m'],
        optimize_timeperiod=True,
        optimize_func__timeperiod=IP(5, 15, default=9, space='buy'),
    ),
    "ema_fast": ind.SeriesIndicator(
        func=ta.EMA,
        columns=["EMA"],
        inf_timeframes=['1h', '30m'],
        optimize_timeperiod=True,
        optimize_func__timeperiod=IP(5, 15, default=9, space='buy'),
    ),
    "ema_slow": ind.SeriesIndicator(
        func=ta.EMA,
        columns=["EMA"],
        inf_timeframes=['1h', '30m'],
        optimize_timeperiod=True,
        optimize_func__timeperiod=IP(90, 110, default=100, space='buy'),
    ),
    "wma_fast": ind.SeriesIndicator(
        func=ta.WMA,
        columns=["WMA"],
        inf_timeframes=['1h', '30m'],
        optimize_timeperiod=True,
        optimize_func__timeperiod=IP(5, 15, default=9, space='buy'),
    ),
    "wma_slow": ind.SeriesIndicator(
        func=ta.WMA,
        columns=["WMA"],
        optimize_timeperiod=False,
        optimize_func__timeperiod=IP(90, 110, default=100, space='buy'),
    ),
    "sma_fast": ind.SeriesIndicator(
        func=ta.SMA,
        columns=["SMA"],
        optimize_timeperiod=False,
        optimize_func__timeperiod=IP(5, 15, default=9, space='buy'),
    ),
    "sma_slow": ind.SeriesIndicator(
        func=ta.SMA,
        columns=["SMA"],
        optimize_timeperiod=False,
        optimize_func__timeperiod=IP(190, 210, default=200, space='buy'),
    ),
    "t3": ind.SeriesIndicator(
        func=lambda df, timeperiod: ci.T3(df, length=timeperiod),
        columns=[
            "T3Average",
        ],
        inf_timeframes=['1h', '30m'],
        optimize_timeperiod=False,
        optimize_func__timeperiod=IP(5, 10, default=5, space='buy'),
    ),
    "zema": ind.SeriesIndicator(
        func=lambda df, timeperiod: tita.zema(df, period=timeperiod),
        columns=[
            "zema",
        ],
        inf_timeframes=['1h', '30m'],
        optimize_timeperiod=False,
        optimize_func__timeperiod=IP(5, 80, default=20, space='buy'),
    ),
}
# endregion

# region SpecialIndicators
SPECIAL_INDICATORS = {
    "macd": ind.SpecialIndicator(
        func=lambda df, fast, slow, smooth: qta.macd(
            df['close'], fast=fast, slow=slow, smooth=smooth
        ),
        columns=['macd', 'signal'],
        optimize_func__fast=IP(5, 20, default=12, space='buy', optimize=False),
        optimize_func__slow=IP(15, 30, default=26, space='buy', optimize=False),
        optimize_func__smooth=IP(5, 20, default=14, space='buy', optimize=True),
        compare={'macd': 'signal'},
    )
}


# endregion
class Indicators:
    populated = False
    indicators: dict[
        str,
        Union[
            ind.SeriesIndicator,
            ind.ValueIndicator,
            ind.SeriesIndicator,
            ind.InformativeIndicator,
        ],
    ] = {
        **VALUE_INDICATORS,
        **SERIES_INDICATORS,
        **SPECIAL_INDICATORS,
    }

    @classmethod
    def get(cls, name: str) -> ind.Indicator:
        """
        Get an indicator by name.
        """
        return cls.indicators[name]

    @classmethod
    def create_informatives(cls):
        """
        Create the informative indicators using indicator.create_informatives.
        create_informatives returns a dict that we will use to update each set of indicators.
        """
        if cls.populated:
            return
        for indicator in cls.indicators.copy().values():
            cls.indicators.update(indicator.create_informatives())
        cls.populated = True

    @classmethod
    def set_indicator_names(cls) -> None:
        """
        For each indicator value in each indicator dict, set the indicator.name to its key.
        """
        for indicator_name, indicator in cls.indicators.items():
            indicator.name = indicator_name


if __name__ == '__main__':
    Indicators.set_indicator_names()
    Indicators.create_informatives()
    print(Indicators.indicators)
    indicator = Indicators.get('hema_slow_1h')
    # print(indicator.create_informatives())
    # # create a dataframe with random ohlcv data
    df = pd.DataFrame(
        {
            'open': np.random.randint(1, 100, size=500),
            'high': np.random.randint(1, 100, size=500),
            'low': np.random.randint(1, 100, size=500),
            'close': np.random.randint(1, 100, size=500),
            'volume': np.random.randint(1, 100, size=500),
        }
    )
    #
    # print(indicator.value_parameters)
    # print(indicator.function_parameters)
    # print(indicator.populate(df))
    #
    # test indicator.populate by iterating through each indicator
    for indicator_name, indicator in Indicators.indicators.items():
        print(indicator_name)
