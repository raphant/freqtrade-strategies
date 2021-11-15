from __future__ import annotations

import logging
import operator
import sys
from abc import ABC, abstractmethod
from functools import reduce
from itertools import zip_longest
from pathlib import Path
from typing import Optional, Union

import freqtrade.vendor.qtpylib.indicators as qta
from freqtrade.strategy import CategoricalParameter, IStrategy
from freqtrade.strategy.hyper import (
    BaseParameter,
)
from pandas import DataFrame
from pydantic.dataclasses import dataclass

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
from indicators import Indicators
import entities.indicator as ind

logger = logging.getLogger('freqtrade').getChild('indicator_opt')

op_map = {
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
    'crossed_above': qta.crossed_above,
    'crossed_below': qta.crossed_below,
}


class IndicatorTools:
    """
    Class to contain indicator functions
    """

    indicator_ids = dict()

    @staticmethod
    def load_indicators():
        """
        Load indicators from file
        """
        Indicators.set_indicator_names()
        Indicators.create_informatives()
        return Indicators.indicators

    @staticmethod
    def get_max_columns():
        """
        Get the maximum number of columns for any indicator
        """
        n_columns_list = [len(indicator.columns) for indicator in indicators.values()]
        return max(n_columns_list)

    @staticmethod
    def get_all_regular_series():
        """
        Get the non value/special type series
        """
        non_value_type_indicators = []
        for indicator in indicators.values():
            if indicator.type == ind.IndicatorType.SERIES:
                non_value_type_indicators.append(indicator)
        columns = set()
        # go through each indicator and add the `all_column` value to columns
        # if the indicator has the attribute `timeframe`, append the indicators `column_append`
        # value to the column name
        for indicator in non_value_type_indicators:
            columns.update(indicator.all_columns)
        return list(columns)

    @classmethod
    def get_all_columns(cls):
        column_list = set()
        # combine all indicator.proper_columns with each inf_timeframe
        # iterate through all indicators
        for indicator in indicators.values():
            # iterate through all formatted_columns
            # if getattr(indicator, 'timeframe', None):
            #     for formatted_column in indicator.formatted_columns:
            #         column_list.add(formatted_column + indicator.column_append)
            #     continue
            column_list.update(indicator.all_columns)
        return list(column_list)

    @classmethod
    def create_series_indicator_map(cls):
        # create a map of series to indicator
        series_indicator_map = {}
        for indicator in indicators.values():
            for column in indicator.all_columns:
                # if getattr(indicator, 'timeframe', None):
                #     series_indicator_map[column + '_' + indicator.timeframe] = indicator
                # else:
                series_indicator_map[column] = indicator
        return series_indicator_map


class InvalidSeriesError(Exception):
    pass


class FailedComparisonError(Exception):
    pass


# region models


class Series(ABC):
    series_name: str

    def get(self, dataframe: DataFrame):
        try:
            return dataframe[self.series_name]
        except Exception as e:
            logger.info('\nseries_name: %s\ndataframe: %s', self.series_name, dataframe)
            logger.exception(e)
            raise

    @property
    def name(self):
        return self.series_name


@dataclass
class InformativeSeries(Series):
    series_name: str
    timeframe: str

    def get(self, dataframe: DataFrame):
        try:
            return dataframe[self.series_name + '_' + self.timeframe]
        except Exception:
            logger.info('\nseries_name: %s\ndataframe: %s', self.series_name, dataframe)
            raise KeyError(
                f'{self.series_name + "_" + self.timeframe} not found in '
                f'dataframe. Available columns: {dataframe.columns}'
            )


@dataclass(frozen=True)
class IndicatorSeries(Series):
    series_name: str

    @property
    def indicator(self):
        return series_map[self.series_name]


@dataclass(frozen=True)
class OhlcSeries(Series):
    series_name: str


@dataclass(frozen=True)
class Comparison(ABC):
    series1: IndicatorSeries

    @abstractmethod
    def compare(self, dataframe: DataFrame, type_: str) -> Series:
        pass

    @classmethod
    def create(
        cls,
        series_name: str,
        op_str: str,
        comparison_series_name: str,
    ) -> Optional['Comparison']:
        """
        Create a comparison object from the parameters in the strategy during hyperopt
        """
        if series_name == 'none' or series_name == comparison_series_name:
            raise InvalidSeriesError()
        series1_indicator: Union[
            ind.ValueIndicator,
            ind.SpecialIndicator,
            ind.SeriesIndicator,
            ind.InformativeIndicator,
        ] = series_map[series_name]

        if series1_indicator.type == 'none':
            raise InvalidSeriesError()
        if series1_indicator.informative:
            series1 = InformativeSeries(
                series_name, timeframe=series1_indicator.timeframe
            )
            # check if "30m", "1h", or "15m" is not found in the series_name, breakpoint
            if not any(
                [timeframe in series_name for timeframe in ['30m', '1h', '15m']]
            ):
                breakpoint()

        else:
            series1 = IndicatorSeries(series_name)
        # return comparison based on the compare type
        if series_name == 'none':
            raise InvalidSeriesError()
        if series1_indicator.type == ind.IndicatorType.VALUE:
            return ValueComparison(series1, op_str)

        if series1_indicator.type == ind.IndicatorType.SPECIAL:
            # here we have a SpecialIndicator. They will only be compared to preconfigured values
            # in indicator.compare. We may get the key or value of indicator.compare, so we will
            # flip the compare dict if we have to.
            compare = series1_indicator.formatted_compare.copy()
            # name_split = series_name.split('__')[1]
            if series_name not in compare:
                # flip the compare dict
                compare = {v: k for k, v in compare.items()}
                if comparison_series_name not in compare:
                    raise InvalidSeriesError()

            comparison_series_name = compare[series_name]
            comparison_series = IndicatorSeries(comparison_series_name)
        # create OhlcSeries if the comparison series is in ohlc
        elif comparison_series_name in ['open', 'high', 'low', 'close']:
            comparison_series = OhlcSeries(comparison_series_name)
        elif hasattr(series_map[comparison_series_name], 'timeframe'):
            comparison_series = InformativeSeries(
                comparison_series_name,
                timeframe=series_map[comparison_series_name].timeframe,
            )
        else:
            comparison_series = IndicatorSeries(comparison_series_name)
        return SeriesComparison(series1, op_str, comparison_series)


@dataclass(frozen=True)
class SeriesComparison(Comparison):
    series1: Union[IndicatorSeries, InformativeSeries]
    op: str
    series2: Union[IndicatorSeries, OhlcSeries, InformativeSeries]

    def compare(self, dataframe: DataFrame, *args):
        series1 = self.series1.get(dataframe)
        operation = op_map[self.op]
        series2 = self.series2.get(dataframe)
        return operation(series1, series2)

    @property
    def name(self):
        return f'{self.series1.series_name} {self.op} {self.series2.name}'


@dataclass(frozen=True)
class ValueComparison(Comparison):
    series1: Union[InformativeSeries, IndicatorSeries]
    op: str

    @property
    def indicator(self):
        return series_map[self.series1.series_name]

    def compare(
        self, dataframe: DataFrame, bs: str, optimized_parameter: BaseParameter = None
    ):
        operation = op_map[self.op]
        if optimized_parameter:
            value = optimized_parameter.value
        else:
            value = self.indicator.get_value(self.series1.series_name, bs).value
        return operation(
            self.series1.get(dataframe),
            value,
        )

    @property
    def name(self):
        return f'{self.series1.series_name} {self.op}'

    # def __repr__(self) -> str:
    #     return ','.join([c.name for c in self.comparisons])


# endregion


class IndicatorOptHelper:
    instance = None

    def __init__(self, n_permutations: int = 1) -> None:
        self.populated = set()
        self.n_permutations = n_permutations

    @property
    def inf_timeframes(self) -> set[str]:
        # create inf_timeframes from all indicators
        timeframes = set()
        for indicator in indicators.values():
            if hasattr(indicator, 'timeframe'):
                timeframes.add(indicator.timeframe)
        return timeframes

    @staticmethod
    def compare(dataframe: DataFrame, comparison: Comparison, type_: str) -> Series:
        logger.info('Comparing %s', comparison.name)
        return comparison.compare(dataframe, type_)

    def create_comparison_groups(
        self, type_, n_groups: int = None, skip_groups: list[int] = None
    ) -> dict[int, dict[str, CategoricalParameter]]:
        """
        Create comparison groups for all indicators
        Args:
            type_: buy or sell
            n_groups: number of comparison groups to create
            skip_groups: A list of ints that won't be optimized.
        Returns: A dictionary of comparison groups
        """
        logger.info('creating group parameters')

        comparison_groups = {}

        all_indicators = IndicatorTools.get_all_columns() + ['none']
        series = IndicatorTools.get_all_regular_series() + [
            'open',
            'close',
            'high',
            'low',
        ]
        for i in range(1, (n_groups or self.n_permutations) + 1):
            optimize = True
            if skip_groups and i in skip_groups:
                optimize = False
            group = {
                'series': CategoricalParameter(
                    all_indicators,
                    default='none',
                    space=type_,
                    optimize=optimize,
                ),
                'operator': CategoricalParameter(
                    list(op_map.keys()),
                    default='none',
                    space=type_,
                    optimize=optimize,
                ),
                'comparison_series': CategoricalParameter(
                    series, default='none', space=type_, optimize=optimize
                ),
            }
            comparison_groups[i] = group
        return comparison_groups

    @classmethod
    def get(cls, permutations=2) -> 'IndicatorOptHelper':
        if IndicatorOptHelper.instance:
            return IndicatorOptHelper.instance
        IndicatorOptHelper.instance = cls(permutations)
        return IndicatorOptHelper.instance

    def create_conditions(
        self,
        dataframe: DataFrame,
        comparison_parameters: dict,
        strategy: IStrategy,
        bs,
        n_per_group: int = None,
    ) -> list[Series]:
        """
        Create conditions for each comparison creating in populate_buy/sell_trend
        Args:
            dataframe: DataFrame from populate_buy/sell_trend
            comparison_parameters: dictionary of comparison parameters
            strategy:
            bs:
            n_per_group:

        Returns: list of condition series
        """
        conditions = []
        for n_group in comparison_parameters:
            after_series = getattr(strategy, f'{bs}_series_{n_group}').value
            op = getattr(strategy, f'{bs}_operator_{n_group}').value
            comparison_series = getattr(
                strategy, f'{bs}_comparison_series_{n_group}'
            ).value
            try:
                comparison = Comparison.create(after_series, op, comparison_series)
                logger.info('Created comparison %s', comparison)
            except InvalidSeriesError:
                continue
            conditions.append(self.compare(dataframe, comparison, bs))
            conditions.append((dataframe['volume'] > 0))
        if not n_per_group:
            return conditions
        final: list[Series] = []
        # group conditions by groups of n_per_group
        group = zip_longest(*[iter(conditions)] * n_per_group, fillvalue=True)
        # go through each condition group and make compare the individual conditions
        for g in group:
            combined = reduce(lambda x, y: x & y, g)
            final.append(combined)
        return final

    def create_local_parameters(
        self,
        locals_: dict,
        num_buy=None,
        num_sell=None,
        buy_skip_groups: list[int] = None,
        sell_skip_groups: list[int] = None,
    ):
        """
        Create the local parameters for the strategy using locals()[x]=y. If num_buy or num_sell
        is 0, then the parameter is not created.
        Args:
            locals_: the local parameters
            num_buy: the number of buy conditions
            num_sell: the number of sell conditions
            buy_skip_groups: the buy groups to skip
            sell_skip_groups: the sell groups to skip

        Returns: The buy and sell comparison parameters
        """
        buy_comparisons, sell_comparisons = {}, {}
        if num_buy:
            buy_comparisons = self.create_comparison_groups(
                'buy', num_buy, buy_skip_groups
            )
            for n_group, p_map in buy_comparisons.items():
                for p_name, parameter in p_map.items():
                    locals_[f'buy_{p_name}_{n_group}'] = parameter
        if num_sell:
            sell_comparisons = self.create_comparison_groups(
                'sell', num_sell, sell_skip_groups
            )
            for n_group, p_map in sell_comparisons.items():
                for p_name, parameter in p_map.items():
                    locals_[f'sell_{p_name}_{n_group}'] = parameter
        return buy_comparisons, sell_comparisons


class CombinationTester:
    def __init__(self, buy_params: dict, sell_params: dict) -> None:
        self.buy_comparisons: list[Comparison] = []
        self.sell_comparisons: list[Comparison] = []
        self.iopt = IndicatorOptHelper.get()
        self.parameter_name_map: dict = {}
        self.populate_comparisons(buy_params.copy(), sell_params.copy())

    def populate_comparisons(self, buy_params, sell_params):
        """
        Create comparisons for buy and sell
        """
        i = 1
        # buy
        while any(buy_params):
            if f'buy_comparison_series_{i}' in buy_params:
                # pop from params
                buy_series = buy_params.pop(f'buy_series_{i}')
                buy_compare_to = buy_params.pop(f'buy_comparison_series_{i}')
                by_op = buy_params.pop(f'buy_operator_{i}')

                self.buy_comparisons.append(
                    Comparison.create(buy_series, by_op, buy_compare_to)
                )
            i += 1
        # sell
        i = 1
        while any(sell_params):
            if f'sell_comparison_series_{i}' in sell_params:
                sell_series = sell_params.pop(f'sell_series_{i}')
                sell_operator = sell_params.pop(f'sell_operator_{i}')
                sell_comparison_series = sell_params.pop(f'sell_comparison_series_{i}')
                self.sell_comparisons.append(
                    Comparison.create(
                        sell_series, sell_operator, sell_comparison_series
                    )
                )
            i += 1

    def create_parameters(self):
        parameters = {'buy': {}, 'sell': {}, 'period': {}}
        for bs in ['buy', 'sell']:
            for i, comparison in enumerate(getattr(self, f'{bs}_comparisons')):
                if isinstance(comparison, ValueComparison):
                    series1 = comparison.series1
                    indicator = series1.indicator
                    p_map = indicator.get_parameters()
                    parameters[bs].update(p_map[bs])
                    for name in p_map[bs]:
                        self.parameter_name_map[comparison.series1.series_name] = name
        # for c in self.buy_comparisons + self.sell_comparisons:
        #     if isinstance(c, ValueComparison):
        #         series1 = c.series1
        #         indicator = series1.indicator
        #         p_map = indicator.get_parameters()
        #
        #         parameters['buy'].update(p_map['buy'])
        #         parameters['sell'].update(p_map['sell'])
        #         for name in [*p_map['buy'], *p_map['sell']]:
        #             self.parameter_name_map[c.series1.series_name] = name
        # if isinstance(bc, SeriesComparison):
        #     series1 = bc.series1
        #     series2 = bc.series2
        #     # get all indicators from series1 and series2
        #     indicators_ = [series1.indicator, series2.indicator]
        #     # get all parameters from indicators
        #     parameters = [i.get_parameters() for i in indicators_]
        return parameters

    def update_local_parameters(self, locals_: dict):
        for type_, p_map in self.create_parameters().items():
            for name, param in p_map.items():
                locals_[name] = param

    @staticmethod
    def compare(
        data: DataFrame,
        comparison: Comparison,
        bs: str,
        optimized_parameter: BaseParameter = None,
    ) -> Series:
        if isinstance(comparison, ValueComparison):
            return comparison.compare(data, bs, optimized_parameter)
        else:
            return comparison.compare(data, bs)

    def get_parameter_name(self, series_name):
        return self.parameter_name_map.get(series_name, '')

    def get_conditions(self, dataframe, strategy: IStrategy, bs: str):
        conditions = []
        for c in self.sell_comparisons:
            parameter_name = self.get_parameter_name(c.series1.series_name)
            parameter = getattr(strategy, parameter_name, None)
            conditions.append(
                self.compare(
                    data=dataframe,
                    comparison=c,
                    bs=bs,
                    optimized_parameter=parameter,
                )
            )
        return conditions


load = True
if __name__ != 'indicatormix.indicator_opt':
    logger.info('Most likely not hyperopting or backtesting. Not loading indicators')
else:
    logger.info('Most likely backtesting or hyperopting. Loading indicators')
    Indicators.set_indicator_names()
    Indicators.create_informatives()
    indicators = Indicators.indicators
    series_map = IndicatorTools.create_series_indicator_map()

if __name__ == '__main__':
    print(indicators['cci'].parsed_values)
    print(indicators['cci'].get_parameters())
    print(indicators['rsi'].get_parameters())
    [print(n, f'{i}\n') for n, i in indicators.items()]
    # pprint(IndicatorOptHelper.get().create_parameters('buy'))
    ...
    # print(convert_to_number('5.5'))
