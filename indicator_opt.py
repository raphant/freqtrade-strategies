import logging
import operator
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Callable
from functools import cache, reduce
from itertools import zip_longest
from pathlib import Path
from typing import Optional, Union, Any

import freqtrade.vendor.qtpylib.indicators as qta
import pandas_ta as pta
import talib.abstract as tta
from diskcache import FanoutCache
from finta import TA as fta
from freqtrade.strategy import CategoricalParameter, IntParameter, IStrategy
from freqtrade.strategy.hyper import (
    NumericParameter,
    DecimalParameter,
    RealParameter,
    BaseParameter,
)
from pandas import DataFrame
from pydantic.dataclasses import dataclass

try:
    import custom_indicators as ci
except:
    import lazyft.custom_indicators as ci
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ta_map = {'tta': tta, 'pta': pta, 'qta': qta, 'fta': fta, 'ci': ci}
op_map = {
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
    'crossed_above': qta.crossed_above,
    'crossed_below': qta.crossed_below,
}
# partner_map = {
#     'ma': [''],
#     'mom': ['vol'],
#     'trend': [''],
#     'osc': [''],
#     'vol': [''],
# }


# region Utility functions
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def convert_to_number(value: str) -> Union[int, float, None]:
    # check if string is a float
    if '.' in value:
        return float(value)
    # check if string is an int
    try:
        return int(value)
    except ValueError:
        raise ValueError(f'{value} is not a valid number')


def load_function(indicator_name: str) -> Callable:
    lib, func_name = indicator_name.split('.')
    return getattr(ta_map[lib], func_name)


def create_numeric_parameter(
    type: str, low: Any, high: Any, default: Any, space: str
) -> NumericParameter:
    if type == 'int':
        parameter = IntParameter(low, high, default=default, space=space)
    elif type == 'decimal':
        parameter = DecimalParameter(low, high, default=default, space=space)
    else:
        parameter = RealParameter(low, high, default=default, space=space)
    return parameter


# endregion
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
        indicators_ = {}
        try:
            load_indicators = yaml.load(
                Path(__file__).parent.joinpath('indicators.yml').open(),
                Loader=yaml.FullLoader,
            )
        except Exception as e:
            print(e)
            raise

        for indicator_name, indicator_dict in load_indicators.items():
            # make sure each indicator has an ID
            inf_timeframes = indicator_dict.pop('inf_timeframes', [])
            periods = indicator_dict.pop('periods', [])
            indicators_[indicator_name] = Indicator(
                **indicator_dict, name=indicator_name
            )
            columns = indicator_dict.pop('columns', [])
            for tf in inf_timeframes:
                # make an indicator for each timeframe
                name = f'{indicator_name}_{tf}'
                new_columns = [f'{c}_{tf}' for c in columns]
                indicators_[name] = InformativeIndicator(
                    name=name,
                    timeframe=tf,
                    columns=new_columns,
                    **indicator_dict,
                )
            # for period in periods:
            #     # make an indicator for each period
            #     name = f'{indicator_name}_{period}'
            #     column_append = f'_{period}'
            #     indicators_[name] = Indicator(
            #         name=name,
            #         column_append=column_append,
            #         **indicator_dict,
            #     )

        return indicators_

    @staticmethod
    def get_max_columns():
        """
        Get the maximum number of columns for any indicator
        """
        n_columns_list = [len(indicator.columns) for indicator in indicators.values()]
        return max(n_columns_list)

    @staticmethod
    def get_max_compare_series():
        """
        Get the maximum number of series for any indicator
        """
        len_series = [
            len(indicator.compare.values) for indicator in indicators.values()
        ]
        return max(len_series)

    @staticmethod
    def get_non_value_type_series():
        """
        Get the non value type series
        """
        non_value_type_indicators = []
        for indicator in indicators.values():
            if indicator.compare.type != 'value':
                non_value_type_indicators.append(indicator)
        # get each column from each non value type series
        non_value_type_columns = [
            [c for c in indicator.all_columns]
            for indicator in non_value_type_indicators
        ]
        # flatten non value type columns
        non_value_type_columns = [
            column for sublist in non_value_type_columns for column in sublist
        ]
        return list(set(non_value_type_columns))

    @classmethod
    def get_all_columns(cls):
        column_list = set()
        # combine all indicator.proper_columns with each inf_timeframe
        # iterate through all indicators
        for indicator in indicators.values():
            # iterate through all formatted_columns
            column_list.update(indicator.all_columns)
        return list(column_list)

    @classmethod
    def create_series_indicator_map(cls):
        # create a map of series to indicator
        series_indicator_map = {}
        for indicator in indicators.values():
            for column in indicator.all_columns:
                series_indicator_map[column] = indicator
        return series_indicator_map


class InvalidSeriesError(Exception):
    pass


# region models


@dataclass(frozen=True)
class FuncField:
    loc: str
    columns: Optional[list[str]] = Field(default_factory=list)
    args: Optional[list] = Field(default_factory=list)
    kwargs: Optional[dict] = Field(default_factory=dict)


@dataclass(frozen=True)
class CompareField:
    type: str
    values: list[str] = Field(default_factory=list)


HighLowSpace = namedtuple('space', ['low', 'high'])


class OptimizeField(BaseModel):
    type: str
    defaults: dict


class ValueOptimizeField(OptimizeField):
    buy: HighLowSpace = None
    sell: HighLowSpace = None
    p_type: str = None
    """Parameter type. int, real or decimal"""

    def get(self, buy_or_sell: str) -> NumericParameter:
        if not self.p_type:
            raise ValueError('Parameter type not set')
        if buy_or_sell == 'buy':
            space = self.buy
        else:
            space = self.sell
        parameter = None
        if self.p_type == 'int':
            parameter = IntParameter(
                *space, default=self.defaults[buy_or_sell], space=buy_or_sell
            )
        if self.p_type == 'decimal':
            parameter = DecimalParameter(
                *space, default=self.defaults[buy_or_sell], space=buy_or_sell
            )
        if self.p_type == 'real':
            parameter = RealParameter(
                *space, default=self.defaults[buy_or_sell], space=buy_or_sell
            )
        return parameter


class Value(BaseModel):
    column: str
    buy_or_sell: str
    parameter_type: str
    default: Union[int, float]
    low: Union[int, float]
    high: Union[int, float]

    @property
    def parameter(self) -> NumericParameter:
        return create_numeric_parameter(
            self.parameter_type, self.low, self.high, self.default, self.buy_or_sell
        )

    @classmethod
    def from_string(cls, string, column_append: str):
        """
        Parse string to Value object
        """
        try:
            col, bs, p_type, default, low, high = string.split(',')
        except ValueError:
            raise ValueError(f'Invalid string: {string}')
        default = convert_to_number(default)
        low = convert_to_number(low)
        high = convert_to_number(high)
        return cls(
            column=col + column_append,
            buy_or_sell=bs,
            parameter_type=p_type,
            default=default,
            low=low,
            high=high,
        )


class PeriodOptimizeField(OptimizeField):
    periods: HighLowSpace


class Indicator(BaseModel):
    class Config:
        frozen = True

    func: FuncField
    values: Union[list, dict] = {}
    columns: Optional[list[str]] = []
    compare: CompareField
    optimize_dict: dict = Field(
        default_factory=dict,
        alias='optimize',
    )
    # sell_op: Optional[str]
    # buy_op: Optional[str]
    type: str
    column_append: str = ''
    compare_to: list[str] = Field(default_factory=list)
    inf_timeframes: list[str] = Field(default_factory=list)
    name: str
    timeframe: str = None

    @property
    def formatted_columns(self):
        return [c + self.column_append for c in self.columns]

    @property
    def parsed_values(self) -> dict[str, dict[str, Value]]:
        values = {'buy': {}, 'sell': {}}
        for val_line in self.values:
            try:
                val = Value.from_string(val_line, self.column_append)
            except ValueError:
                raise ValueError(
                    f'Invalid value line: {val_line} for indicator: {self.name}'
                )
            values[val.buy_or_sell][val.column] = val
        return values

    @property
    def all_columns(self):
        return self.formatted_columns

    @property
    def is_informative(self):
        return isinstance(self, InformativeIndicator)

    def get_parameters(self):
        p_map = {'buy': {}, 'sell': {}}
        if self.compare.type == 'value':
            for bs, values in self.parsed_values.items():
                for column, value in values.items():
                    parameter_name = f'{self.name}__{column}__{bs}_value'
                    p_map[bs][parameter_name] = value.parameter

        return p_map

    def get_value(self, column: str, buy_or_sell: str) -> Value:
        try:
            return self.parsed_values[buy_or_sell][column]
        except KeyError:
            raise ValueError(
                f'Value for column: {column} and buy_or_sell: {buy_or_sell} not found'
                f'\nparsed_values: {self.parsed_values}'
            )

    def populate(self, dataframe: DataFrame):
        func = load_function(self.func.loc)
        columns = self.func.columns
        loaded_args = self.func.args
        kwargs = self.func.kwargs
        args = []
        if columns:
            for c in columns:
                args.append(dataframe[c])
        else:
            args.append(dataframe)
        if loaded_args:
            args.extend(loaded_args)

        try:
            result = func(*args, **kwargs)
        except TypeError:
            raise TypeError(
                f'Invalid arguments for function: {func} for indicator: {self.name}\n'
                f'Args: {loaded_args}, Columns: {columns}, Kwargs: {kwargs}'
            )
        if isinstance(result, DataFrame):
            for c in self.columns:
                dataframe[c + self.column_append] = result[c]
        else:
            # split so that "1h" is not displayed twice
            dataframe[self.formatted_columns[0]] = result

        return dataframe

    def get_func_result(self, dataframe):
        func = load_function(self.func.loc)
        columns = self.func.columns
        loaded_args = self.func.args
        kwargs = self.func.kwargs
        args = []
        if columns:
            for c in columns:
                args.append(dataframe[c])
        else:
            args.append(dataframe)
        if loaded_args:
            args.extend(loaded_args)
        result = func(*args, **kwargs)
        return result


class InformativeIndicator(Indicator):
    timeframe: str

    @property
    def columns_stripped_of_tf(self):
        return ['_'.join(c.split('_')[:-1:]) for c in self.columns]

    @property
    def parsed_values(self) -> dict[str, dict[str, Value]]:
        values = {'buy': {}, 'sell': {}}
        for val_line in self.values:
            try:
                append = self.column_append + '_' + self.timeframe
                val = Value.from_string(val_line, append)
            except ValueError:
                raise ValueError(
                    f'Invalid value line: {val_line} for indicator: {self.name}'
                )
            values[val.buy_or_sell][val.column] = val
        return values

    def populate(self, informative: DataFrame):
        result = self.get_func_result(informative)
        if isinstance(result, DataFrame):
            for c in self.columns_stripped_of_tf:
                informative[c + self.column_append] = result[c]
        else:
            # split so that "1h" is not displayed twice
            informative[self.columns_stripped_of_tf[0] + self.column_append] = result

        return informative


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
    """Abstract"""

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
        indicator = series_map[series_name]
        if indicator.compare.type == 'none':
            raise InvalidSeriesError()
        series1 = IndicatorSeries(series_name)
        # return comparison based on the compare type
        if series_name == 'none':
            raise InvalidSeriesError()
        if indicator.compare.type == 'value':
            return ValueComparison(series1, op_str)
        if indicator.compare.type == 'specific':
            # if there is more than on value in indicator.compare.values and the comparison
            # series is not in the list of values, raise invalid
            if (
                len(indicator.compare.values) > 1
                and comparison_series_name not in indicator.compare.values
            ):
                raise InvalidSeriesError()
            # set the comparison series name to the first indicator.compare.values
            comparison_series_name = indicator.compare.values[0]
        # create OhlcSeries if the comparison series is in ohlc
        if comparison_series_name in ['open', 'high', 'low', 'close']:
            comparison_series = OhlcSeries(comparison_series_name)
        else:
            comparison_series = IndicatorSeries(comparison_series_name)
        return SeriesComparison(series1, op_str, comparison_series)


@dataclass(frozen=True)
class SeriesComparison(Comparison):
    series1: IndicatorSeries
    op: str
    series2: Union[IndicatorSeries, OhlcSeries]

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
    series1: IndicatorSeries
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
            value = self.indicator.get_value(self.series1.series_name, bs).default
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
            if indicator.timeframe:
                timeframes.add(indicator.timeframe)
        return timeframes

    @staticmethod
    def compare(dataframe: DataFrame, comparison: Comparison, type_: str) -> Series:
        logger.info('Comparing %s', comparison.name)
        # # check is series1 has been populated
        # if pair not in comparison.series1.indicator.populated:
        #     dataframe = comparison.series1.indicator.populate(
        #         dataframe, comparison.series1.series_name, pair
        #     )
        # # check if series2 has been populated
        # if (
        #     isinstance(comparison, SeriesComparison)
        #     and isinstance(comparison.series2, IndicatorSeries)
        #     and pair not in comparison.series2.indicator.populated
        # ):
        #     dataframe = comparison.series2.indicator.populate(
        #         dataframe, comparison.series2.series_name, pair
        #     )
        # compare
        return comparison.compare(dataframe, type_)

    def create_comparison_groups(
        self, type_, n_groups: int = None
    ) -> dict[int, dict[str, CategoricalParameter]]:
        """
        Create comparison groups for all indicators
        Args:
            type_: buy or sell
            n_groups: number of comparison groups to create
        Returns: A dictionary of comparison groups
        """
        logger.info('creating group parameters')

        comparison_groups = {}

        all_indicators = IndicatorTools.get_all_columns() + ['none']
        series = IndicatorTools.get_non_value_type_series() + [
            'open',
            'close',
            'high',
            'low',
        ]
        for i in range(1, (n_groups or self.n_permutations) + 1):
            group = {
                'series': CategoricalParameter(
                    all_indicators,
                    default=all_indicators[0],
                    space=type_,
                ),
                'operator': CategoricalParameter(
                    list(op_map.keys()), default=list(op_map.keys())[0], space=type_
                ),
                'comparison_series': CategoricalParameter(
                    series,
                    default=series[0],
                    space=type_,
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
            except InvalidSeriesError:
                continue
            conditions.append(self.compare(dataframe, comparison, bs))
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

    def create_local_parameters(self, locals_: dict, num_buy=None, num_sell=None):
        """
        Create the local parameters for the strategy using locals()[x]=y. If num_buy or num_sell
        is 0, then the parameter is not created.
        Args:
            locals_: the local parameters
            num_buy: the number of buy conditions
            num_sell: the number of sell conditions

        Returns: The buy and sell comparison parameters
        """
        buy_comparisons, sell_comparisons = {}, {}
        if num_buy:
            buy_comparisons = self.create_comparison_groups('buy', num_buy)
            for n_group, p_map in buy_comparisons.items():
                for p_name, parameter in p_map.items():
                    locals_[f'buy_{p_name}_{n_group}'] = parameter
        if num_sell:
            sell_comparisons = self.create_comparison_groups('sell', num_sell)
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


indicators = IndicatorTools.load_indicators()
series_map = IndicatorTools.create_series_indicator_map()
if __name__ == '__main__':
    # print(load_function('tta.ATR'))
    # pprint(
    #     set(
    #         IndicatorOptHelper()
    #         .indicators['ema']
    #         .populate(pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']))
    #     )
    # )
    # print(load_function('tta.WMA'))
    # Console().print(IndicatorOptHelper().indicators)

    # permutes = IndicatorOptHelper().comparisons
    # permutes_3 = [p for p in list(product(permutes.items(), repeat=2)) if p[0] != p[1]]
    # Console().print(len(permutes), permutes.keys())
    # Console().print(IndicatorOptHelper().get_parameter('buy'))
    # # objects = list(IndicatorOptHelper().get_combinations())
    # # Console().print(objects, len(objects))
    # # print(IndicatorOptHelper().comparisons['WMA < high'])
    # # print(IndicatorOptHelper().combinations[1])
    # indicator_helper = IndicatorOptHelper.get()
    # combinations = indicator_helper.combinations
    # print(len(combinations))
    # pprint(combinations[4818325])
    # pprint(IndicatorOptHelper().comparisons)
    # print(IndicatorTools.get_all_indicators())
    # print(IndicatorTools.get_non_value_type_series())
    # ct = CombinationTester(
    #     {
    #         "buy_comparison_series_1": "EMA_100",
    #         "buy_comparison_series_2": "T3Average_1h",
    #         "buy_comparison_series_3": "EMA",
    #         "buy_operator_1": "<",
    #         "buy_operator_2": ">=",
    #         "buy_operator_3": "<=",
    #         "buy_series_1": "ewo",
    #         "buy_series_2": "bb_middleband_1h",
    #         "buy_series_3": "T3Average",
    #     },
    #     {
    #         "sell_comparison_series_1": "sar_1h",
    #         "sell_comparison_series_2": "bb_middleband_40",
    #         "sell_operator_1": ">",
    #         "sell_operator_2": ">=",
    #         "sell_series_1": "stoch80_sma10",
    #         "sell_series_2": "T3Average",
    #     },
    # )
    # print(ct.buy_comparisons)
    # print(ct.sell_comparisons)
    # print(indicators['stoch_sma'].get_parameters())
    print(indicators['cci'].parsed_values)
    print(indicators['cci'].get_parameters())
    print(indicators['rsi'].get_parameters())
    [print(n, f'{i}\n') for n, i in indicators.items()]
    # pprint(IndicatorOptHelper.get().create_parameters('buy'))
    ...
    # print(convert_to_number('5.5'))
