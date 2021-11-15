from enum import Enum
from typing import Callable, Union

import numpy as np
from freqtrade.strategy.hyper import BaseParameter, DecimalParameter, IntParameter
from pandas import DataFrame, Series
import logging

logger = logging.getLogger()


class IndicatorType(Enum):
    VALUE = 'value'
    SERIES = 'series'
    SPECIAL = 'special'
    SPECIFIC = 'specific'


class Indicator:
    type = ''

    def __init__(
        self,
        func: Callable,
        columns: list[str],
        func_columns=None,
        inf_timeframes=None,
        name='',
        **kwargs,
    ) -> None:
        self.func = func
        self.func_columns = func_columns or []
        self.columns = columns
        self.inf_timeframes = inf_timeframes or []
        self.optimize_dict = {k: v for k, v in kwargs.items() if 'optimize' in k}
        self.name = name
        self.informative = False

    @property
    def function_parameters(self):
        """
        Returns a dictionary of the parameters for the function
        All function parameters are in the optimize_dict and start with `optimize_func`
        Each key in the return value will have self.name prepended to it.
        The structure of each key is: optimize_func__arg.
        We will only return the arg portion of the key
        """
        return {
            f'{self.name}__{k.split("__", 1)[1]}': v
            for k, v in self.optimize_dict.items()
            if 'optimize_func' in k
        }

    @property
    def value_parameters(self):
        """
        Returns a dictionary of the parameters for the function
        All function parameters are in the optimize_dict and start with `optimize_value`.
        Each key in the return value will have self.name prepended to it.
        The structure of each value is: optimize_value__col__buy_or_sell.
        We will only return the col__buy_or_sell portion of the key
        """
        return {
            f'{self.name}__{k.split("__", 1)[1]}': v
            for k, v in self.optimize_dict.items()
            if 'optimize_value' in k
        }

    @property
    def formatted_columns(self):
        return [f'{self.name}__{c}' for c in self.columns]

    @property
    def all_columns(self):
        return self.formatted_columns

    def set_name(self, name: str):
        self.name = name

    def get_value(self, column: str, buy_or_sell: str):
        """
        Returns the value of the indicator for the given column and buy or sell from the value_parameters
        """
        name = f'{column}__{buy_or_sell}'
        try:
            return self.value_parameters[name]
        except KeyError:
            # raise an error and inform user of available keys
            raise KeyError(
                f'{name} is not a valid key for {self.name}. '
                f'Available keys are: {list(self.value_parameters.keys())}'
            )

    def get_default_timeperiod(self):
        """
        1. Grab the timeperiod from the function_parameters. The syntax is self.name__timeperiod.
        2. The timeperiod is an IntParameter, we want to return the value.
        3. If it does not exist, return None
        """
        timeperiod = self.function_parameters.get(f'{self.name}__timeperiod')
        if timeperiod:
            return timeperiod.value
        return None

    def get_name_of_func_parameter_args(self):
        """
        Returns a list of the names of the args for the function
        """
        return [k.split("__", 1)[1] for k, v in self.function_parameters.items()]

    def get_map_of_func_parameters(
        self,
    ) -> dict[str, Union[IntParameter, DecimalParameter]]:
        """
        Returns a dictionary of the parameters for each parameter in function parameters.
        The structure of each key is: name__arg.
        We will only return the arg portion of the key and its parameter mapped to it.
        """
        return {
            f'{k.split("__", 1)[1]}': v for k, v in self.function_parameters.items()
        }

    def get_map_of_func_parameters_to_values(
        self,
    ) -> dict[str, Union[float, int]]:
        """
        Returns a dictionary of the parameters for each parameter in function parameters
        The structure of each key is: name__arg.
        We will only return the arg portion of the key and it's value mapped to it.
        """
        return {
            f'{k.split("__", 1)[1]}': v.value
            for k, v in self.function_parameters.items()
        }

    def get_function_arguments(self, dataframe):
        args = []
        if self.func_columns:
            for column in self.func_columns:
                args.append(dataframe[column])
        else:
            args.append(dataframe.copy())
        return args

    def populate(self, dataframe: DataFrame) -> DataFrame:
        """
        Populates the dataframe with the indicator function by running self.func

        1. We will pass the args and kwargs from get_func_args_and_kwargs to the function and get the return value as
        func_result.

        2. We will then populate the dataframe with the func_result as follows: If the return value
        is a Series or numpy.ndarray: we want to set our Series name to the name of the string in our
        self.columns list.

        If the return value is a DataFrame: we want iterate through the columns in our self.columns list
        and grab that string from the func_result and add it our DataFrame.
        """
        args = self.get_function_arguments(dataframe)
        kwargs = self.get_map_of_func_parameters_to_values()
        dataframe = self._execute_func(dataframe, args, kwargs)
        return dataframe

    def _execute_func(self, dataframe, args, kwargs, append=''):
        try:
            func_result = self.func(*args, **kwargs)
        except Exception as e:
            # show debug information
            logger.error(
                f'{self.name} failed to run with args: {args} and kwargs: {kwargs}'
            )
            logger.error('Func parameters: %s', self.function_parameters)
            logger.error('Optimize dict: %s', self.optimize_dict)
            raise e
        if isinstance(func_result, (Series, np.ndarray)):
            dataframe[self.formatted_columns[0] + append] = func_result
        elif isinstance(func_result, DataFrame):
            for idx, _ in enumerate(self.columns):
                try:
                    dataframe[self.formatted_columns[idx] + append] = func_result[
                        self.columns[idx]
                    ]
                except KeyError:
                    # raise error and let the user know what the func_result columns are
                    raise KeyError(
                        f'The function for {self.name} returned a DataFrame with the '
                        f'columns {func_result.columns}'
                        f' but the columns {self.columns} were expected.'
                    )
        else:
            raise ValueError(
                f'{self.name} returned a value of type {type(func_result)}'
                f' which is not supported. Please return a Series or DataFrame.'
            )
        return dataframe

    def populate_with_ranges(self, dataframe: DataFrame):
        """
        Populate the dataframe with the ranges of each function_parameter's range.
        1. Get args and kwargs from get_func_args_and_kwargs with ranges=True
        2. Run self.func once for each value in the range of each parameter
        """
        args = self.get_function_arguments(dataframe)
        kwargs = self.get_map_of_func_parameters()
        for k, v in kwargs.items():
            if not v.optimize:
                continue
            # v.in_space = True
            for value in v.range:
                kwargs = kwargs.copy()
                # the accompanying parameters in kwargs will be of type BaseParameter, we need to
                # set those to their values using parameter.value
                kwargs[k] = value
                for key, parameter in kwargs.items():
                    if isinstance(parameter, BaseParameter):
                        kwargs[key] = parameter.value
                dataframe = self._execute_func(
                    dataframe, args, kwargs, append='_' + str(value)
                )
        return dataframe

    def create_informatives(self) -> dict[str, 'InformativeIndicator']:
        """
        Create an informative indicator for each timerange in inf_timeframes.
        Each key will be self.name plus the timerange appended to it
        """
        inf_indicators = {}
        for timeframe in self.inf_timeframes:
            name___timeframe = self.name + '_' + timeframe
            dict__ = self.__dict__.copy()
            dict__['name'] = name___timeframe
            try:
                inf_indicators[name___timeframe] = InformativeIndicator(
                    type=self.type,
                    timeframe=timeframe,
                    **dict__,
                )
            except Exception:
                logger.error(f'Error creating InformativeIndicator {name___timeframe}')
                return {}
        return inf_indicators

    def __str__(self):
        return f'{self.__class__.__name__} - {self.name}'

    def __repr__(self):
        return f'{self.__class__.__name__} - {self.name}'


class ValueIndicator(Indicator):
    type = IndicatorType.VALUE

    def __init__(self, value=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value


class SeriesIndicator(Indicator):
    type = IndicatorType.SERIES


class SpecialIndicator(Indicator):
    """These indicators will only be compared to indicators created by its TA function"""

    type = IndicatorType.SPECIAL

    def __init__(self, compare: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.compare = compare

    @property
    def formatted_compare(self):
        """
        Return a dict of the compare dict with the keys formatted to the
        format of the columns in the dataframe.
        """
        return {f'{self.name}__{k}': f'{self.name}__{v}' for k, v in self.compare.items()}


class SpecificIndicator(Indicator):
    """These indicators will only be compared to other specified indicators"""

    type = IndicatorType.SPECIFIC

    def __init__(self, compare: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.compare = compare


class InformativeIndicator(Indicator):
    def __init__(
        self, type: str, timeframe: str, optimize_dict: dict, **kwargs
    ) -> None:
        super().__init__(**kwargs, optimize_dict={})
        self.timeframe = timeframe
        self.optimize_dict = optimize_dict
        self.type = type
        self.column_append = '_' + timeframe
        self.informative = True

    def get_value(self, column: str, buy_or_sell: str):
        """
        Returns the value of the indicator for the given column and buy or sell from the value_parameters
        """
        name = f'{column}__{buy_or_sell}{self.column_append}'
        try:
            return self.value_parameters[name]
        except KeyError:
            # raise an error and inform user of available keys
            raise KeyError(
                f'{name} is not a valid key for {self.name}. '
                f'Available keys are: {list(self.value_parameters.keys())}'
            )

    @property
    def value_parameters(self):
        """
        Returns a dictionary of the parameters for the function
        All function parameters are in the optimize_dict and start with `optimize_value`.
        Each key in the return value will have self.name prepended to it.
        The structure of each value is: optimize_value__col__buy_or_sell.
        We will only return the col__buy_or_sell portion of the key
        """
        return {
            f'{self.name}__{k.split("__", 1)[1]}{self.column_append}': v
            for k, v in self.optimize_dict.items()
            if 'optimize_value' in k
        }

    # @property
    # def formatted_columns(self):
    #     """
    #     Return each column with the timeframe appended.
    #     """
    #     return [f'{self.name}__{col}' for col in self.columns]
