# Indicator Mix

Indicator Mix is my fully automated strategy generator that helps you create strategies using predefined indicators
located in indicators.yml.

If you are running this from Docker, make sure you have the `finta`, `pandas_ta`, and the `scikit-optimize` libraries
installed in the container.

I've included a [Dockerfile](https://github.com/raph92/freqtrade-strategies/blob/master/user_data/strategies/indicatormix/Dockerfile)


## Table of Contents

- [Indicator Mix](#indicator-mix)
    * [Getting Started](#getting-started)
    * [Indicators](#indicators)
        + [Types of indicators](#types-of-indicators)
        + [Defining Indicators](#defining-indicators)
    * [Hyperopting](#hyperopting)
        + [Hyperoptable Parameters](#hyperoptable-parameters)
        + [Comparison](#comparison)
        + [Comparison Groups](#comparison-groups)
        + [Multiple Comparison Groups](#multiple-comparison-groups)
        + [Applying a strategy](#applying-a-strategy)
    * [~~Advanced Hyper-optimization~~](#advanced-hyper-optimization)
        + [im_test.py](#im-testpy)
            - [Parameters](#parameters)
    * [Todo](#todo)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with
markdown-toc</a></i></small>

## Getting Started
Simply download `user_data/strategies/indicator_mix.py` and the entire `indicatormix` folder to `user_data/strategies`.

Your `user_data/strategies` folder should look close to the following:
```
// user_data/strategies
├── im_test.py
├── indicatormix
│   ├── entities
│   │   ├── indicator.py
│   │   └── __init__.py
│   ├── helpers
│   │   └── custom_indicators.py
│   ├── indicator_opt.py
│   ├── indicators.py
├── indicator_mix.py
```
## Docker
For docker make sure you install the following packages: 

`pip install --user finta pandas_ta scikit-optimize`
## Indicators

### Types of indicators

There are two types of indicators supported at the moment:

1. **ValueIndicator** - these are indicators that are compared with very specific numbers/values. For example: RSI. RSI
   will be compared to some number between 0 and 100 since those are its value ranges.
2. **SeriesIndicator** - these are indicators that will be compared with another indicator. For example: EMA can be
   compared with SMA, WMA, TEMA, etc. These can also be compared with ohlc series.
3. **SpecialIndicator** - these are indicators that are comparable with predefined indicators. For example: MACD. 
   MACD is comparable with MACD_signal.
### Defining Indicators

Here is an example of how an EWO oscillator indicator would be defined in `indicators.py`:

```python
VALUE_INDICATORS = {
    ...,
    "cci": ind.ValueIndicator(
        func=lambda df, timeperiod: pta.cci(df, length=timeperiod),
        func_columns=['high', 'low', 'close'],
        columns=['cci'],
        optimize_func__timeperiod=IntParameter(10, 25, default=14, space='buy'),
        optimize_value__cci__buy=IntParameter(50, 150, default=100, space='buy'),
        optimize_value__cci__sell=IntParameter(-150, -50, default=-100, space='sell'),
        inf_timeframes=['1h', '30m'],
    ),
           ...
}
```
Let's go over the example line by line:
#### "cci": ind.ValueIndicator
This is just the identifier for the indicator. This will not show up in the hyperopt output.
The specific class one be one of `ValueIndicator`, `SeriesIndicator`, or `SpecialIndicator`.
#### func
this is where we define the function that will be used to populate the indicator. We will use lambda when we are 
defining a function that has atypical argument names. Such as: `length` or `window` instead of `timeperiod`. The 
purpose for this is to standardize the function names and make it easier to read/code.
#### func_columns
this is where we define the columns that the function will be applied to. For example, if we want to apply the 
function to the `high` and `low` columns, we would define this as `['high', 'low']` and they will be passed to the 
function.

#### columns
The names in the column list are the indicator series that are expected to be added from the TA function. In our 
example, the column name is **cci**. 

We will use these values to identify and compare the indicators.
The full column name of our example is **cci__cci**. This may look a bit redundant, but this syntax is important for 
informative timeframes. Example: **cci_1h__cci**

In another example, for `BollingerBands` 3 columns are expected to be added: the **lower**, **middle**, and **upper** bands. 
Example:

```python
SERIES_INDICATORS = {
    ...,
    "bb_fast": ind.SeriesIndicator(
        columns=[
            "bb_lowerband", # bb_fast__bb_lowerband
            "bb_middleband", # bb_fast__bb_middleband
            "bb_upperband", # bb_fast__bb_upperband
        ]),
    ...
    }
```

These columns must match what the function returns. At least one column has to be specified per indicator.

#### optimize_func__timeperiod
All keyword arguments that start with `optimize_func` are used to define the parameters that will be used to 
optimize the population indicator. In other words, they will be used in the `IStrategy.populate_indicators` method.
The word after the double underscore will be passed to the func argument along with the value of its parameter.

#### optimize_value__cci__buy and sell
Keyword arguments that start with `optimize_value` are only used in ValueIndicator and are used 
to do advanced optimization. IndicatorMix will not optimize these values.
The syntax is: `optimize_value__{column_name}__{buy_or_sell}`. **column_name** is the name of the column that 
the value will be compared against.

#### inf_timeframes=['1h', '30m']
This will be a list of extra informative timeframes you want to generate indicators for.
The output will be a list of indicators for each timeframe.
For 1h, the new indicator name will be: `cci_1h`. The column will be: `cci_1h__cci`.

For BollingerBands each of the low, middle, and upper bands will have additional `_1h` counterparts.

## Hyperopting

The `IndicatorMix` strategy is found in indicator_mix.py and is what we will pass to the hyperopt command.

### Hyperoptable Parameters

The following code will automatically generate
`CategoricalParameters` for all the indicators in `indicators.yml`, allowing them to be hyperopted:

```python
class IndicatorMix(IStrategy):
    # region Parameters
    iopt = IndicatorOptHelper.get()
    buy_comparisons, sell_comparisons = iopt.create_local_parameters(
        locals(), num_buy=3, num_sell=2
    )
```

### Comparison

A **comparison** is simply the usage of an indicator.

An example of a **comparison** would be: `EMA crossed_above SMA200`.

### Comparison Groups

A **comparison group** would be 2 or more of these **comparisons** working together:

```
EMA crossed_above SMA200, RSI crossed_below 30, & SAR crossed_above WMA
```

### Multiple Comparison Groups

You can instruct the generator to create multiple comparison groups by specifying a ```n_per_group``` variable:

```python
# indicator_mix
buy_n_per_group = 1
sell_n_per_group = 1
```

#### If n_per_group is 1

Then all the comparisons will work together... meaning that all the comparisons have to be True for a signal to be
generated.

#### If n_per_group is 2 or more

Then each indicator will be divided by `n_per_group`. So if you specify `num_buy=4` and `buy_n_per_group=2`, then you
will have 2 separate buy comparison groups that will generate a signal.

Example:
Let's say we have four indicators: A, B, C, D.

```
(A & B) = group 1 (G1)
(C & D) = group 2 (G2)
BUY_SIGNAL = (G1 OR G2)
```

If G1 or G2's conditions are True, then a buy signal is generated.

### Applying a strategy

If you accept a result from your hyperopt, then you can just paste the output into the IndicatorMix like so:

```python
class IndicatorMix(IStrategy)
    # Buy hyperspace params:
    buy_params = {
        "buy_comparison_series_1": "bb_middleband_40",
        "buy_comparison_series_2": "high",
        "buy_comparison_series_3": "T3Average_1h",
        "buy_operator_1":          "crossed_above",
        "buy_operator_2":          ">=",
        "buy_operator_3":          "crossed_above",
        "buy_series_1":            "SMA_200",
        "buy_series_2":            "bb_upperband_1d",
        "buy_series_3":            "EMA",
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_comparison_series_1": "low",
        "sell_comparison_series_2": "TEMA_1h",
        "sell_operator_1":          "crossed_below",
        "sell_operator_2":          "<=",
        "sell_series_1":            "bb_upperband_1h",
        "sell_series_2":            "bb_upperband_40",
    }
```

Now you can elect to only opt the buy or sell space to further optimize the strategy.

## Advanced Hyper-optimization

**_`im_test.py` is not finished yet. Disregard this section for now._**

So we've found our strategy. But it's "okay". It is very general because it is using default values. What if we could
further optimize our strategy so that it is tailored for our specific set of pairs?

Here is where im_test.py comes in.

### im_test.py

In `im_test.py` we will be using CombinationTester to automatically hyperopt different values for our chosen strategy.
We start by defining our comparison groups and initializing CombinationTester.

```python
# Buy hyperspace params:
buy_params = {
    "buy_comparison_series_1": "close",
    "buy_comparison_series_2": "T3Average_1h",
    "buy_operator_1":          "<=",
    "buy_operator_2":          ">=",
    "buy_series_1":            "rsi",
    "buy_series_2":            "rsi_1h",
}
# Sell hyperspace params:
sell_params = {
    "sell_comparison_series_1": "sar_1h",
    "sell_operator_1":          ">",
    "sell_series_1":            "stoch80_sma10",
}
ct = CombinationTester(buy_params, sell_params)
```

#### Parameters

Now we can generate all the hyperoptable values

```python
class IMTest(IStrategy):
    # region Parameters
    ct.update_local_parameters(locals())
    # endregion
```

This will create optable parameters such as `rsi__rsi__buy_value` and `stoch_sma__stoch80_sma10__sell_value`. You can
then hyperopt and thus optimize your strategy.

**Make sure you set the same `n_per_group` that you had in the indicator_mix, otherwise your results may differ.**

## Todo

- [ ] Implement n_per_group for comparisons in im_test.py
- [ ] Add ability to auto-generate different time-periods parameters for indicators

