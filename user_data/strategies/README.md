# Indicator Mix
Indicator Mix is my fully automated strategy generator that helps you create strategies using predefined indicators located in indicators.yml.

## Table of Contents
- [Indicator Mix](#indicator-mix)
  * [Indicators](#indicators)
    + [Types of indicators](#types-of-indicators)
    + [Defining Indicators](#defining-indicators)
      - [ewo:](#ewo-)
      - [func:](#func-)
        * [loc: "ci.EWO"](#loc---ciewo-)
      - [columns:](#columns-)
      - [compare:](#compare-)
      - [values:](#values-)
        * ["ewo,buy,decimal,0,-2.0,2.0"](#-ewo-buy-decimal-0--20-20-)
      - [type:](#type-)
      - [inf_timeframes:](#inf-timeframes-)
  * [indicator_mix](#indicator-mix)
    + [Hyperoptable Parameters](#hyperoptable-parameters)
    + [Comparison](#comparison)
    + [Comparison Groups](#comparison-groups)
    + [Multiple Comparison Groups](#multiple-comparison-groups)
      - [If n_per_group is 1](#if-n-per-group-is-1)
      - [If n_per_group is 2 or more](#if-n-per-group-is-2-or-more)
    + [Applying a strategy](#applying-a-strategy)
  * [Advanced Hyper-optimization](#advanced-hyper-optimization)
    + [im_test.py](#im-testpy)
      - [Parameters](#parameters)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## Indicators
### Types of indicators
There are two types of indicators supported at the moment:
1. **Value-based** - these are indicators that are compared with very specific numbers/values. For example: RSI. RSI will be compared to some number between 0 and 100 since those are its value ranges. 
2. **Series-based** - these are indicators that will be compared with another indicator. For example: EMA can be compared with SMA, WMA, TEMA, etc. These can also be compared with ohlc series.
### Defining Indicators
Here is an example of how an EWO oscillator indicator would be defined in `indicators.yml`:
```yml
ewo:
  func:
    loc: "ci.EWO"
    columns: null
    kwargs:
      ema_length: 5
  columns:
    - "ewo"
  compare:
    type: "value"
  values:
    - ewo,buy,decimal,0,-2.0,2.0
    - ewo,sell,decimal,0,-2.0,2.0
  type: "osc"
  inf_timeframes:
    - "1h"
```
#### ewo:
This is just the identifier for the indicator. This will not show up in the hyperopt output however.

#### func:
this is where we define the location of the indicators function and what variables to pass.

##### loc: "ci.EWO"
`"ci.EWO"` is essentially "`{library_shortname}`.`{function_name}`" that defines where the function is. 

`ci` stands for custom_indicators. A map of all the function names is found in `ta_map`  in `indicator_opt.py`
```python
# tta = talib, pta = pandas_ta, qta = qtpylib, fta = finta, ci = custom_indicators.py
ta_map = {'tta': tta, 'pta': pta, 'qta': qta, 'fta': fta, 'ci': ci}
```
`kwargs` is a dict of keyword arguments to pass to the function `def EWO(dataframe, ema_length=5, ema2_length=35)` in `custom_indicators.py`.
#### columns:
The names in the list are the indicator series that are expected to be added from the TA function.
In our example, the column name is **"ewo"**

For `RSI`: 1 column is expected to be added called "rsi"

For `BollingerBands`: 3 columns are expected to be added: the lower, middle, and upper bands. Example:
```yml
  columns:
    - "bb_lowerband"
    - "bb_middleband"
    - "bb_upperband"
```
These columns must match the functions. At least one column has to be specified per indicator.
#### compare:
`type`: This value determines whether the indicator will be compared to a series or a custom value. For `EWO`, this 
will be "value".

Accepted values are: `value` or `series`
#### values:
##### "ewo,buy,decimal,0,-2.0,2.0"
These values define the parameters that will be hyper-opted from `im_tests.py`. This will not affect 
hyperopts with `indicator_mix.py`

The syntax is: `{column_name},{buy_or_sell_space},{parameter_type},{default_value},{low},{high}`
#### type:
This value is current not being used and will be for future updates.
#### inf_timeframes:
This will be a list of extra informative timeframes you want to generate indicators for.

In our example: with `1h`, an `ewo_1h` indicator will be generated.

For BollingerBands each of the low, middle, and upper bands will have additional `_1h` counterparts. 
## indicator_mix
`indicator_mix.py` is what we will pass to the hyperopt command. 
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
Then each indicator will be divided by `n_per_group`. 
So if you specify `num_buy=4` and `buy_n_per_group=2`, then you will have 2 separate buy comparison groups that will generate a signal. 

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
        "buy_operator_1": "crossed_above",
        "buy_operator_2": ">=",
        "buy_operator_3": "crossed_above",
        "buy_series_1": "SMA_200",
        "buy_series_2": "bb_upperband_1d",
        "buy_series_3": "EMA",
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_comparison_series_1": "low",
        "sell_comparison_series_2": "TEMA_1h",
        "sell_operator_1": "crossed_below",
        "sell_operator_2": "<=",
        "sell_series_1": "bb_upperband_1h",
        "sell_series_2": "bb_upperband_40",
    }
```
Now you can elect to only opt the buy or sell space to further optimize the strategy.
## Advanced Hyper-optimization
So we've found our strategy. But it's "okay". It is very general because it is using default values. What if we 
could further optimize our strategy so that it is tailored for our specific set of pairs? 

Here is where im_test.py comes in.
### im_test.py
In `im_test.py` we will be using CombinationTester to automatically hyperopt different values for our chosen strategy.
We start by defining our comparison groups and initializing CombinationTester.
```python
# Buy hyperspace params:
buy_params = {
    "buy_comparison_series_1": "close",
    "buy_comparison_series_2": "T3Average_1h",
    "buy_operator_1": "<=",
    "buy_operator_2": ">=",
    "buy_series_1": "rsi",
    "buy_series_2": "rsi_1h",
}
# Sell hyperspace params:
sell_params = {
    "sell_comparison_series_1": "sar_1h",
    "sell_operator_1": ">",
    "sell_series_1": "stoch80_sma10",
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

This will create optable parameters such as `rsi__rsi__buy_value` and `stoch_sma__stoch80_sma10__sell_value`.
You can then hyperopt and thus optimize your strategy.

**Make sure you set the same `n_per_group` that you had in the indicator_mix, otherwise your results may differ.**
