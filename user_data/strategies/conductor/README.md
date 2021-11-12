
# Conductor
## Why
### The Problem
Many times I've thought to myself: I wish I can combine many low trading, high profit-per-trade strategies.

Do you have a strategy that has the accuracy of world-class sniper, but barely takes shots? What happens when you combine 2 of them? 3? 4? You see what I'm getting at...

### The Solution
With the Conductor Strategy you can utilize the strengths of many different strategies in one! You no longer have to find that _one_ strategy that does it all. You no longer need to sacrifice accuracy (Average profit per trade) for volume (# of trades).

Running multiple low volume, high accuracy strategies brings volume (and value) back to these more patient snipers.

Could this be the solution to our bear market problem? For me, it's starting to feel that way. Check out my live stats below.

## Instructions
Simply fill `STRATEGIES = []` in conductor.py with the strategies you want to use

Alternatively, you can create an ensemble.json and fill in the strategies there.
```json
["Strategy1", "Strategy2"]
```
Make sure these are the class names of the strategy and not the file names.

## How it works
1. Every strategy that you add to the conductor can generate a buy signal.
2. When a buy signal occurs, the strategies that decided to buy the pair will be added to the buy_tag of the trade.
3. A trade will only honor the sell signal of a strategy that generated its buy signal.
4. More than one strategy can generate a buy signal at once, therefore all of their sell-signals will be honored.
5. Each strategy will have all of their features available to them when generating signals: indicators, informative indicators, roi, stoploss, trailing, custom_sell, custom_stoploss.

## Results from a live session
![](https://media.discordapp.net/attachments/908130693475868774/908131766978609282/unknown.png?width=1440&height=408)
![](https://media.discordapp.net/attachments/908130693475868774/908132187847684106/unknown.png?width=1440&height=495)

## Backtest result
See [Conductor_backtest_result.txt ](https://github.com/raph92/freqtrade-strategies/blob/master/user_data/strategies/Conductor_backtest_result.txt)
