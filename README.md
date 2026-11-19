[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/farcasteralpha/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/farcasteralpha/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Leonida Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
Exophase at 2026-11-26T00:00:00Z — one week after launch — as five
quantiles per trophy, from the house's reads of the trophy page up to the
case checkpoint. Entered on [make.fun](https://make.fun) for the
$250 gta6-trophy-rarity-forecast-01 bounty.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every
`*.in` case in `$MAKEFUN_INPUT_DIR` (or `/input`) and writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (or `/output`).

The key observation is that Exophase refreshes its counts in discrete
jumps: when the tracked-player count jumps, every rate is diluted at
once, then recovers as the new players play. So the forecaster works in
logit space and decomposes each trophy into

```
earned(t) = rate(t) × players_tracked(t)
```

It fits recency-weighted lines of logit(rate), log(earned) and
log(players) against log time since the release day, projects all three
to the target, and takes the rate projection (earned / players)
cross-checked against the direct rate trend as the median. The spread
combines the house random walk, the fit's residual, a dilution term
from the volatility of players_tracked, and a 0.5-logit floor for one
discrete refresh — with the spread multiplier calibrated by
walk-forward backtest on the case's own history, so a flat history
falls back to a house-style forecast instead of an overconfident one.

The program is deterministic: two runs on the same case write
byte-identical output.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` — the whole forecaster, one file.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
