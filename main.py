#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

One file, Python 3.13 standard library only, no network, no wall clock,
no randomness: two runs on the same case write byte-identical output.

Method (per trophy, in logit space)
-----------------------------------
Exophase refreshes a game's counts on its own discrete schedule, so a
read's rates can jump when the tracked-player count jumps: new players
arrive holding few trophies, which dilutes every rate, and rates then
recover as those players play. A model that only watches the rate is
surprised by every refresh; a model that watches the earned count is
not, because the earned count never falls. So each trophy is decomposed

    earned(t) = rate(t) * players_tracked(t)

and both sides are modelled in log time, x = ln(1 + hours since
2026-11-19T00:00Z):

* a recency-weighted least-squares line of logit(rate) on x, fitted on
  every read, projected to the target;
* a recency-weighted least-squares line of log(earned) on x and one of
  log(players_tracked) on x (the players line is shared by all
  trophies), projected to the target, giving rate = earned / players;
* the median is the average of the two projections in logit space,
  clamped to 2.5 logit of the last read's rate so a steep log-time
  extrapolation cannot run away.

Spread (logit sd) combines the house random walk (0.5 * sqrt(days)),
the residual sd of the rate fit (floor 0.15), a dilution term from the
observed volatility of log players_tracked, and a refresh floor of
0.5 logit sd: one discrete Exophase refresh moves rates by roughly
that much in logit terms, and the practice histories are flat, so
nothing in them can see a refresh coming. A spread multiplier is
calibrated by walk-forward backtest on the case's own history
(forecast each past read several reads ahead from the reads before
it, pick the multiplier on a grid that minimises mean pinball loss), clamped to [1.0, 2.5] so
the backtest can never shrink the spread below the house walk on a
flat history. Flat history / fewer than 3 reads therefore falls back
to the house-style forecast (last rate, random-walk spread) with the
calibrated spread.

Quantiles use the standard normal z values at the case's levels.
All rates are clamped to [0.01, 99.99] before any logit.
"""

import glob
import json
import math
import os
import sys

# Standard normal quantiles at 0.05, 0.25, 0.5, 0.75, 0.95.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

EPOCH = "2026-11-19T00:00:00Z"  # log-time origin: the release day.
LO, HI = 0.01, 99.99            # clamp for rates before the logit.
RESID_FLOOR = 0.15              # floor on the per-trophy residual sd.
REFRESH_FLOOR = 0.5             # logit sd of one discrete Exophase refresh.
MEDIAN_CAP = 2.5                # logit cap on median movement vs last rate.
SPREAD_GRID = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
SPREAD_MIN, SPREAD_MAX = 1.0, 2.5


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp to epoch seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


def log_time(t, t0):
    """x = ln(1 + hours from the release-day origin to t)."""
    return math.log1p(max(0.0, (t - t0) / 3600.0))


def logit(pct):
    p = min(HI, max(LO, pct)) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(l):
    if l >= 0.0:
        return 1.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return e / (1.0 + e)


def wls(xs, ys, ws):
    """Recency-weighted least squares line y ~ a + b*x.

    Returns (intercept, slope, weighted residual sd). With fewer than
    2 distinct x values the slope is 0 and the intercept is the
    weighted mean of y.
    """
    sw = sum(ws)
    if sw <= 0.0 or not xs:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = math.sqrt(
            sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / sw)
        return my, 0.0, resid
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(
        sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw)
    return a, b, resid


def pinball(y_pct, q_pct, tau):
    d = logit(y_pct) - logit(q_pct)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


class Case:
    """One parsed case: reads, times, and per-trophy series."""

    def __init__(self, raw):
        self.raw = raw
        self.trophies = raw["trophies"]
        self.n_troph = len(self.trophies)
        self.t0 = parse_time(EPOCH)
        self.at = parse_time(raw["at"])
        self.target = parse_time(raw["target"])
        self.horizon_days = float(raw["horizon_minutes"]) / 1440.0
        self.x_target = log_time(self.target, self.t0)
        hist = raw["history"]
        self.times = [parse_time(r["at"]) for r in hist]
        self.xs_all = [log_time(t, self.t0) for t in self.times]
        self.players = [float(r["players_tracked"]) for r in hist]
        # rates[i][k]: trophy i's rate at read k (None where missing).
        self.rates = []
        for i in range(self.n_troph):
            col = []
            for r in hist:
                v = r["rates"][i]
                col.append(None if v is None else float(v))
            self.rates.append(col)

    def weights(self, upto, now):
        """Recency weights: half-life 24 h back from `now`."""
        return [0.5 ** (max(0.0, now - t) / 86400.0)
                for t in self.times[:upto]]

    def players_fit(self, upto):
        """Weighted line of log(players) on x over reads [:upto]."""
        xs, ys, ws = [], [], []
        w = self.weights(upto, self.times[upto - 1])
        for k in range(upto):
            if self.players[k] > 0:
                xs.append(self.xs_all[k])
                ys.append(math.log(self.players[k]))
                ws.append(w[k])
        return wls(xs, ys, ws)

    def forecast_one(self, i, upto, target_x, horizon_days, players_fit):
        """Median logit + core sd for trophy i from reads [:upto].

        Core sd excludes the calibrated multiplier and the refresh
        floor; those are applied by the caller.
        """
        rates = self.rates[i]
        xs, ys, ws = [], [], []
        w = self.weights(upto, self.times[upto - 1])
        for k in range(upto):
            if rates[k] is not None:
                xs.append(self.xs_all[k])
                ys.append(logit(rates[k]))
                ws.append(w[k])
        if not xs:
            return logit(50.0), 1.0
        last = rates[upto - 1]
        if last is None:
            last = next((v for v in reversed(rates[:upto]) if v is not None),
                        50.0)
        last_logit = logit(last)
        a, b, resid = wls(xs, ys, ws)
        direct = a + b * target_x
        # Earned / players projection.
        xs_e, ys_e, ws_e = [], [], []
        for k in range(upto):
            if rates[k] is not None and self.players[k] > 0:
                earned = max(rates[k] / 100.0 * self.players[k], 0.5)
                xs_e.append(self.xs_all[k])
                ys_e.append(math.log(earned))
                ws_e.append(w[k])
        ae, be, _ = wls(xs_e, ys_e, ws_e)
        ap, bp, _ = players_fit
        log_e = ae + be * target_x
        log_p = ap + bp * target_x
        rate_ep = math.exp(log_e - log_p) * 100.0 if log_p > -700 else last
        rate_ep = min(HI, max(LO, rate_ep))
        med = 0.5 * (direct + logit(rate_ep))
        med = min(last_logit + MEDIAN_CAP, max(last_logit - MEDIAN_CAP, med))
        # Dilution term: volatility of log players_tracked jumps.
        jumps = []
        for k in range(1, upto):
            if self.players[k] > 0 and self.players[k - 1] > 0:
                jumps.append(math.log(self.players[k] / self.players[k - 1]))
        if len(jumps) >= 2:
            mj = sum(jumps) / len(jumps)
            vol = math.sqrt(sum((j - mj) ** 2 for j in jumps) / (len(jumps) - 1))
        else:
            vol = 0.0
        rw = 0.5 * math.sqrt(max(horizon_days, 0.0))
        resid_c = min(max(resid, RESID_FLOOR), 1.5)
        dilution = vol * math.sqrt(max(horizon_days, 0.0))
        core = math.sqrt(rw * rw + resid_c * resid_c + dilution * dilution)
        return med, core

    def calibrate(self):
        """Walk-forward backtest: spread multiplier minimising pinball.

        Forecast each past read (last 12 splits) from the reads before
        it and score the five quantiles against that read's rates.
        """
        n = len(self.times)
        if n < 4:
            return 1.0
        # Forecast H reads ahead, not the very next read: a one-read gap
        # (10-30 min) is so short that the spread barely matters and the
        # multiplier is unidentifiable. H scales with history length so
        # the backtest horizon carries real spread signal.
        horizon_reads = max(1, min(48, n // 6))
        splits = list(range(max(1, n - horizon_reads - 13),
                            n - horizon_reads))
        if not splits:
            return 1.0
        totals = {m: 0.0 for m in SPREAD_GRID}
        counts = {m: 0 for m in SPREAD_GRID}
        for k in splits:
            upto = k + 1  # reads [:upto] forecast read k+horizon_reads
            tgt = k + horizon_reads
            gap_days = (self.times[tgt] - self.times[upto - 1]) / 86400.0
            tx = self.xs_all[tgt]
            pf = self.players_fit(upto)
            for i in range(self.n_troph):
                y = self.rates[i][tgt]
                if y is None:
                    continue
                med, core = self.forecast_one(i, upto, tx, gap_days, pf)
                for m in SPREAD_GRID:
                    sd = math.sqrt((m * core) ** 2 + REFRESH_FLOOR ** 2)
                    for tau, z in zip((0.05, 0.25, 0.5, 0.75, 0.95), Z):
                        q = sigmoid(med + z * sd) * 100.0
                        totals[m] += pinball(y, q, tau)
                        counts[m] += 1
        best, best_loss = 1.0, None
        for m in SPREAD_GRID:
            if counts[m] == 0:
                continue
            loss = totals[m] / counts[m]
            if best_loss is None or loss < best_loss:
                best, best_loss = m, loss
        return min(SPREAD_MAX, max(SPREAD_MIN, best))


def forecast_case(raw):
    case = Case(raw)
    mult = case.calibrate()
    pf = case.players_fit(len(case.times))
    rows = []
    for i, tr in enumerate(case.trophies):
        med, core = case.forecast_one(
            i, len(case.times), case.x_target, case.horizon_days, pf)
        sd = math.sqrt((mult * core) ** 2 + REFRESH_FLOOR ** 2)
        qs = []
        prev = LO
        for z in Z:
            q = sigmoid(med + z * sd) * 100.0
            q = round(min(HI, max(LO, q)), 4)
            q = max(q, prev)  # keep quantiles non-decreasing after rounding
            prev = q
            qs.append(q)
        rows.append({"id": int(tr["id"]), "q": qs})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def fallback_case(raw):
    """Last read's rates, flat quantiles: used only if parsing fails."""
    hist = raw.get("history") or []
    last = hist[-1]["rates"] if hist else []
    rows = []
    for tr in raw.get("trophies", []):
        idx = tr["id"] - 1
        v = last[idx] if 0 <= idx < len(last) and last[idx] is not None else 50.0
        v = round(min(HI, max(LO, float(v))), 4)
        rows.append({"id": int(tr["id"]), "q": [v, v, v, v, v]})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        name = os.path.basename(path)[:-3]  # strip ".in"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            out = forecast_case(raw)
        except Exception:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                out = fallback_case(raw)
            except Exception:
                continue
        text = json.dumps(out, separators=(",", ":"))
        with open(os.path.join(out_dir, name + ".out"),
                  "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
