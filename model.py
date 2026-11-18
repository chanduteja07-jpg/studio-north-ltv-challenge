"""Beta-discrete-Weibull retention + age-trended spend per active member.

All functions use only the observed tables passed in. The forecast origin is
the month before the first calendar month in the prediction grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln

CELL = ['channel', 'primary_need']
KEYS = ['cohort_month', 'channel', 'primary_need', 'calendar_month', 'age_month']
PREDICTIONS = ['predicted_retained', 'predicted_net_revenue', 'predicted_contribution']


def _months(series: pd.Series) -> np.ndarray:
    d = pd.to_datetime(series.astype(str).str[:10])
    return (d.dt.year * 12 + d.dt.month - 1).to_numpy()


def build_customers(tables: dict[str, pd.DataFrame], origin: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per acquired customer and transactions tagged by month."""
    tx = tables['transactions'].copy()
    for col in ['gross_amount', 'discount_amount', 'refund_amount', 'variable_cost']:
        tx[col] = pd.to_numeric(tx[col]).astype(float).fillna(0.0)
    tx['m'] = _months(tx.occurred_at)
    if (tx.m > origin).any():
        raise ValueError('Transactions after the forecast origin were supplied')
    tx['net'] = tx.gross_amount - tx.discount_amount - tx.refund_amount
    tx['con'] = tx.net - tx.variable_cost
    pay = tx[tx.transaction_type.astype(str).eq('payment') & (tx.gross_amount - tx.discount_amount > 0)]
    pay = pay.sort_values(['person_id', 'occurred_at', 'transaction_id'])
    first = pay.groupby('person_id').m.first().rename('cohort')

    camp = tables['campaign_spend'][['campaign_id', 'channel']].drop_duplicates('campaign_id')
    acq = tables['acquisition_attribution'][['person_id', 'credited_campaign_id', 'source_type']]
    acq = acq.merge(camp, left_on='credited_campaign_id', right_on='campaign_id', how='left')
    acq['channel'] = acq.channel.fillna(acq.source_type).astype(str)
    cust = acq[['person_id', 'channel']].merge(
        tables['people'][['person_id', 'primary_need']], on='person_id')
    cust['primary_need'] = cust.primary_need.astype(str)
    cust = cust.merge(first, left_on='person_id', right_index=True)

    p = tables['subscription_periods'][['person_id', 'period_start', 'renewal_outcome']].copy()
    p['m'] = _months(p.period_start)
    p = p[p.m <= origin + 1]
    paid = p[p.renewal_outcome.astype(str).eq('paid') & (p.m <= origin)]
    last = paid.groupby('person_id').m.max().rename('last_paid')
    cust = cust.merge(last, left_on='person_id', right_index=True, how='left')
    cust['last_age'] = cust.last_paid - cust.cohort
    # Censored: still active at origin (paid in the origin month).
    cust['censored'] = cust.last_paid.eq(origin)
    cust = cust[cust.last_age.notna()].copy()
    cust['last_age'] = cust.last_age.astype(int)
    return cust, tx


def _log_surv(a, b, c, ages):
    ages = np.asarray(ages, dtype=float)
    return betaln(a, b + np.power(ages, c)) - betaln(a, b)


def fit_bdw(last_age: np.ndarray, censored: np.ndarray, weights: np.ndarray | None = None,
            weibull: bool = True) -> tuple[float, float, float]:
    """MLE of beta-discrete-Weibull survival S(t)=B(a,b+t^c)/B(a,b); c=1 is sBG."""
    w = np.ones(len(last_age)) if weights is None else weights
    L = last_age.astype(float)

    def nll(x):
        a, b = np.exp(x[0]), np.exp(x[1])
        c = np.exp(x[2]) if weibull else 1.0
        s0 = _log_surv(a, b, c, L)
        s1 = _log_surv(a, b, c, L + 1)
        churn = np.log(np.clip(np.exp(s0) - np.exp(s1), 1e-300, None))
        ll = np.where(censored, s0, churn)
        return -np.sum(w * ll)

    best = None
    for start in ([0.0, 1.5, 0.0], [-1.0, 0.5, -0.3], [1.0, 2.5, 0.3]):
        r = minimize(nll, np.array(start if weibull else start[:2] + [0.0]), method='Nelder-Mead',
                     options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    a, b = np.exp(best.x[0]), np.exp(best.x[1])
    c = np.exp(best.x[2]) if weibull else 1.0
    return float(a), float(b), float(c)


def spend_curves(cust: pd.DataFrame, tx: pd.DataFrame, origin: int, lookback: int, trend: bool = False) -> dict:
    """Per-active monthly net revenue and contribution, as a function of age.

    log(value per active) = level_cell + slope * log(1 + age), fitted by
    weighted least squares on the last `lookback` calendar months, with the
    slope shared across cells (pooled) and cell levels.
    """
    t = tx.merge(cust[['person_id', 'cohort'] + CELL], on='person_id')
    t['age'] = t.m - t.cohort
    t = t[(t.m > origin - lookback) & (t.age >= 0)]
    money = t.groupby(CELL + ['cohort', 'm'])[['net', 'con']].sum()
    # Active members per cohort-cell-month from survival records.
    rows = []
    for (ch, nd, co), g in cust.groupby(CELL + ['cohort']):
        last = g.last_paid.to_numpy()
        for m in range(max(co, origin - lookback + 1), origin + 1):
            rows.append((ch, nd, co, m, int((last >= m).sum())))
    act = pd.DataFrame(rows, columns=CELL + ['cohort', 'm', 'active']).set_index(CELL + ['cohort', 'm'])
    df = act.join(money, how='left').fillna(0.0).reset_index()
    df = df[df.active > 0].copy()
    df['age'] = df.m - df.cohort
    out = {}
    for col in ['net', 'con']:
        df['y'] = df[col] / df.active
        d = df[df.y > 0]
        cells = sorted(d.groupby(CELL).groups)
        X = np.column_stack([np.log1p(d.age.to_numpy())] +
                            [((d.channel == c) & (d.primary_need == n)).to_numpy().astype(float) for c, n in cells] +
                            ([(d.m - origin).to_numpy().astype(float)] if trend else []))
        W = d.active.to_numpy()
        y = np.log(d.y.to_numpy())
        beta = np.linalg.lstsq(X * np.sqrt(W)[:, None], y * np.sqrt(W), rcond=None)[0]
        resid = y - X @ beta
        smear = float(np.sum(W * np.exp(resid)) / np.sum(W))  # Duan smearing
        gamma = float(beta[-1]) if trend else 0.0
        beta = beta[:1 + len(cells)]
        out[col] = {'slope': float(beta[0]), 'level': dict(zip(cells, beta[1:])), 'smear': smear, 'trend': gamma}
        # Cohort-cell calibration: observed / fitted value in the last 3 months.
        recent = df[df.m > origin - 3].copy()
        lvl = recent.apply(lambda r: beta[1 + cells.index((r.channel, r.primary_need))]
                           if (r.channel, r.primary_need) in cells else np.nan, axis=1)
        recent['fit'] = np.exp(lvl + beta[0] * np.log1p(recent.age) + gamma * (recent.m - origin)) * smear * recent.active
        agg = recent.groupby(['cohort'] + CELL)[[col, 'fit', 'active']].sum()
        out[col]['calib'] = {k: (float(v[col]), float(v['fit']), float(v['active'])) for k, v in agg.iterrows()}
    return out


def forecast(tables: dict[str, pd.DataFrame], grid: pd.DataFrame, *, weibull: bool = True,
             lookback: int = 6, recency_halflife: float | None = 12.0,
             calib_k: float | None = None, trend: bool = False, trend_damping: float = 1.0) -> pd.DataFrame:
    grid = grid.copy()
    origin_ts = pd.to_datetime(grid.calendar_month).min() - pd.DateOffset(months=1)
    origin = origin_ts.year * 12 + origin_ts.month - 1
    cust, tx = build_customers(tables, origin)
    params = {}
    for key, g in cust.groupby(CELL):
        w = None
        if recency_halflife:
            # Down-weight older cohorts so drift in acquisition quality is tracked.
            w = 0.5 ** ((origin - g.cohort.to_numpy()) / recency_halflife)
        params[key] = fit_bdw(g.last_age.to_numpy(), g.censored.to_numpy(), w, weibull)
    spend = spend_curves(cust, tx, origin, lookback, trend)

    keys = list(zip(grid.channel.astype(str), grid.primary_need.astype(str)))
    cm = _months(grid.cohort_month)
    age = grid.age_month.to_numpy(dtype=float)
    age0 = (origin - cm).astype(float)
    ret = np.zeros(len(grid))
    net = np.zeros(len(grid))
    con = np.zeros(len(grid))
    for i, k in enumerate(keys):
        a, b, c = params[k]
        ratio = np.exp(_log_surv(a, b, c, age[i]) - _log_surv(a, b, c, age0[i]))
        r = float(grid.origin_retained.iloc[i]) * ratio
        vals = {}
        for col in ['net', 'con']:
            s = spend[col]
            ahead = age[i] - age0[i]
            v = np.exp(s['level'][k] + s['slope'] * np.log1p(age[i]) + trend_damping * s['trend'] * ahead) * s['smear']
            if calib_k is not None:
                obs, fit, n_act = s['calib'].get((int(cm[i]),) + k, (0.0, 0.0, 0.0))
                if fit > 0 and obs > 0:
                    # Shrink the cohort-cell ratio toward 1 by member-months of evidence.
                    w = n_act / (n_act + calib_k)
                    v *= w * (obs / fit) + (1 - w)
            vals[col] = v
        n = max(0.0, vals['net'])
        ct = min(n, max(0.0, vals['con']))
        ret[i], net[i], con[i] = r, r * n, r * ct
    grid['predicted_retained'] = ret
    grid['predicted_net_revenue'] = net
    grid['predicted_contribution'] = con
    return grid
FINAL_SETTINGS = dict(weibull=True, recency_halflife=6.0, lookback=12, calib_k=20.0, trend=True, trend_damping=1.0)
