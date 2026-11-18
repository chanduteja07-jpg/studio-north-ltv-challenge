"""Chronological backtest: rebuild the observed tables at earlier origins,
forecast forward, and score cohort-month contribution WAPE against the
released history (never the private 2026 outcomes).

Usage: uv run --frozen python backtest.py --data-dir /path/to/data
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import main as entry  # noqa: E402
import model  # noqa: E402

ORIGINS = ['2024-12-01', '2025-03-01', '2025-06-01', '2025-09-01']


def mi(ts):
    t = pd.Timestamp(ts)
    return t.year * 12 + t.month - 1


def truncate(tables, origin):
    """Tables exactly as they would have looked at the end of `origin` month."""
    cut = pd.Timestamp(origin) + pd.DateOffset(months=1)
    out = {k: v.copy() for k, v in tables.items()}
    tx = out['transactions']
    out['transactions'] = tx[pd.to_datetime(tx.occurred_at) < cut]
    p = out['subscription_periods']
    start = pd.to_datetime(p.period_start)
    p = p[start < cut + pd.DateOffset(months=1)].copy()
    nxt = pd.to_datetime(p.period_start) >= cut
    p.loc[nxt, 'renewal_outcome'] = 'pending'
    p.loc[nxt, 'cancellation_effective_at'] = pd.NA
    out['subscription_periods'] = p
    return out


def actuals(tables):
    """Full-history cohort-cell-month retained and contribution (truth)."""
    cust, tx = model.build_customers(tables, 10**9)
    t = tx.merge(cust[['person_id', 'cohort'] + model.CELL], on='person_id')
    money = t.groupby(['cohort'] + model.CELL + ['m'])[['net', 'con']].sum()
    p = tables['subscription_periods']
    p = p[p.renewal_outcome.astype(str).eq('paid')].merge(cust[['person_id', 'cohort'] + model.CELL], on='person_id')
    p['m'] = model._months(p.period_start)
    act = p.groupby(['cohort'] + model.CELL + ['m']).size().rename('retained')
    return act.to_frame().join(money, how='outer').fillna(0.0).reset_index()


def grid_at(truth, origin, horizon, n_acq):
    o = mi(origin)
    rows = []
    obs = truth.set_index(['cohort'] + model.CELL + ['m'])
    for (co, ch, nd), n in n_acq.items():
        if co > o:
            continue
        r0 = obs.retained.get((co, ch, nd, o), 0.0)
        for h in range(1, horizon + 1):
            m = o + h
            rows.append({'cohort_month': f'{co // 12}-{co % 12 + 1:02d}-01', 'channel': ch, 'primary_need': nd,
                         'calendar_month': f'{m // 12}-{m % 12 + 1:02d}-01', 'age_month': m - co,
                         'n_acquired': int(n), 'origin_retained': int(r0), '_co': co, '_m': m})
    return pd.DataFrame(rows)


def score(pred, truth):
    t = truth.groupby(['cohort', 'm']).con.sum()
    p = pred.groupby(['_co', '_m']).predicted_contribution.sum()
    a = t.reindex(p.index).fillna(0.0).to_numpy()
    tr = truth.groupby(['cohort', 'm']).retained.sum().reindex(p.index).fillna(0.0)
    pr = pred.groupby(['_co', '_m']).predicted_retained.sum()
    print('   retained bias', round(float((pr.sum() - tr.sum()) / tr.sum()), 4), end=' ')
    return float(np.abs(p.to_numpy() - a).sum() / np.abs(a).sum()), float((p.sum() - a.sum()) / np.abs(a).sum())


VARIANTS = {
    'constant_hazard_scaffold': None,
    'sBG_cell_pooled': dict(weibull=False, recency_halflife=None, lookback=6),
    'BdW_cell_pooled': dict(weibull=True, recency_halflife=None, lookback=6),
    'sBG_recency6_lb3_calib20': dict(weibull=False, recency_halflife=6.0, lookback=3, calib_k=20.0),
    'BdW_recency6_lb3_calib20': dict(weibull=True, recency_halflife=6.0, lookback=3, calib_k=20.0),
    'sBG_rec6_lb12_trend_calib20': dict(weibull=False, recency_halflife=6.0, lookback=12, calib_k=20.0, trend=True),
    'sBG_rec6_lb9_trend_calib20': dict(weibull=False, recency_halflife=6.0, lookback=9, calib_k=20.0, trend=True),
    'sBG_rec6_lb12_trend_half_calib20': dict(weibull=False, recency_halflife=6.0, lookback=12, calib_k=20.0, trend=True, trend_damping=0.5),
    'BdW_rec6_lb12_trend_calib20': dict(weibull=True, recency_halflife=6.0, lookback=12, calib_k=20.0, trend=True),
    'sBG_rec3_lb12_trend_calib20': dict(weibull=False, recency_halflife=3.0, lookback=12, calib_k=20.0, trend=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT / 'backtest_results.json')
    args = ap.parse_args()
    contract = json.loads((ROOT / 'submission-contract.json').read_text())
    tables = entry.load_tables(args.data_dir, contract)
    truth = actuals(tables)
    cust, _ = model.build_customers(tables, 10**9)
    n_acq = cust.groupby(['cohort'] + model.CELL).size()
    last_obs = truth.m.max()
    results = {}
    for origin in ORIGINS:
        horizon = min(12, last_obs - mi(origin))
        g = grid_at(truth, origin, horizon, n_acq)
        g = g[g.origin_retained >= 0]
        tt = truncate(tables, origin)
        for name, kw in VARIANTS.items():
            base = g.drop(columns=['_co', '_m'])
            if kw is None:
                pr = entry.example_predict(tt, base, seed=0)
            else:
                pr = model.forecast(tt, base, **kw)
            pr = pr.merge(g[model.KEYS + ['_co', '_m']], on=model.KEYS)
            w, bias = score(pr, truth)
            results.setdefault(name, {})[origin] = {'horizon_months': int(horizon), 'wape': round(w, 4), 'bias': round(bias, 4)}
            print(origin, horizon, name, round(w, 4), round(bias, 4), flush=True)
    for name, r in results.items():
        r['mean_wape'] = round(float(np.mean([v['wape'] for v in r.values()])), 4)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({k: v['mean_wape'] for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
