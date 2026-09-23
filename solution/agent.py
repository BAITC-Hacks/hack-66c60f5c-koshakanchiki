"""Pilot-driven campaigns using only the public environment API."""
import logging
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from beliefs import Beliefs, NOISE_STD
from candidates import build_catalog, cell_stats, load_or_build_historical_rank
from planner import build_portfolio

LOG = logging.getLogger(__name__)
COSTS = {'push': 0, 'sms': 4, 'digital_ads': 22}
FIELDS = ('campaign_name', 'filter_arpu_segment', 'filter_data_segment',
          'filter_call_segment', 'filter_current_tariff', 'target_tariff', 'channel')


def _row(key, channel, data=None, call=None):
    current, segment, target = key
    name = f'main_{current}_{segment}_{target}_{channel}_{data or "all"}_{call or "all"}'
    return dict(zip(FIELDS, (name, segment, data, call, current, target, channel)))


def _audience(profile, row):
    result = profile
    for column in ('arpu_segment', 'data_segment', 'call_segment'):
        value = row.get('filter_' + column)
        if value is not None:
            result = result[result[column] == value]
    current = row.get('filter_current_tariff')
    if current is not None:
        result = result[result.current_tariff.isin(str(current).split(';'))]
    return result.sort_values('ID_NUMBER', kind='stable').iloc[:5000]


def _fallback(profile, tariffs, beliefs):
    """Smallest estimated downside among real, nonempty filterable groups."""
    known = sorted(str(t) for t in tariffs.tariff_plan_code.dropna())
    options = []
    columns = ['current_tariff', 'arpu_segment', 'data_segment', 'call_segment']
    for values, group in profile.groupby(columns, observed=True, dropna=True):
        current, segment, data, call = values
        alternatives = [key for key in beliefs.observed if key[:2] == (current, segment)]
        if not alternatives:
            target = next((t for t in known if t != current), None)
            alternatives = [(current, segment, target)] if target else []
        arpu = float(group.sort_values('ID_NUMBER', kind='stable').iloc[:5000].predicted_arpu.sum())
        for key in sorted(alternatives):
            value = (beliefs.mean_ratio(key, 'push') - beliefs.sd_ratio(key, 'push')) * arpu
            options.append((value, -len(group), key, data, call))
    if options:
        best = max(options, key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        return [_row(best[2], 'push', best[3], best[4])]
    for values, group in profile.groupby(columns[:2], observed=True):
        target = next((t for t in known if t != values[0]), None)
        if target:
            return [_row((*values, target), 'push')]
    return []  # No nonempty campaign exists on an empty audience.


def _validate_plan(profile, tariffs, rows, resources):
    known = set(tariffs.tariff_plan_code)
    money, contacts = resources['remaining_budget'], resources['remaining_contacts']
    result, seen = [], set()
    for raw in rows or []:
        row = {key: raw.get(key) for key in FIELDS}
        for key, value in row.items():
            if value is not None and pd.isna(value):
                row[key] = None
        if row['target_tariff'] not in known or row['channel'] not in COSTS:
            continue
        group = _audience(profile, row)
        n = min(len(group), max(0, int(contacts)))
        cost = COSTS[row['channel']]
        if cost:
            n = min(n, max(0, int(money // cost)))
        ids = set(group.iloc[:n].ID_NUMBER)
        if not ids or ids & seen:
            continue
        row['campaign_name'] = str(row['campaign_name'] or f'main_{len(result)+1}')
        result.append(row)
        seen.update(ids)
        contacts -= n
        money -= n * cost
        if len(result) == 10:
            break
    return result


class Agent:
    """Each act has a fresh posterior, ledger and deterministic randomness."""
    def act(self, env):
        random.seed(42)
        np.random.seed(42)
        started = time.monotonic()
        self.pilot_log, self.summary = [], {}
        beliefs = Beliefs()
        self.beliefs = beliefs
        plan, fallback = [], []
        profile, tariffs = None, None
        try:
            profile, tariffs = env.customer_profile.copy(), env.tariffs.copy()
            # Establish a legal recovery plan before optional data preparation.
            fallback = _fallback(profile, tariffs, beliefs)
            stats = cell_stats(profile)
            history_path = Path(__file__).parent / 'data' / 'change_tariff.csv'
            if not history_path.is_file():
                history_path = Path(__file__).with_name('historical_candidates.json')
            history = load_or_build_historical_rank(history_path, tariffs)
            catalog = build_catalog(profile, tariffs, history)
            keys = [(c['from_tariff'], c['arpu_segment'], c['target']) for c in catalog]
            beliefs.register(keys)
            initial_h, initial_b = int(env.remaining_contacts), float(env.remaining_budget)
            reserve_h = min(12520, max(1, initial_h - 2480))
            reserve_b = max(0.0, initial_b - 9920)

            def resources():
                return {'remaining_contacts': max(0, int(env.remaining_contacts)),
                        'remaining_budget': max(0.0, float(env.remaining_budget)),
                        'pilot_log': self.pilot_log, 'profile': profile,
                        'max_campaigns': 10, 'max_customers_per_campaign': 5000,
                        'deadline': started + 210}

            refined = set()
            for step, stage_n in enumerate([120] * 8 + [180] * 4):
                if time.monotonic() - started >= 180 or env.pilots_left <= 0:
                    break
                if step < 8:
                    if step >= len(keys):
                        continue
                    key = keys[step]
                else:
                    choices = []
                    for key in sorted(beliefs.observed):
                        if key in refined:
                            continue
                        mu, variance = beliefs.posterior(key)
                        if mu + math.sqrt(variance) <= 0:
                            continue
                        cell = stats[key[:2]]
                        n = min(stage_n, cell['N'])
                        after = 1.0 / (1.0 / variance + n / NOISE_STD**2)
                        value = cell['prefix_arpu'][min(5000, cell['N'])] * (math.sqrt(variance) - math.sqrt(after))
                        choices.append((value, key))
                    if not choices:
                        break
                    key = min(choices, key=lambda pair: (-pair[0], pair[1]))[1]
                    refined.add(key)
                cell = stats.get(key[:2])
                if not cell or cell['N'] <= 0:
                    continue
                allowed = min(200, stage_n, cell['N'], int(env.remaining_contacts) - reserve_h,
                              int((float(env.remaining_budget) - reserve_b) // 4))
                if allowed <= 0 or (allowed < 10 and cell['N'] > allowed):
                    break
                request = {'target_tariff': key[2], 'channel': 'sms', 'n_customers': max(10, int(allowed)),
                           'filter_current_tariff': key[0], 'filter_arpu_segment': key[1]}
                before_h, before_b = int(env.remaining_contacts), float(env.remaining_budget)
                pilot_error = None
                try:
                    response = env.run_pilot(**request)
                except Exception as error:
                    pilot_error, response = str(error), {}
                if not isinstance(response, dict):
                    LOG.warning('Pilot returned a non-dictionary response')
                    response = {}
                # Resource deltas remain valid even when the response is malformed.
                actual_n = max(0, before_h - int(env.remaining_contacts))
                cost = max(0.0, before_b - float(env.remaining_budget))
                record = {'action_key': key, 'request': request, 'result': dict(response),
                          'channel': 'sms', 'actual_n': actual_n,
                          'observed_ratio': response.get('observed_lift_ratio'),
                          'cost': cost, 'elapsed_seconds': time.monotonic() - started}
                if pilot_error is not None:
                    record['error'] = pilot_error
                self.pilot_log.append(record)
                if pilot_error is not None:
                    LOG.warning('Pilot stopped: %s', pilot_error)
                    break
                if beliefs.update(key, 'sms', actual_n, record['observed_ratio']):
                    y, total = float(record['observed_ratio']), response.get('observed_lift_total')
                    try:
                        if total is not None and y != 0 and np.isfinite(float(total) / y):
                            record['sample_arpu'] = float(total) / y
                    except (TypeError, ValueError, OverflowError):
                        LOG.warning('Ignoring malformed optional pilot total')
                # A successful update may invalidate yesterday's positive choice.
                plan = []
                proposed = build_portfolio(beliefs, stats, resources())
                plan = _validate_plan(profile, tariffs, proposed, resources())
            final_resources = resources()
            plan = _validate_plan(profile, tariffs, plan, final_resources)
            if not plan:
                plan = _validate_plan(profile, tariffs, _fallback(profile, tariffs, beliefs), final_resources)
            self.summary = {'pilot_count': len(self.pilot_log), 'campaign_count': len(plan),
                            'pilot_contacts': sum(r['actual_n'] for r in self.pilot_log),
                            'pilot_cost': sum(r['cost'] for r in self.pilot_log),
                            'elapsed_seconds': time.monotonic() - started,
                            'missing_cell_keys': int(profile[['current_tariff', 'arpu_segment']].isna().any(axis=1).sum())}
            return plan
        except Exception as error:
            LOG.exception('Agent recovered from %s', error)
            self.summary['error'] = str(error)
            if profile is not None and tariffs is not None:
                try:
                    current = {'remaining_budget': env.remaining_budget, 'remaining_contacts': env.remaining_contacts}
                    return (_validate_plan(profile, tariffs, plan, current)
                            or _validate_plan(profile, tariffs, _fallback(profile, tariffs, beliefs), current))
                except Exception:
                    pass
            return fallback
