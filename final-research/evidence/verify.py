#!/usr/bin/env python3
"""Reproduce research, not Agent.act. Run from any cwd; needs numpy/pandas.

Preserved experiment scripts execute with outputs in TemporaryDirectory.
Only public case files and invented models are used. No organizer secrets/API.
"""
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import platform
import random
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'project-context/case-files'
sys.path.insert(0, str(CASE))
import numpy as np
import pandas as pd
from scoring_core import CHANNELS, score_campaign, score_campaigns
from environment import make_environment

EVIDENCE = Path(__file__).resolve().parent
OUT = EVIDENCE / 'results.json'
EXPERIMENTS = EVIDENCE / 'experiments'
EXPECTED = EVIDENCE / 'expected.json'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hashes():
    paths = []
    for name in ('project-context', 'final-research/evidence/experiments'):
        paths.extend(p for p in (ROOT / name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.append(EXPECTED)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def no_timing(value):
    if isinstance(value, dict):
        return {k: no_timing(v) for k, v in value.items() if k != 'elapsed_seconds'}
    if isinstance(value, list):
        return [no_timing(v) for v in value]
    return value


def reproduce(tmp):
    expected = json.loads(EXPECTED.read_text())['reports']
    audit = load('amir_data', EXPERIMENTS / 'research_data_audit.py')
    audit.ROOT, audit.OUT = CASE, tmp / 'data-audit'
    mechanics = load('amir_mechanics', EXPERIMENTS / 'research_mechanics_checks.py')
    mechanics.ROOT = tmp
    priors = load('az_priors', EXPERIMENTS / 'priors_audit.py')
    portfolio = load('az_portfolio', EXPERIMENTS / 'portfolio_toy.py')
    exploration = load('az_exploration', EXPERIMENTS / 'exploration_toy.py')
    portfolio.__file__ = str(tmp / 'portfolio_toy.py')
    exploration.__file__ = str(tmp / 'exploration_toy.py')
    with contextlib.redirect_stdout(io.StringIO()):
        audit.main()
        mechanics.main()
        portfolio.main()
        exploration.main()
    results = {}
    comparisons = {
        'data_audit': tmp / 'data-audit/audit_summary.json',
        'mechanics': tmp / 'research_tables/mechanics_checks.json',
        'portfolio': tmp / 'portfolio_toy_results.json',
        'exploration': tmp / 'exploration_toy_results.json',
    }
    for name, actual in comparisons.items():
        a, b = json.loads(actual.read_text()), expected[name]
        # Library versions are recorded, but not a numerical research result.
        aa, bb = no_timing(a), no_timing(b)
        aa.pop('python_libraries', None); bb.pop('python_libraries', None)
        assert aa == bb, f'Reproduction mismatch: {name}'
        results[name] = {'matches_saved': True}
        if name == 'mechanics': results[name]['checks_passed'] = a['checks_passed']
        if name == 'portfolio': results[name]['summary'] = a['summary']
        if name == 'exploration': results[name]['results'] = a['results']
    prior = priors.audit(CASE)
    assert prior == expected['priors']
    results['priors'] = {'matches_saved': True, 'coverage': prior['coverage']}
    return results, portfolio


def reconcile_data():
    h = pd.read_csv(CASE / 'data/change_tariff.csv')
    p = pd.read_csv(CASE / 'customer_profile.csv')
    def actions(df, threshold):
        df = df.loc[df.AVG_ARPU_PREV_3M >= threshold].copy() if threshold else df.loc[df.AVG_ARPU_PREV_3M > 0].copy()
        df['segment'] = pd.cut(df.AVG_ARPU_PREV_3M, [-np.inf, 1000, 5000, np.inf], labels=['LOW','MID','HIGH'])
        keys = set(df[['tariff_plan_code_from','segment','tariff_plan_code_to']].itertuples(index=False, name=None))
        return len(df), keys
    rows_a, a = actions(h, 100)
    rows_z, z = actions(h.drop_duplicates(), 0)
    rows_clean, clean = actions(h.drop_duplicates(), 100)
    assert (len(h), rows_a, len(a), rows_z, len(z)) == (14823, 12705, 451, 13410, 456)
    assert len(z - a) == 5 and not (a-z)
    overlap = {}
    for name in ('change_tariff','traffic','arpu_monthly'):
        ids = pd.read_csv(CASE / f'data/{name}.csv', usecols=['ID_NUMBER']).ID_NUMBER
        overlap[name] = len(set(p.ID_NUMBER) & set(ids))
    assert set(overlap.values()) == {0}
    boundary_counts = {str(x): int((h.AVG_ARPU_PREV_3M == x).sum()) for x in (1000, 5000)}
    assert boundary_counts['1000'] == 0
    cells = p.groupby(['current_tariff','arpu_segment'], observed=True).predicted_arpu.agg(['size','sum'])
    return {
        'raw_rows':len(h), 'exact_duplicates':int(h.duplicated().sum()),
        'amir_mock_cleaning':{'rows':rows_a,'actions':len(a),'deduplicates':False,'previous_arpu_min':100},
        'azamat_cleaning':{'rows':rows_z,'actions':len(z),'deduplicates':True,'previous_arpu_strictly_above':0},
        'recommended_ranking_cleaning':{'rows':rows_clean,'actions':len(clean),'deduplicates':True,'previous_arpu_min':100},
        'five_extra_actions': sorted([list(k) for k in z-a]),
        'history_boundary_counts':boundary_counts, 'id_overlap':overlap,
        'profiles':len(p),'known_cells':len(cells),'known_profiles':int(cells['size'].sum()),
        'max_cell_size':int(cells['size'].max()),'baseline_arpu':float(p.predicted_arpu.sum()),
        'top_four_baseline_share':float(cells['sum'].nlargest(4).sum()/p.predicted_arpu.sum()),
        'history_columns':list(h.columns),
    }


def mechanism_probes():
    # Observations include channel, so likelihood uses y = a * SMS_effect + noise.
    mu, var = .03, .25**2
    records = [('push', 200, .04), ('sms', 120, -.02), ('digital_ads', 90, .06)]
    precision, natural = 1/var, mu/var
    for channel, n, y in records:
        a = CHANNELS[channel]['conversion_multiplier']/.65
        R = .804**2/n
        precision += a*a/R; natural += a*y/R
        w = var*a/(a*a*var+R)
        mu += w*(y-a*mu); var *= R/(a*a*var+R)
    assert np.allclose([mu,var], [natural/precision,1/precision])
    # Official scorer: identical cheap-channel products can imply different call effects.
    profile = pd.DataFrame({'ID_NUMBER':[1], 'current_tariff':['a'], 'arpu_segment':['MID'], 'predicted_arpu':[100.]})
    tariffs = pd.DataFrame({'tariff_plan_code':['a','b'], 'price_tariff':[100.,200.]})
    worlds=[]
    for delta, conversion in ((.4,.5),(.2,1.)):
        model=pd.DataFrame([{'tariff_plan_code_from':'a','tariff_plan_code_to':'b','arpu_segment':'MID','arpu_change_pct':delta,'conversion_rate':conversion}])
        ratios={ch:float(score_campaign(profile,'b',model,tariffs,conversion,ch,lambda *args:(0.,0.)).expected_lift_per_customer.iloc[0]/100) for ch in CHANNELS}
        worlds.append(ratios)
    assert np.allclose([worlds[0][c] for c in ('push','sms','digital_ads')], [worlds[1][c] for c in ('push','sms','digital_ads')])
    assert np.allclose([w['call'] for w in worlds],[.24,.20])
    # A public aggregate identifies sample baseline, not the sampled IDs.
    heterogeneous = pd.DataFrame({'ID_NUMBER':np.arange(100), 'current_tariff':['a']*100,
        'arpu_segment':['MID']*100, 'predicted_arpu':100.+np.arange(100)**2})
    env, internals = make_environment(heterogeneous, model, tariffs, CHANNELS,
        100000, 15000, lambda *args:(0.,0.), seed=42)
    observed = env.run_pilot('b','sms',n_customers=20,filter_current_tariff='a',filter_arpu_segment='MID')
    recovered = observed['observed_lift_total']/observed['observed_lift_ratio']
    # Only this test harness sees internals of its own invented world.
    sample_ids = internals.executed_pilot_campaigns()[0]['explicit_ids']
    actual_baseline = float(heterogeneous.loc[heterogeneous.ID_NUMBER.isin(sample_ids),'predicted_arpu'].sum())
    assert np.isclose(recovered,actual_baseline)
    final = {'target_tariff':'b','channel':'push','filter_current_tariff':'a','filter_arpu_segment':'MID'}
    scored = score_campaigns(pd.DataFrame(internals.executed_pilot_campaigns()+[final]),
        heterogeneous,model,tariffs,float(heterogeneous.predicted_arpu.sum()),lambda *args:(0.,0.))
    full_cell_formula = float(heterogeneous.predicted_arpu.sum())*.10+recovered*max(.13-.10,0.)
    assert np.isclose(full_cell_formula,scored['gross_arpu_lift'])
    # Exact enumeration of two independent size-2 pilots in a four-person cell.
    subsets=list(itertools.combinations(range(4),2))
    pilot_ratios=[-.2,.1]; final_ratio=-.05; final_ids={0,1}; baseline=np.array([100.,200.,300.,400.])
    exact=[]
    for s1,s2 in itertools.product(subsets,repeat=2):
        gross=0.
        for i,b in enumerate(baseline):
            touched=([pilot_ratios[0]] if i in s1 else [])+([pilot_ratios[1]] if i in s2 else [])+([final_ratio] if i in final_ids else [])
            gross+=b*(max(touched) if touched else 0.)
        exact.append(gross)
    def expected_best(actions):
        total=0.; not_better=1.
        for ratio,q in sorted(actions,reverse=True):
            total+=ratio*q*not_better; not_better*=1-q
        return total
    expected=sum(b*expected_best([(-.2,.5),(.1,.5)]+([(-.05,1.)] if i in final_ids else [])) for i,b in enumerate(baseline))
    assert np.isclose(np.mean(exact),expected)
    sigma=.804/math.sqrt(120)
    # E[max(Y,0)] at theta=0; this positive bias must not enter the likelihood.
    clipping_bias=sigma/math.sqrt(2*math.pi)
    return {'gaussian_update':{'mean':mu,'variance':var,'sequential_matches_batch':True},
            'call_nonidentifiability':worlds,
            'public_pilot_baseline':{'seed':42,'n':20,'recovered_sum':recovered,'harness_actual_sum':actual_baseline,
                'full_cell_formula_gross':full_cell_formula,'scorer_gross':scored['gross_arpu_lift'],'ids_not_returned_to_agent':True},
            'expected_overlap':{'enumerated_outcomes':len(exact),'mean_gross':float(np.mean(exact)),'formula_gross':expected,'includes_negative_effects':True},
            'negative_observation_clipping':{'true_ratio':0,'n':120,'expected_clipped_ratio':clipping_bias},
            'pilot_se':{str(n):.804/math.sqrt(n) for n in (40,120,150,180,200)},
            'sms_information_over_push':(.65/.5)**2,
            'numpy_legacy_aliases':{name:hasattr(np,name) for name in ('infty','alltrue')},
            'net_per_cost_at_zero_cost':'undefined; do not use for push'}


def portfolio_stress(mod):
    # Adds negative/weak effects and an ACTIVE row limit to the original toy.
    families=('mixed_signs','weak_effects','all_negative','three_slots')
    output=[]
    for family in families:
        mod.ROWS = 3 if family == 'three_slots' else 10
        mod.BUDGET, mod.CONTACTS = 90080, 12520
        cases=[]
        for seed in range(20):
            rng=random.Random(1000+seed)
            blocks=[]
            for _ in range(8):
                n=rng.randrange(5,46)*100; arpu=rng.uniform(400,12000)
                lo,hi={'mixed_signs':(-.15,.16),'weak_effects':(-.003,.012),'all_negative':(-.15,-.001),'three_slots':(.001,.16)}[family]
                effect=rng.uniform(lo,hi)
                blocks.append([(0.,0,0,0)]+[(n*(arpu*effect*m-c),n,n*c,1) for _,m,c in mod.CHANNELS[1:]])
            baseline=mod.sms_first(blocks)
            multi=max([baseline]+[mod.greedy(blocks,c) for c in range(3)],key=lambda s:mod.score(blocks,s))
            local=mod.improve(blocks,multi)
            exact=max(mod.score(blocks,s) for s in itertools.product(range(4),repeat=8))
            values={name:mod.score(blocks,state) for name,state in [('sms_first',baseline),('multistart',multi),('local',local)]}
            assert exact+1e-7 >= values['local'] >= values['multistart'] >= values['sms_first']
            cases.append({'world_seed':1000+seed,'exact':exact,**values})
        summary={}
        for name in ('sms_first','multistart','local'):
            gaps=[r['exact']-r[name] for r in cases]
            nonzero=[r[name]/r['exact'] for r in cases if r['exact']>0]
            summary[name]={'matches_exact':sum(abs(g)<1e-7 for g in gaps),'mean_gap':float(np.mean(gaps)),
                           'worst_gap':max(gaps),'mean_pct_when_exact_positive':100*float(np.mean(nonzero)) if nonzero else None}
        output.append({'family':family,'world_seeds':'1000..1019','cases':20,'row_limit':mod.ROWS,'summary':summary})
    return {'scope':'Known synthetic effects, 8 disjoint whole blocks, no pilots, skip-all allowed. Not an Agent/API performance test.',
            'enumerated_assignments_per_world':4**8,'families':output}


def main():
    started=time.perf_counter(); before=hashes()
    with tempfile.TemporaryDirectory(prefix='koshakanchiki-verify-') as tmp:
        replay,portfolio=reproduce(Path(tmp))
        data=reconcile_data(); probes=mechanism_probes(); stress=portfolio_stress(portfolio)
    assert hashes() == before, 'Research inputs changed during verification'
    manifest = {}
    for line in (ROOT / 'project-context/SHA256SUMS.txt').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        actual = hashlib.sha256((ROOT / 'project-context' / relative).read_bytes()).hexdigest()
        assert actual == digest, f'Official input changed: {relative}'
        manifest[relative] = True
    provenance = json.loads(EXPECTED.read_text())['source_scripts']
    for name, record in provenance.items():
        assert hashlib.sha256((EXPERIMENTS / name).read_bytes()).hexdigest() == record['sha256'], name
    output={'checked_on':'2026-09-23','command':'python final-research/evidence/verify.py',
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
            'elapsed_seconds':round(time.perf_counter()-started,3),'inputs_unchanged':True,'official_manifest_matches':manifest,
            'preserved_experiments_match_originals':True,'source_sha256':before,
            'reproduced':replay,'data_reconciliation':data,'new_mechanism_probes':probes,'new_portfolio_stress':stress}
    OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'elapsed_seconds':output['elapsed_seconds'],
                      'mechanics':replay['mechanics'],'inputs_unchanged':True,'new_portfolio_stress':stress},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
