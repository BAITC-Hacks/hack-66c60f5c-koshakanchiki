"""Independent toy, NOT the organizer/mock environment. Run with Python + NumPy.

8 disjoint cells, 5000 customers each, constant ARPU within a cell, one
SMS action per cell. Pilots have independent Gaussian observation noise.
Scores integrate random pilot overlap analytically, using the true cell
means ONLY for evaluation. Final selections consume <=15000 contacts incl.
pilots and <=100000 money, <=10 campaigns, <=5000 people per campaign.
If no cell has positive estimated marginal gain, a 100-person eligible
subsegment of the least-bad cell is the compulsory final fallback.
This toy does not reproduce all Beeline action choices or exact sampled IDs.
"""
import json
import time
from pathlib import Path
import numpy as np

K, N, SIGMA, C = 8, 5000, .804, 4.
ARPU = np.array([2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500.])
MAX_PILOT_CONTACTS = 2480
N_SEEDS = 100
Z, W = np.polynomial.hermite.hermgauss(7)
Z, W = np.sqrt(2) * Z, W / np.sqrt(np.pi)


def plan(mean, q, spent, ncontacts):
    """Exact mean-optimal within this toy's disjoint, homogeneous candidates."""
    available = min(15000 - ncontacts, int((100000 - spent) // C))
    final = np.zeros(K)
    marginal = mean * ARPU * (1 - q) - C
    for arm in np.argsort(-marginal, kind="stable"):
        if available <= 0 or marginal[arm] <= 0:
            break
        size = min(N, available)
        final[arm] = size
        available -= size
    if final.sum() == 0 and available > 0:
        final[int(np.argmax(marginal))] = min(100, available)
    value = np.dot(mean * ARPU, final + (N - final) * q)
    value -= spent + C * final.sum()
    return float(value), final


def run(truth, seed, policy):
    # Same pre-generated per-arm observation stream across policies; sizes differ.
    noise_rng = np.random.default_rng(seed)
    standard_noise = noise_rng.normal(size=(K, 20))
    policy_rng = np.random.default_rng(seed + 100000)
    visits = np.zeros(K, dtype=int)
    mean = np.zeros(K)
    variance = np.full(K, .25 ** 2)
    q = np.zeros(K)
    spent, ncontacts, pilots = 0., 0, 0
    for step in range(20):
        if step < 8:
            arm, size = step, 120  # common diverse warm-up
        elif policy == "fixed":
            arm, size = step % K, 120
        elif policy == "thompson":
            sample = policy_rng.normal(mean, np.sqrt(variance))
            arm, size = int(np.argmax(sample * ARPU)), 120
        else:
            base = plan(mean, q, spent, ncontacts)[0]
            best, action = -np.inf, None
            for a in range(K):
                for n in (120, 200):
                    if ncontacts + n > MAX_PILOT_CONTACTS:
                        continue
                    noise_var = SIGMA ** 2 / n
                    mean_change_sd = variance[a] / np.sqrt(variance[a] + noise_var)
                    qnext = q.copy()
                    qnext[a] = 1 - (1 - q[a]) * (1 - n / N)
                    expected = 0.
                    for z, w in zip(Z, W):
                        nextmean = mean.copy()
                        nextmean[a] += mean_change_sd * z
                        expected += w * plan(nextmean, qnext, spent + C*n, ncontacts+n)[0]
                    improvement = expected - base
                    if improvement > best:
                        best, action = improvement, (a, n)
            if action is None or best <= 0:
                break
            arm, size = action
        if ncontacts + size > MAX_PILOT_CONTACTS:
            break
        observed = truth[arm] + SIGMA / np.sqrt(size) * standard_noise[arm, visits[arm]]
        noise_var = SIGMA ** 2 / size
        weight = variance[arm] / (variance[arm] + noise_var)
        mean[arm] += weight * (observed - mean[arm])
        variance[arm] *= 1 - weight
        visits[arm] += 1
        q[arm] = 1 - (1-q[arm])*(1-size/N)
        spent += size*C
        ncontacts += size
        pilots += 1
    final = plan(mean, q, spent, ncontacts)[1]
    true_score = np.dot(truth*ARPU, final + (N-final)*q) - spent - C*final.sum()
    assert ncontacts + final.sum() <= 15000
    assert spent + C*final.sum() <= 100000
    assert pilots <= 20 and np.count_nonzero(final) <= 10
    return float(true_score), pilots, ncontacts


def main():
    started = time.monotonic()
    scenarios = {
        "sparse_winners": np.array([-.12,-.10,-.08,-.04,.00,.03,.15,.25]),
        "near_ties": np.array([.08,.08,.08,.08,.08,.08,.08,.08]),
        "all_negative": np.array([-.04,-.06,-.08,-.10,-.12,-.14,-.16,-.18]),
        "weak_positive": np.array([.003,.005,.007,.009,.011,.013,.015,.017]),
    }
    rows = []
    for name, truth in scenarios.items():
        for policy in ("fixed", "thompson", "one_step"):
            results = np.array([run(truth, seed, policy) for seed in range(N_SEEDS)])
            scores = results[:,0]
            row = dict(scenario=name, policy=policy, mean_net=round(float(scores.mean()),2),
                       p10_net=round(float(np.quantile(scores,.1)),2),
                       positive_runs=int((scores>0).sum()),
                       mean_pilots=round(float(results[:,1].mean()),2),
                       mean_pilot_contacts=round(float(results[:,2].mean()),2))
            rows.append(row)
            print(json.dumps(row), flush=True)
    output = dict(seeds=N_SEEDS, scenarios={k:v.tolist() for k,v in scenarios.items()},
                  elapsed_seconds=round(time.monotonic()-started,2), results=rows)
    Path(__file__).with_name("exploration_toy_results.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__ == "__main__":
    main()
