"""Independent Gaussian action posteriors on the SMS ratio scale."""
import hashlib
import math
import numpy as np

MULTIPLIERS = {'push': 0.50, 'sms': 0.65, 'digital_ads': 0.85}
NOISE_STD = 0.804

class Beliefs:
    def __init__(self, prior_mu=0.0, prior_var=0.25**2):
        if not math.isfinite(prior_mu) or not math.isfinite(prior_var) or prior_var <= 0:
            raise ValueError('Invalid Gaussian prior')
        self.prior_mu, self.prior_var = float(prior_mu), float(prior_var)
        self._state = {}
        self.observed = set()

    @property
    def actions(self):
        return tuple(sorted(self._state))

    def register(self, keys):
        for key in keys:
            self._state.setdefault(tuple(key), (self.prior_mu, self.prior_var))

    def posterior(self, action_key):
        return self._state.get(tuple(action_key), (self.prior_mu, self.prior_var))

    def update(self, action_key, channel, actual_n, observed_ratio):
        try:
            n, y = float(actual_n), float(observed_ratio)
        except (TypeError, ValueError, OverflowError):
            return False
        if channel not in MULTIPLIERS or n <= 0 or not np.isfinite([n, y]).all():
            return False
        key = tuple(action_key)
        mu, variance = self.posterior(key)
        h = MULTIPLIERS[channel] / MULTIPLIERS['sms']
        normalized_y = y / h
        normalized_r = NOISE_STD**2 / n / h**2
        new_variance = 1.0 / (1.0 / variance + 1.0 / normalized_r)
        self._state[key] = (new_variance * (mu / variance + normalized_y / normalized_r), new_variance)
        self.observed.add(key)
        return True

    def mean_ratio(self, action_key, channel):
        return MULTIPLIERS[channel] / MULTIPLIERS['sms'] * self.posterior(action_key)[0]

    def sd_ratio(self, action_key, channel):
        return MULTIPLIERS[channel] / MULTIPLIERS['sms'] * math.sqrt(self.posterior(action_key)[1])

    def draws(self, n_draws, seed=42):
        """theta[draw, action]; fixed keyed Z survive updates/catalog additions."""
        n_draws = int(n_draws)
        if n_draws < 0:
            raise ValueError('n_draws must be nonnegative')
        samples = np.empty((n_draws, len(self.actions)), dtype=float)
        for column, key in enumerate(self.actions):
            digest = hashlib.sha256((str(seed) + '|' + '|'.join(key)).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:16], 'little'))
            mu, variance = self.posterior(key)
            samples[:, column] = mu + math.sqrt(variance) * rng.standard_normal(n_draws)
        return samples
