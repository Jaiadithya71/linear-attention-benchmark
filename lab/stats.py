import math
import numpy as np


def wilson_interval(successes, n, z=1.959963984540054):
    """95% Wilson score interval for a binomial proportion."""
    p = successes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def bootstrap_ci(bits, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean of a 0/1 vector (degenerate at 0% / 100%)."""
    bits = np.asarray(bits, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bits), size=(n_boot, len(bits)))
    means = bits[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize_times(ms):
    a = np.asarray(ms, dtype=float)
    return dict(median_ms=float(np.median(a)), q25_ms=float(np.quantile(a, 0.25)),
                q75_ms=float(np.quantile(a, 0.75)), min_ms=float(a.min()))
