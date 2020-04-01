from scipy.stats import lognorm, norm
import numpy as np


def make_normal_from_interval(lb, ub, alpha):
    ''' Creates a normal distribution SciPy object from intervals.

    This function is a helper to create SciPy distributions by specifying the
    amount of wanted density between a lower and upper bound. For example,
    calling with (lb, ub, alpha) = (2, 3, 0.95) will create a Normal
    distribution with 95% density between 2 a 3.

    Args:
        lb (float): Lower bound
        ub (float): Upper bound
        alpha (float): Total density between lb and ub

    Returns:
        scipy.stats.norm
    
    Examples:
        >>> dist = make_normal_from_interval(-1, 1, 0.63)
        >>> dist.mean()
        0.0
        >>> dist.std()
        1.1154821104064199
        >>> dist.interval(0.63)
        (-1.0000000000000002, 1.0)

    '''
    z = norm().interval(alpha)[1]
    mean_norm = (ub + lb) / 2
    std_norm = (ub - lb) / (2 * z)
    return norm(loc=mean_norm, scale=std_norm)


def make_lognormal_from_interval(lb, ub, alpha):
    ''' Creates a lognormal distribution SciPy object from intervals.

    This function is a helper to create SciPy distributions by specifying the
    amount of wanted density between a lower and upper bound. For example,
    calling with (lb, ub, alpha) = (2, 3, 0.95) will create a LogNormal
    distribution with 95% density between 2 a 3.

    Args:
        lb (float): Lower bound
        ub (float): Upper bound
        alpha (float): Total density between lb and ub

    Returns:
        scipy.stats.lognorm
    
    Examples:
        >>> dist = make_lognormal_from_interval(2, 3, 0.95)
        >>> dist.mean()
        2.46262863041182
        >>> dist.std()
        0.25540947842844575
        >>> dist.interval(0.95)
        (1.9999999999999998, 2.9999999999999996)

    '''
    z = norm().interval(alpha)[1]
    mean_norm = np.sqrt(ub * lb)
    std_norm = np.log(ub / lb) / (2 * z)
    return lognorm(s=std_norm, scale=mean_norm)


class EmpiricalDistribution:
    def __init__(self, observations, method='sequential'):
        self.observations = np.array(observations)
        self.method = 'sequential'
        self.rvs = (self._sequential_rvs if method == 'sequential' else
                    self._uniform_rvs)

    def _sequential_rvs(self, size):
        assert size <= len(self.observations)
        return self.observations[:size]

    def _uniform_rvs(self, size):
        return np.random.choice(self.observations, size, replace=True)
