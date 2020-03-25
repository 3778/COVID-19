from scipy.stats import lognorm, norm
import numpy as np


def make_normal_from_interval(lb, ub, alpha):
    z = norm().interval(alpha)[1]
    mean_norm = (ub + lb) / 2
    std_norm = (ub - lb) / (2 * z)
    return norm(loc=mean_norm, scale=std_norm)


def make_lognormal_from_interval(lb, ub, alpha):
    z = norm().interval(alpha)[1]
    mean_norm = np.sqrt(ub * lb)
    std_norm = np.log(ub / lb) / (2 * z)
    return lognorm(s=std_norm, scale=mean_norm)
