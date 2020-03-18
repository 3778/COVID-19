import numpy as np
from scipy.stats import norm


default_values = {
    'N': 13_000_000,  # population size
    'E0': 50,  # init. exposed population
    'I0': 152,  # init. infected population
    'R0': 1,  # init. removed population
    'R0__loc': 2.2,  # repr. rate shape
    'gamma_loc': 1/10,  # removal rate shape
    'alpha_loc': 1/5.2,  # incubation rate shape
    't_max': 30*6,  # numer of days to run
    'runs': 100,  # number of run
}


def make_normal_scale(lb, ub, ci, loc):
    z = norm.ppf((1+ci)/2)
    scale_ub = -(loc - ub)/z
    return scale_ub


class Model():
    def __init__(self, params=default_values):
        for p in default_values:
            if p in params:
                continue
            params[p] = default_values[p]
        params['S0'] = (
            params['N'] - (params['I0'] + params['R0'] + params['E0'])
            )
        params['t_space'] = np.arange(0, params['t_max'])
        params['size'] = (params['t_max'], params['runs'])
        params['R0__scale'] = make_normal_scale(
            1.96, 2.55, .95, params['R0__loc']
            )
        params['gamma_scale'] = make_normal_scale(
            1/14, 1/7, .95, params['gamma_loc']
            )
        params['alpha_scale'] = make_normal_scale(
            1/7, 1/4.1, .95, params['alpha_loc']
            )
        params['S'] = None
        params['E'] = None
        params['I'] = None
        params['R'] = None
        for k, v in params.items():
            setattr(self, k, v)
