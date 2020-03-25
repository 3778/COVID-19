import numpy.random as npr
import numpy as np
from scipy.stats import expon
from covid19.utils import make_lognormal_from_interval


class SEIRBayes:
    def __init__(self, 
                 NEIR0=(100, 1, 0, 0),
                 r0_dist=make_lognormal_from_interval(2.5, 6.0, 0.95),
                 gamma_inv_dist=make_lognormal_from_interval(7, 14, 0.95),
                 alpha_inv_dist=make_lognormal_from_interval(4.1, 7, 0.95),
                 fator_subr=1,
                 t_max=30):

        self.params = {
            'NEIR0': NEIR0,
            'r0_dist': r0_dist,
            'gamma_inv_dist': gamma_inv_dist,
            'alpha_inv_dist': alpha_inv_dist,
            'fator_subr': fator_subr,
            't_max': t_max
        }

        N, E0, I0, R0 = NEIR0
        S0 = N - R0 - fator_subr*(I0 + E0)

        self._params = {
            'init_conditions': (S0, fator_subr*E0, fator_subr*I0, R0),
            'fator_subr': fator_subr,
            'total_population': N,
            'alpha_inv_dist': alpha_inv_dist,
            'gamma_inv_dist': gamma_inv_dist,
            'r0_dist': r0_dist,
            't_max': t_max,
            'param_samples': {}
        }

    @classmethod
    def init_from_intervals(cls,
                            r0_interval,
                            gamma_inv_interval,
                            alpha_inv_interval,
                            **kwargs):
        r0_dist=make_lognormal_from_interval(*r0_interval)
        gamma_inv_dist=make_lognormal_from_interval(*gamma_inv_interval)
        alpha_inv_dist=make_lognormal_from_interval(*alpha_inv_interval)
        return cls(r0_dist=r0_dist,
                   alpha_inv_dist=alpha_inv_dist,
                   gamma_inv_dist=gamma_inv_dist,
                   **kwargs)


    def sample(self, size=1, return_param_samples=False):
        t_space = np.arange(0, self._params['t_max'])
        N = self._params['total_population']
        S, E, I, R = [np.zeros((self._params['t_max'], size))
                      for _ in range(4)]
        S[0, ], E[0, ], I[0, ], R[0, ] = self._params['init_conditions']

        r0 = self._params['r0_dist'].rvs(size)
        gamma = 1/self._params['gamma_inv_dist'].rvs(size)
        alpha = 1/self._params['alpha_inv_dist'].rvs(size)
        beta = r0*gamma

        for t in t_space[1:]:
            SE = npr.binomial(S[t-1, ].astype(int),
                              expon(scale=1/(beta*I[t-1, ]/N)).cdf(1))
            EI = npr.binomial(E[t-1, ].astype(int),
                              expon(scale=1/alpha).cdf(1))
            IR = npr.binomial(I[t-1, ].astype(int),
                              expon(scale=1/gamma).cdf(1))

            dS =  0 - SE
            dE = SE - EI
            dI = EI - IR
            dR = IR - 0

            S[t, ] = S[t-1, ] + dS
            E[t, ] = E[t-1, ] + dE
            I[t, ] = I[t-1, ] + dI
            R[t, ] = R[t-1, ] + dR

        if return_param_samples:
            return S, E, I, R, t_space, r0, alpha, gamma, beta
        else:
            return S, E, I, R, t_space


if __name__ == '__main__':
    model = SEIRBayes()
    S, E, I, R, t_space = model.sample(100)
