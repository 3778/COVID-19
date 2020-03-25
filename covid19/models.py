import numpy.random as npr
import numpy as np
from scipy.stats import expon
from covid19.utils import make_lognormal_from_interval


class SEIRBayes:
    ''' Model with Susceptible, Exposed, Infectious and Recovered compartments.

    This class implements the SEIR model with stochastic incubation and
    infectious periods as well as the basic reproduction number R0. 

    Examples:
        Default init.
        
        >>> np.random.seed(0)
        >>> model = SEIRBayes(t_max=5)
        >>> S, E, I, R, t_space = model.sample(3)
        >>> I
        array([[10., 10., 10.],
               [15., 10.,  9.],
               [17., 15.,  8.],
               [22., 17., 12.],
               [24., 18., 15.]])
        >>> model.params['r0_dist'].interval(0.95)
        (2.5, 6.0)

        Init by specifying parameter distributions by density interval.

        >>> np.random.seed(0)
        >>> model = SEIRBayes.init_from_intervals(
        ...             r0_interval=(1.9, 4.0, 0.90),
        ...             alpha_inv_interval=(4.1, 7.0, 0.80),
        ...             gamma_inv_interval=(7, 14, 0.99)
        ...         )
        >>> model.params['r0_dist'].mean()
        2.8283077760987947
        >>> model.params['r0_dist'].std()
        0.6483104983294321
        >>> model.params['alpha_inv_dist'].interval(0.8)
        (4.1, 7.0)

        Return parameter samples for analysis.

        >>> np.random.seed(0)
        >>> model = SEIRBayes(t_max=5)
        >>> (S, E, I, R, t_space, r0,
        ...  alpha, gamma, beta) = model.sample(5, return_param_samples=True)
        >>> r0
        array([5.74313347, 4.23505111, 4.81923138, 6.3885136 , 5.87744241])
        >>> alpha
        array([0.18303002, 0.15306351, 0.16825044, 0.18358956, 0.17569263])
        >>> gamma
        array([0.12007063, 0.08539356, 0.10375533, 0.1028759 , 0.09394099])
        >>> np.isclose(r0, beta/gamma)
        array([ True,  True,  True,  True,  True])
        >>> t_space
        array([0, 1, 2, 3, 4])
    '''
    def __init__(self, 
                 NEIR0=(100, 20, 10, 0),
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
