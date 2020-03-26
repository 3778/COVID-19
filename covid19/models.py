import numpy.random as npr
import numpy as np
from scipy.stats import expon
from covid19.utils import make_lognormal_from_interval


class SEIRBayes:
    ''' Model with Susceptible, Exposed, Infectious and Recovered compartments.

    This class implements the SEIR model with stochastic incubation and
    infectious periods as well as the basic reproduction number R0. 

    This model is an implicit density function on 4 time series S(t), E(t), 
    I(t) and R(t) for t = 0 to t_max-1. Sampling is done via numerical 
    resolution of a system of stochastic differential equations with 6 
    degrees of randomness: alpha, gamma, r0 and the number of subjects 
    transitioning between compartments; S -> E, E -> I, I -> R.

    Infectious (1/gamma) and incubation (1/alpha) periods can be specified by
    their lower and upper bounds via the init_from_intervals method, or with
    SciPy distribution objects, taken from scipy.stats, with the default class
    constructor. The same goes for the basic reproduction number r0.

    The probability of an individual staying in a compartment up to time t
    is proportional to exp(-p*t), therefore the probability of leaving is
    1 - exp(-p*t). The rate p is different for each pair of source and 
    destination compartments. They are as follows

        ======== ============= ========== ============== 
         Source   Destination     Rate        Period     
        ======== ============= ========== ============== 
         S        E             beta*I/N   1/(beta*I/N)  
         E        I             alpha      1/alpha       
         I        R             gamma      1/gamma       
        ======== ============= ========== ============== 

    Since the above discussion is for a single individual, a Binomial 
    distribution with success rate 1 - exp(-p*t) is used to scale to the
    total number of individuals in each compartment.

    Attributes:
        params (dict): Summary of model parameter values. Is compatible with
            the constructor; SEIRBayes(*params).
        _params (dict): Similar to params, but with modifications to
            facilitate usage internally. Isn't compatible with the constructor.

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


    '''
    def __init__(self, 
                 NEIR0=(100, 20, 10, 0),
                 r0_dist=make_lognormal_from_interval(2.5, 6.0, 0.95),
                 gamma_inv_dist=make_lognormal_from_interval(7, 14, 0.95),
                 alpha_inv_dist=make_lognormal_from_interval(4.1, 7, 0.95),
                 fator_subr=1,
                 t_max=30):
        '''Default constructor method.

        Args:
            NEIR0 (tuple): Initial conditions in the form of 
                (population size, exposed, infected, recovered). Notice that
                S0, the initial susceptible population, is not needed as it 
                can be calculated as S0 = N - R0 - fator_subr*(E0 + I0).
            r0_dist (object): scipy.stats compatible distribution object. Used
                to sample values for r0 (basic reproduction number).
            alpha_inv_dist (object): scipy.stats compatible distribution object. Used
                to sample values for alpha_inv (incubation period).
            gamma_inv_dist (object): scipy.stats compatible distribution object. Used
                to sample values for gamma_inv (infectious period).
            fator_subr (float): Multiplicative factor of I0 and E0 to take
                into account sub-reporting.
            t_max (int): Length of the time-series.

        Examples:
            >>> np.random.seed(0)
            >>> model = SEIRBayes(fator_subr=2)
            >>> model.params['fator_subr']
            2
            >>> model.params['r0_dist'].rvs(10)
            array([5.74313347, 4.23505111, 4.81923138, 6.3885136 , 5.87744241,
                   3.11354468, 4.7884938 , 3.74424985, 3.78472191, 4.24493851])
            >>> model.params['NEIR0']
            (100, 20, 10, 0)
            >>> model._params['init_conditions']
            (40, 40, 20, 0)
            >>> model._params['total_population']
            100
            
        '''

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
        ''' Init by specifying lognormal distribution by density interval.

        Args:
            All parameters are tuples and should be specified as
            (lower bound, upper bound, wanted density)

            r0_interval (tuple)
            alpha_inv_interval (tuple)
            gamma_inv_interval (tuple)

        Examples
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
        '''

        r0_dist=make_lognormal_from_interval(*r0_interval)
        gamma_inv_dist=make_lognormal_from_interval(*gamma_inv_interval)
        alpha_inv_dist=make_lognormal_from_interval(*alpha_inv_interval)
        return cls(r0_dist=r0_dist,
                   alpha_inv_dist=alpha_inv_dist,
                   gamma_inv_dist=gamma_inv_dist,
                   **kwargs)


    def sample(self, size=1, return_param_samples=False):
        '''Sample from model.
        Args:
            size (int): Number of samples.
            return_param_samples (bool): If true, returns the parameter
                samples (taken from {r0,gamma,alpha}_dist) used.

        Examples:

            >>> np.random.seed(0)
            >>> model = SEIRBayes(t_max=5)
            >>> S, E, I, R, t_space = model.sample(3)
            >>> S.shape, E.shape, I.shape, R.shape
            ((5, 3), (5, 3), (5, 3), (5, 3))
            >>> I
            array([[10., 10., 10.],
                   [15., 10.,  9.],
                   [17., 15.,  8.],
                   [22., 17., 12.],
                   [24., 18., 15.]])
            >>> t_space
            array([0, 1, 2, 3, 4])

            Return parameter samples for analysis.

            >>> np.random.seed(0)
            >>> model = SEIRBayes(t_max=5)
            >>> (S, E, I, R, t_space, r0,
            ...  alpha, gamma, beta) = model.sample(5, True)
            >>> r0
            array([5.74313347, 4.23505111, 4.81923138, 6.3885136 , 5.87744241])
            >>> alpha
            array([0.18303002, 0.15306351, 0.16825044, 0.18358956, 0.17569263])
            >>> gamma
            array([0.12007063, 0.08539356, 0.10375533, 0.1028759 , 0.09394099])
            >>> np.isclose(r0, beta/gamma).all()
            True
            >>> t_space
            array([0, 1, 2, 3, 4])
            
        '''
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
