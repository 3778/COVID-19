import numpy as np
import numpy.random as npr
from scipy.integrate import odeint
from scipy.stats import norm
import matplotlib.pyplot as plt



def make_lognormal_params_95_ci(lb, ub):
    mean = (ub*lb)**(1/2)
    std = (ub/lb)**(1/4)
    return mean, std


def run_SEIR_BAYES_model(
        N: 'population size',
        E0: 'init. exposed population',
        I0: 'init. infected population',
        R0: 'init. removed population',
        R0__params: 'repr. rate mean and std',
        gamma_inv_params: 'removal rate mean and std',
        alpha_inv_params: 'incubation rate mean and std',
        t_max: 'numer of days to run',
        runs: 'number of runs'
    ):

    S0 = N - (I0 + R0 + E0)
    t_space = np.arange(0, t_max)

    size = (t_max, runs)

    S = np.zeros(size)
    E = np.zeros(size)
    I = np.zeros(size)
    R = np.zeros(size)
    
    S[0, ], E[0, ], I[0, ], R[0, ] = S0, E0, I0, R0

    R0_ = npr.lognormal(*map(np.log, R0__params), runs)
    gamma = 1/npr.lognormal(*map(np.log, gamma_inv_params), runs)
    alpha = 1/npr.lognormal(*map(np.log, alpha_inv_params), runs)
    beta = R0_*gamma
   
    for t in t_space[1:]:
        SE = npr.binomial(S[t-1, ].astype('int'), 1 - np.exp(-beta*I[t-1, ]/N))
        EI = npr.binomial(E[t-1, ].astype('int'), 1 - np.exp(-alpha))
        IR = npr.binomial(I[t-1, ].astype('int'), 1 - np.exp(-gamma))

        dS =  0 - SE
        dE = SE - EI
        dI = EI - IR
        dR = IR - 0

        S[t, ] = S[t-1, ] + dS
        E[t, ] = E[t-1, ] + dE
        I[t, ] = I[t-1, ] + dI
        R[t, ] = R[t-1, ] + dR
    
    return S, E, I, R, t_space


def seir_bayes_plot(N, E0, I0, R0,
                    R0__params,
                    gamma_inv_params,
                    alpha_inv_params,
                    t_max, runs, S, E, I, R, t_space):
    S0 = N - (I0 + R0 + E0)
    # plot
    algorithm_text = (
        f"for {runs} runs, do:\n"
        f"\t$S_0={S0}$\n\t$E_0={E0}$\n\t$I_0={I0}$\n\t$R_0={R0}$\n"
         "\t$\\gamma \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         "\t$\\alpha \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         "\t$R0 \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
        f"\t$\\beta = \\gamma R0$\n"
        f"\tSolve SEIR$(\\alpha, \\gamma, \\beta)$"
    ).format(*gamma_inv_params, *alpha_inv_params, *R0__params)

    title = '(RESULTADO PRELIMINAR) Pessoas afetadas pelo COVID-19, segundo o modelo SEIR-Bayes'
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16,9))
    plt.plot(t_space, E.mean(axis=1), '--', t_space, I.mean(axis=1), '--', marker='o')
    plt.title(title, fontsize=20)
    plt.legend(['Expostas ($\mu \pm \sigma$)',
                'Infectadas ($\mu \pm \sigma$)'],
               fontsize=20, loc='lower right')
    plt.xlabel('t (Dias a partir de 17/Março/2020)', fontsize=20)
    plt.ylabel('Pessoas', fontsize=20)
    plt.fill_between(t_space,
                     I.mean(axis=1) + I.std(axis=1), 
                     (I.mean(axis=1) - I.std(axis=1)).clip(I0),
                     color='b', alpha=0.2)
    plt.fill_between(t_space, 
                     E.mean(axis=1) + E.std(axis=1), 
                     (E.mean(axis=1) - E.std(axis=1)).clip(I0),
                     color='r', alpha=0.2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, algorithm_text,
            transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)
    plt.yscale('log')
    return fig


def seir_bayes_interactive_plot(N, E0, I0, R0,
                                t_max, runs, S, E, I, R, t_space,
                                scale='log', show_uncertainty=True):

    from .visualization import prep_tidy_data_to_plot, make_combined_chart
    source = prep_tidy_data_to_plot(E, I, t_space)
    chart = make_combined_chart(source, 
                                scale=scale, 
                                show_uncertainty=show_uncertainty)
    return chart


if __name__ == '__main__':
    N = 13_000_000
    E0, I0, R0 = 300, 250, 1
    R0__params = make_lognormal_params_95_ci(1.96, 2.55)
    gamma_inv_params = make_lognormal_params_95_ci(10, 16)
    alpha_inv_params = make_lognormal_params_95_ci(4.1, 7)
    t_max = 30*6
    runs = 1_000
    S, E, I, R, t_space = run_SEIR_BAYES_model(
                                      N, E0, I0, R0,
                                      R0__params,
                                      gamma_inv_params,
                                      alpha_inv_params,
                                      t_max, runs)

    fig = seir_bayes_plot(N, E0, I0, R0,
                          R0__params,
                          gamma_inv_params,
                          alpha_inv_params,
                          t_max, runs, S, E, I, R, t_space)
    plt.show()
