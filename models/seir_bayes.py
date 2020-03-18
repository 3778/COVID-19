import numpy as np
import numpy.random as npr
from scipy.integrate import odeint
from scipy.stats import norm
import matplotlib.pyplot as plt


def make_normal_scale(lb, ub, ci, loc):
    z = norm.ppf((1+ci)/2)
    scale_ub = -(loc - ub)/z
    return scale_ub


def run_SEIR_BAYES_model(
        N: 'population size',
        E0: 'init. exposed population',
        I0: 'init. infected population',
        R0: 'init. removed population',
        R0__loc: 'repr. rate shape',
        R0__scale: 'repr. rate scale',
        gamma_loc: 'removal rate shape',
        gamma_scale: 'removal rate scale',
        alpha_loc: 'incubation rate shape',
        alpha_scale: 'incubation rate scale',
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
    for r in range(runs):
        S[0, r] = S0
        E[0, r] = E0
        I[0, r] = I0
        R[0, r] = R0

        R0_ = npr.normal(R0__loc, R0__scale)
        gamma = npr.normal(gamma_loc, gamma_scale)
        alpha = npr.normal(alpha_loc, alpha_scale)
        beta = R0_*gamma
        for t in t_space[1:]:
            SE = npr.binomial(S[t-1, r], 1 - np.exp(-beta*I[t-1, r]/N))
            EI = npr.binomial(E[t-1, r], 1 - np.exp(-alpha))
            IR = npr.binomial(I[t-1, r], 1 - np.exp(-gamma))

            dS =  0 - SE
            dE = SE - EI
            dI = EI - IR
            dR = IR - 0

            S[t, r] = S[t-1, r] + dS
            E[t, r] = E[t-1, r] + dE
            I[t, r] = I[t-1, r] + dI
            R[t, r] = R[t-1, r] + dR

    return S, E, I, R, t_space


def seir_bayes_plot(N, E0, I0, R0,
                    R0__loc, R0__scale,
                    gamma_loc, gamma_scale,
                    alpha_loc, alpha_scale,
                    t_max, runs, S, E, I, R, t_space):
    S0 = N - (I0 + R0 + E0)
    # plot
    algorithm_text = (
        f"for {runs} runs, do:\n"
        f"\t$S_0={S0}$\n\t$E_0={E0}$\n\t$I_0={I0}$\n\t$R_0={R0}$\n"
        f"\t$\\gamma \\sim Normal(\mu={gamma_loc:.04}, \\sigma={gamma_scale:.04})$\n"
        f"\t$\\alpha \\sim Normal(\mu={alpha_loc:.04}, \\sigma={alpha_scale:.04})$\n"
        f"\t$R0 \\sim Normal(\mu={R0__loc:.04}, \\sigma={R0__scale:.04})$\n"
        f"\t$\\beta = \\gamma R0$\n"
        f"\tSolve SEIR$(\\alpha, \\gamma, \\beta)$"
    )

    title = '(RESULTADO PRELIMINAR) Pessoas afetadas pelo COVID-19, segundo o modelo SEIR-Bayes'
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16,9))
    plt.plot(t_space, E.mean(axis=1), '--', t_space, I.mean(axis=1), '--', marker='o')
    plt.title(title, fontsize=20)
    plt.legend(['Expostas ($\mu \pm \sigma$)',
                'Infectadas ($\mu \pm \sigma$)'],
               fontsize=20, loc='lower right')
    plt.xlabel('t (Dias a partir de 16/Março/2020)', fontsize=20)
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


if __name__ == '__main__':
    N = 13_000_000
    E0, I0, R0 = 50, 152, 1
    R0__loc = 2.2
    R0__scale = make_normal_scale(1.96, 2.55, .95, R0__loc)
    gamma_loc = 1/10
    gamma_scale = make_normal_scale(1/14, 1/7, .95, gamma_loc)
    alpha_loc = 1/5.2
    alpha_scale = make_normal_scale(1/7, 1/4.1, .95, alpha_loc)
    t_max = 30*6
    runs = 100
    S, E, I, R, t_space = run_SEIR_BAYES_model(
                                      N, E0, I0, R0,
                                      R0__loc, R0__scale,
                                      gamma_loc, gamma_scale,
                                      alpha_loc, alpha_scale,
                                      t_max, runs)

    fig = seir_bayes_plot(N, E0, I0, R0,
                          R0__loc, R0__scale,
                          gamma_loc, gamma_scale,
                          alpha_loc, alpha_scale,
                          t_max, runs, S, E, I, R, t_space)
    plt.show()
