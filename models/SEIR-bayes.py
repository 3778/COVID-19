import numpy as np
import numpy.random as npr
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shapescale_to_meanstd(shape, scale):
    mean = shape*scale
    std = mean*scale
    return mean, std

def make_shape_and_scale_from_ci(lb, up, ci):
    # TODO
    scale = 0.1
    return (lb + up)/(2*scale), scale

def run_SEIR_BAYES_model(
        N: 'population size',
        E0: 'init. exposed population',
        I0: 'init. infected population',
        R0: 'init. removed population',
        R0___shape: 'repr. rate shape',
        R0___scale: 'repr. rate scale',
        gamma_shape: 'removal rate shape',
        gamma_scale: 'removal rate scale',
        alpha_shape: 'incubation rate shape',
        alpha_scale: 'incubation rate scale',
        t_max: 'numer of days to run',
        runs: 'number of runs',
        dist: 'prior distribution'
    ):

    S0 = N - I0 - R0 - E0
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

        R0_ = dist(R0__shape, R0__scale)
        alpha = dist(alpha_shape, alpha_scale)
        gamma = dist(gamma_shape, gamma_scale)
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
        

if __name__ == '__main__':
    N = 13_000_000
    E0, I0, R0 = 50, 152, 1
    S0 = N - (E0 + I0 + R0)
    R0__shape, R0__scale = make_shape_and_scale_from_ci(1.96, 2.55, .95)
    gamma_shape, gamma_scale = make_shape_and_scale_from_ci(1/14, 1/7, .95)
    alpha_shape, alpha_scale = make_shape_and_scale_from_ci(1/6, 1/4.4, .95)
    t_max = 30*2
    runs = 1000
    S, E, I, R, t_space = run_SEIR_BAYES_model(
                                      N, E0, I0, R0, 
                                      R0__shape, R0__scale,
                                      gamma_shape, gamma_scale,
                                      alpha_shape, alpha_scale,
                                      t_max, runs, npr.gamma)
    # plot
    gamma_mean, gamma_std = shapescale_to_meanstd(gamma_shape, gamma_scale)
    alpha_mean, alpha_std = shapescale_to_meanstd(alpha_shape, alpha_scale)
    R0__mean, R0__std = shapescale_to_meanstd(R0__shape, R0__scale)
    algorithm_text = (
        f"$S_0={S0}$\n$E_0={E0}$\n$I_0={I0}$\n$R_0={R0}$\n"
        f"$\\gamma_t \\sim gamma(\mu={gamma_mean:.04}, \\sigma={gamma_std:.04})$\n"
        f"$\\alpha_t \\sim gamma(\mu={alpha_mean:.04}, \\sigma={alpha_std:.04})$\n"
        f"$R0_t \\sim gamma(\mu={R0__mean:.04}, \\sigma={R0__std:.04})$\n"
        f"$\\beta_t = \\gamma_t R0_t$"
    )

    title = 'Pessoas afetadas pelo COVID-19, segundo o modelo SEIR-Bayes'
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16,9))
    plt.plot(t_space, E.mean(axis=1), '--', t_space, I.mean(axis=1), '--', marker='o')
    plt.title(title, fontsize=20)
    plt.legend(['Expostas ($\pm \sigma$)', 'Infectadas ($\pm \sigma$)'],
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
    plt.show()
