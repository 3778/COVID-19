import numpy as np
import numpy.random as npr
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

def run_SEIR_SDE_model(
        N: 'population size',
        E0: 'init. exposed population',
        I0: 'init. infected population',
        R0: 'init. removed population',
        beta: 'infection rate',
        gamma: 'removal rate', 
        alpha_inv: 'incubation period',
        t_max: 'numer of days to run',
        runs: 'number of runs'
    ) -> pd.DataFrame:

    S0 = N - I0 - R0 - E0
    t_space = np.arange(0, t_max)
    S, E, I, R = S0, E0, I0, R0 
    alpha = 1/alpha_inv
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

    return S, E, I, R
        

if __name__ == '__main__':
    N = 13_000_000
    E0, I0, R0 = 50, 152, 1
    R0_ = 3.2
    gamma = 1/7
    alpha_inv = 5.2
    beta = R0_*gamma
    t_max = 21 
    runs = 100
    S, E, I, R = run_SEIR_SDE_model(N, E0, I0, R0, beta, gamma, alpha_inv, t_max, runs)

    # plot
    plt.style.use('ggplot')
    results = pd.DataFrame({'I': I.mean(axis=1), 'E': E.mean(axis=1)})
    (results
     [['E', 'I']]
     .plot(figsize=(16,9), fontsize=20, logy=False, style='o--'))
    title = (
        'Numero de Pessoas Atingidas com modelo:\n'
        f'SEIR-SDE($\gamma$={gamma:.02}, $\\beta$={beta:.02}, $R0$={R0_:.02}, '
        f'$\\alpha$={1/alpha_inv:.02}, $N$={N}, '
        f'$E(0)$={E0}, $I(0)$={I0}, $R(0)$={R0})'
    )
    plt.title(title, fontsize=20)
    plt.legend(['Expostas ($\pm 3\sigma$)', 'Infectadas ($\pm 3\sigma$)'], fontsize=20)
    plt.xlabel('Dias (a partir de 16/Março/2020)', fontsize=20)
    plt.ylabel('Pessoas', fontsize=20)
    plt.fill_between(results.index, 
                     I.mean(axis=1) + 3*I.std(axis=1), 
                     I.mean(axis=1) - 3*I.std(axis=1),
                     color='b', alpha=0.2)
    plt.fill_between(results.index, 
                     E.mean(axis=1) + 3*E.std(axis=1), 
                     E.mean(axis=1) - 3*E.std(axis=1),
                     color='r', alpha=0.2)
    plt.xticks([t for t in results.index if t % 2 != 0])
    plt.show()
