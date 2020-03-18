import numpy as np
import numpy.random as npr
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from model import Model


def run_SEIR_BAYES_model(m):
    S = np.zeros(m.size)
    E = np.zeros(m.size)
    I = np.zeros(m.size)
    R = np.zeros(m.size)
    for r in range(m.runs):
        S[0, r] = m.S0
        E[0, r] = m.E0
        I[0, r] = m.I0
        R[0, r] = m.R0

        R0_ = npr.normal(m.R0__loc, m.R0__scale)
        gamma = npr.normal(m.gamma_loc, m.gamma_scale)
        alpha = npr.normal(m.alpha_loc, m.alpha_scale)
        beta = R0_*gamma
        for t in m.t_space[1:]:
            SE = npr.binomial(S[t-1, r], 1 - np.exp(-beta*I[t-1, r]/m.N))
            EI = npr.binomial(E[t-1, r], 1 - np.exp(-alpha))
            IR = npr.binomial(I[t-1, r], 1 - np.exp(-gamma))

            dS = 0 - SE
            dE = SE - EI
            dI = EI - IR
            dR = IR - 0

            S[t, r] = S[t-1, r] + dS
            E[t, r] = E[t-1, r] + dE
            I[t, r] = I[t-1, r] + dI
            R[t, r] = R[t-1, r] + dR
    m.S = S
    m.E = E
    m.I = I
    m.R = R
    return m


def seir_bayes_plot(m):
    algorithm_text = (
        f"for {m.runs} runs, do:\n"
        f"\t$S_0={m.S0}$\n\t$E_0={m.E0}$\n\t$I_0={m.I0}$\n\t$R_0={m.R0}$\n"
        f"\t$\\gamma \\sim Normal(\mu={m.gamma_loc:.04},"
        f"\\sigma={m.gamma_scale:.04})$\n"
        f"\t$\\alpha \\sim Normal(\mu={m.alpha_loc:.04},"
        f"\\sigma={m.alpha_scale:.04})$\n"
        f"\t$R0 \\sim Normal(\mu={m.R0__loc:.04},"
        f"\\sigma={m.R0__scale:.04})$\n"
        f"\t$\\beta = \\gamma R0$\n"
        f"\tSolve SEIR$(\\alpha, \\gamma, \\beta)$"
    )

    title = ('(RESULTADO PRELIMINAR) Pessoas afetadas '
             'pelo COVID-19, segundo o modelo SEIR-Bayes')
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(
        m.t_space, m.E.mean(axis=1), '--',
        m.t_space, m.I.mean(axis=1), '--',
        marker='o'
        )
    plt.title(title, fontsize=20)
    plt.legend(['Expostas ($\mu \pm \sigma$)',
                'Infectadas ($\mu \pm \sigma$)'],
               fontsize=20, loc='lower right')
    plt.xlabel('t (Dias a partir de 16/Março/2020)', fontsize=20)
    plt.ylabel('Pessoas', fontsize=20)
    plt.fill_between(m.t_space,
                     m.I.mean(axis=1) + m.I.std(axis=1),
                     (m.I.mean(axis=1) - m.I.std(axis=1)).clip(m.I0),
                     color='b', alpha=0.2)
    plt.fill_between(m.t_space,
                     m.E.mean(axis=1) + m.E.std(axis=1),
                     (m.E.mean(axis=1) - m.E.std(axis=1)).clip(m.I0),
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
    model = Model()
    model = run_SEIR_BAYES_model(model)
    fig = seir_bayes_plot(model)
    plt.show()
