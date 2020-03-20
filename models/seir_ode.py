import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# The SEIR model differential equations.
def deriv(y, t, N, beta, gamma, alpha):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = -dSdt - alpha*E
    dIdt = alpha*E - gamma*I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def run_SEIR_ODE_model(
        N: 'population size',
        E0: 'init. exposed population',
        I0: 'init. infected population',
        R0: 'init. removed population',
        beta: 'infection probability',
        gamma: 'removal probability', 
        alpha_inv: 'incubation period', 
        t_max: 'numer of days to run'
    ) -> pd.DataFrame:

    S0 = N - I0 - R0 - E0
    alpha = 1/alpha_inv

    # A grid of time points (in days)
    t = range(t_max)

    # Initial conditions vector
    y0 = S0, E0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, alpha))
    S, E, I, R = ret.T

    return pd.DataFrame({'S': S, 'E': E, 'I': I, 'R': R}, index=t)


if __name__ == '__main__':
    N = 13_000_000
    E0, I0, R0 = 0, 152, 1
    beta, gamma, alpha_inv = 1.75, 0.5, 5 
    t_max = 60
    results = run_SEIR_ODE_model(N, E0, I0, R0, beta, gamma, alpha_inv, t_max)

    # plot
    plt.style.use('ggplot')
    (results
     # .div(1_000_000)
     [['E', 'I']]
     .plot(figsize=(8,6), fontsize=20, logy=True))
    params_title = (
        f'SEIR($\gamma$={gamma}, $\\beta$={beta}, $\\alpha$={1/alpha_inv}, $N$={N}, '
        f'$E_0$={E0}, $I_0$={I0}, $R_0$={R0})'
    )
    plt.title(f'Numero de Pessoas Atingidas com modelo:\n' + params_title,
              fontsize=20)
    plt.legend(['Expostas', 'Infectadas'], fontsize=20)
    plt.xlabel('Dias', fontsize=20)
    plt.ylabel('Pessoas', fontsize=20)
    plt.savefig("ode.png")
    plt.show()
