import pandas as pd
from covid19.models import SEIRBayes
from covid19.data import load_cases, load_population

if __name__ == '__main__':

    def generate_sier(
                      # Municipio
                      city, 
                      date,

                      # Condições iniciais
                      S0 = None, # População total (N)
                      I0 = None, # Indivíduos infecciosos inicialmente (I0)
                      E0 = None, # Indivíduos expostos inicialmente (E0)
                      R0 = 0, # Indivíduos removidos com imunidade inicialmente (R0)
                      
                      # R0, período de infecção (1/γ) e tempo incubação (1/α)
                      r0_inf = 1.9, # Limite inferior do número básico de reprodução médio (R0)
                      r0_sup = 5, # Limite superior do número básico de reprodução médio (R0)
                      reduce_r0 = [0, 0.25, 0.50, 0.65],

                      gamma_inf = 10, # Limite inferior do período infeccioso médio em dias (1/γ)
                      gamma_sup = 14, # Limite superior do período infeccioso médio em dias (1/γ)
                      alpha_inf = 4.2, # Limite inferior do tempo de incubação médio em dias (1/α)
                      alpha_sup = 5, # Limite superior do tempo de incubação médio em dias (1/α)

                      # Parâmetros gerais
                      t_max = 180, # Período de simulação em dias (t_max)
                      sample_size = 1000 #Qtde. de iterações da simulação (runs)

                      ): 

        cases = load_cases('city')[city]
        population = load_population('city')[city]

        S0 = S0 or population
        I0 = I0 or cases.loc[date]['totalCases']
        E0 = E0 or 2*I0
        R0 = R0 or 0

        sample_size = sample_size or 1000
        t_max = t_max or 180

        for reduce_by in reduce_r0:
            model = SEIRBayes.init_from_intervals(NEIR0=(S0, E0, I0, R0),
                                                  r0_interval=((1-reduce_by)*r0_inf, (1-reduce_by)*r0_sup, 0.95),
                                                  gamma_inv_interval=(gamma_inf, gamma_sup, 0.95),
                                                  alpha_inv_interval=(alpha_inf, alpha_sup, 0.95),
                                                  t_max=t_max)

            S, E, I, R, t = model.sample(sample_size)
            pred = pd.DataFrame(index=(pd.date_range(start=date, periods=t.shape[0])
                                        .strftime('%Y-%m-%d')),
                                data={'S': S.mean(axis=1),
                                    'E': E.mean(axis=1),
                                    'I': I.mean(axis=1),
                                    'R': R.mean(axis=1)})

            df = (pred
                .join(cases, how='outer')
                .assign(cases=lambda df: df.totalCases.fillna(df.I))
                .assign(newly_infected=lambda df: df.cases - df.cases.shift(1) + df.R - df.R.shift(1))
                .assign(newly_R=lambda df: df.R.diff())
                .rename(columns={'cases': 'totalCases OR I'}))

            df = df.assign(days=range(1, len(df) + 1))

            print(model._params)

            df.to_csv(f'seir_output-{reduce_by}_2.csv')


    generate_sier('Rio de Janeiro/RJ',
                  '2020-03-24')