import pandas as pd
from covid19.models import SEIRBayes
from covid19.data import load_cases, load_population

if __name__ == '__main__':
    city = 'Rio de Janeiro/RJ'
    cases = load_cases('city')[city]
    population = load_population('city')[city]

    date_for_pred = '2020-03-24'
    S0 = population
    I0 = cases.loc[date_for_pred]['totalCases']
    E0 = 2*I0
    R0 = 0

    for reduce_by in [0, 0.25, 0.50, 0.65]:
        model = SEIRBayes.init_from_intervals(NEIR0=(population, E0, I0, R0),
                                              r0_interval=((1-reduce_by)*1.9, (1-reduce_by)*5, 0.95),
                                              gamma_inv_interval=(10, 14, 0.95),
                                              alpha_inv_interval=(4.2, 5, 0.95),
                                              t_max=180)
        S, E, I, R, t = model.sample(1000)
        pred = pd.DataFrame(index=(pd.date_range(start=date_for_pred, periods=t.shape[0])
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
        print(city)
        print(model.params['r0_dist'].interval(0.95))
        print(model._params)
        df.to_csv(f'seir_output-{reduce_by}.csv')
