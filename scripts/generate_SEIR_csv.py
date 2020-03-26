import pandas as pd
from covid19.models import SEIRBayes
from covid19.data import load_cases



date_for_pred = '2020-03-24'
cases = load_cases('city')['São Paulo/SP']
I0 = cases.loc[date_for_pred ]['totalCases']
E0 = 2*I0
R0 = 259 # coletei do http://painel.covid19br.org/
model = SEIRBayes.init_from_intervals(NEIR0=(12252023, E0, I0, R0),
                                      r0_interval=(1.9, 5, 0.95),
                                      gamma_inv_interval=(10, 14, 0.95),
                                      alpha_inv_interval=(4.2, 5, 0.95),
                                      t_max=30)
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
      .assign(newly_infected=lambda df: df.cases - df.cases.shift(1) + df.R)
      .rename(columns={'cases': 'totalCases OR I'}))
df.to_csv('SEIR_csv.py')
