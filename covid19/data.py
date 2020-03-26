import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
COVID_19_BY_CITY_URL=('https://raw.githubusercontent.com/wcota/covid19br/'
                      'master/cases-brazil-cities-time.csv')
IBGE_POPULATION_PATH=DATA_DIR / 'ibge_population.csv'

def load_cases(by):
    '''Load cases from wcota/covid19br

    Args:
        by (string): either 'state' or 'city'.

    Returns:
        pandas.DataFrame

    Examples:
        
        >>> cases_city = load_cases('city')
        >>> cases_city['São Paulo/SP']['newCases']['2020-03-20']
        99.0

        >>> cases_state = load_cases('state')
        >>> cases_state['SP']['newCases']['2020-03-20']
        109.0
        
    '''
    assert by in ['state', 'city']

    return (pd.read_csv(COVID_19_BY_CITY_URL)
              .query("state != 'TOTAL'")
              .groupby(['date', by])
              [['newCases', 'totalCases']]
              .sum()
              .unstack(by)
              .sort_index()
              .swaplevel(axis=1)
              .fillna(0))

def load_population(by):
    '''Load cases from wcota/covid19br

    Args:
        by (string): either 'state' or 'city'.

    Returns:
        pandas.DataFrame

    Examples:
        
        >>> load_population('state').head()
        state
        AC      881935
        AL     3337357
        AM     4144597
        AP      845731
        BA    14873064
        Name: estimated_population, dtype: int64

        >>> load_population('city').head()
        city
        Abadia de Goiás/GO          8773
        Abadia dos Dourados/MG      6989
        Abadiânia/GO               20042
        Abaetetuba/PA             157698
        Abaeté/MG                  23237
        Name: estimated_population, dtype: int64
        
    '''
    assert by in ['state', 'city']

    return (pd.read_csv(IBGE_POPULATION_PATH)
              .rename(columns={'uf': 'state'})
              .assign(city=lambda df: df.city + '/' + df.state)
              .groupby(by)
              ['estimated_population']
              .sum()
              .sort_index())
