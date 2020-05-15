import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from covid19.utils import state2initial

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
COVID_19_BY_CITY_URL=('https://raw.githubusercontent.com/wcota/covid19br/'
                      'master/cases-brazil-cities-time.csv')
IBGE_POPULATION_PATH=DATA_DIR / 'ibge_population.csv'
WORLD_POPULATION_PATH=DATA_DIR / 'country_population.csv'
COVID_SAUDE_URL = ('https://raw.githubusercontent.com/3778/COVID-19/'
                   'master/data/latest_cases_ms.csv')

FIOCRUZ_URL = 'https://bigdata-covid19.icict.fiocruz.br/sd/dados_casos.csv'


def _prepare_fiocruz_data(df, by):
    if by == 'country':
        return (df.assign(country=np.where((df['name'].str.contains('^[\wA-z\wÀ-ú]')),
                                           df['name'],
                                           None)))
                  #.replace({'country': state2initial}))

    if by == 'state':
        return (df.assign(state=np.where(df['name'].str.startswith('#BR'),
                                         df['name'].str[5:],
                                         None))
                  .replace({'state': state2initial}))
    if by == 'city':
        return (df.assign(city=np.where(df['name'].str.startswith('#Mun BR'),
                                        df['name'].str[9:],
                                        None))
                  .assign(city=lambda df: df['city'].str.rsplit(' ', 1)
                                                    .str.join('/')))

def load_cases(by, source='fiocruz'):
    '''Load cases from wcota/covid19br or covid.saude.gov.br or fiocruz

    Args:
        by (string): either 'state' or 'city'.

    Returns:
        pandas.DataFrame

    Examples:

        >>> cases_city = load_cases('city')
        >>> cases_city['São Paulo/SP']['newCases']['2020-03-20']
        47

        >>> cases_state = load_cases('state')
        >>> cases_state['SP']['newCases']['2020-03-20']
        110

        >>> cases_ms = load_cases('state', source='ms')
        >>> cases_ms['SP']['newCases']['2020-03-20']
        110

    '''
    assert source in ['ms', 'wcota', 'fiocruz']
    assert by in ['country', 'state', 'city']

    if source == 'monitora':
        assert by == 'state'
        df = (pd.read_csv(COVID_MONITORA_URL,
                          sep=';',
                          parse_dates=['date'],
                          dayfirst=True)
                .rename(columns={'casosNovos': 'newCases',
                                 'casosAcumulados': 'totalCases',
                                 'estado': 'state'}))
       
    if source == 'ms':
        assert by == 'state'
        df = (pd.read_csv(COVID_SAUDE_URL,
                          sep=';',
                          parse_dates=['date'],
                          dayfirst=True)
                .rename(columns={'casosNovos': 'newCases',
                                 'casosAcumulados': 'totalCases',
                                 'estado': 'state'}))

    elif source == 'wcota':
        df = (pd.read_csv(COVID_19_BY_CITY_URL, parse_dates=['date'])
                .query("state != 'TOTAL'"))

    elif source == 'fiocruz':
        df = (pd.read_csv(FIOCRUZ_URL, parse_dates=['date'])
                .rename(columns={'new_cases': 'newCases'})
                .pipe(_prepare_fiocruz_data, by=by)
                .assign(totalCases=lambda df: df.groupby([by])['newCases'].cumsum()))

    return (df.groupby(['date', by])
              [['newCases', 'totalCases']]
              .sum()
              .unstack(by)
              .sort_index()
              .swaplevel(axis=1)
              .fillna(0)
              .astype(int))


def load_population(by):
    ''''Load population from IBGE.

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
    assert by in ['country', 'state', 'city']

    if by == 'country':
        return (pd.read_csv(WORLD_POPULATION_PATH)
                    .groupby('country')
                    ['population']
                    .first())
    else:

        return (pd.read_csv(IBGE_POPULATION_PATH)
                   .rename(columns={'uf': 'state'})
                   .assign(city=lambda df: df.city + '/' + df.state)
                   .groupby(by)
                   ['estimated_population']
                   .sum()
                   .sort_index())
