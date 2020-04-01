import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
COVID_19_BY_CITY_URL=('https://raw.githubusercontent.com/wcota/covid19br/'
                      'master/cases-brazil-cities-time.csv')
IBGE_POPULATION_PATH=DATA_DIR / 'ibge_population.csv'
IBGE_CODE_PATH=DATA_DIR / 'ibge_city_state.csv'

COVID_SAUDE_URL = ('https://covid.saude.gov.br/assets/files/COVID19_')


def load_cases(by, source='wcota'):
    '''Load cases from wcota/covid19br or covid.saude.gov.br

    Args:
        by (string): either 'state' or 'city'.

    Returns:
        pandas.DataFrame

    Examples:

        >>> cases_city = load_cases('city')
        >>> cases_city['São Paulo/SP']['newCases']['2020-03-20']
        99

        >>> cases_state = load_cases('state')
        >>> cases_state['SP']['newCases']['2020-03-20']
        109

        >>> cases_ms = load_cases('state', source='ms')
        >>> cases_ms['SP']['newCases']['2020-03-20']
        110

    '''
    assert source in ['ms', 'wcota']
    assert by in ['state', 'city']

    if source == 'ms':
        assert by == 'state'
        dates = (pd.date_range(end='today', start='2020-03-31', freq='D')
                   .strftime("%Y%m%d"))
        for date in reversed(dates):
            url = f'{COVID_SAUDE_URL}{date}.csv'
            try:
                df = (pd.read_csv(url,
                                  sep=';',
                                  parse_dates=['data'],
                                  dayfirst=True)
                        .rename(columns={'data': 'date',
                                         'casosNovos': 'newCases',
                                         'casosAcumulados': 'totalCases',
                                         'estado': 'state'}))
            except:
                continue

    elif source == 'wcota':
        df = (pd.read_csv(COVID_19_BY_CITY_URL, parse_dates=['date'])
                .query("state != 'TOTAL'"))

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
    assert by in ['state', 'city']

    return (pd.read_csv(IBGE_POPULATION_PATH)
              .rename(columns={'uf': 'state'})
              .assign(city=lambda df: df.city + '/' + df.state)
              .groupby(by)
              ['estimated_population']
              .sum()
              .sort_index())

def get_ibge_code(city, state):
    '''Load cases from wcota/covid19br

    Args:
        city (string)
        state (string)

    Returns:
        pandas.DataFrame

    Examples:
        
        >>> get_ibge_code(city, state)
        3106200
        
    '''
    df = pd.read_csv(IBGE_CODE_PATH)
    code = df[(df['state']== state) & (df['city'] == city)]['cod_ibge'].values[0]

    return code
