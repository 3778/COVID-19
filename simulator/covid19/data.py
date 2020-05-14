import pandas as pd
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
COVID_19_BY_CITY_URL = 'https://raw.githubusercontent.com/wcota/covid19br/' \
                      'master/cases-brazil-cities-time.csv'
COVID_19_BY_CITY_TOTALS_URL = 'https://raw.githubusercontent.com/wcota/covid19br/' \
                               'master/cases-brazil-cities.csv'
COVID_19_BY_STATE_URL = 'https://raw.githubusercontent.com/wcota/covid19br/'\
                        'master/cases-brazil-states.csv'
COVID_19_BY_STATE_TOTALS_URL = ('https://raw.githubusercontent.com/wcota/covid19br/'
                                'master/cases-brazil-states.csv')
IBGE_POPULATION_PATH = DATA_DIR / 'ibge_population.csv'
IBGE_CODE_PATH = DATA_DIR / 'ibge_city_state.csv'

COVID_SAUDE_URL = 'https://covid.saude.gov.br/assets/files/COVID19_'

hosp2020_hosp2010 = 4.2/3.6               #adjust for the age distribution in 2020
MIN_DATA_BRAZIL = '2020-03-26'
import streamlit as st


def load_cases(by):
    """Load cases from wcota/covid19br or covid.saude.gov.br

    :param: by either 'state' or 'city'.
    :return: pandas.DataFrame

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

    """
    assert by in ['state', 'city']

    if by == 'state':
        df = (pd.read_csv(COVID_19_BY_STATE_URL, parse_dates=['date'])
              .query("state != 'TOTAL'"))

    elif by == 'city':
        df = (pd.read_csv(COVID_19_BY_CITY_URL, parse_dates=['date'])
                .query("state != 'TOTAL'"))

    return (df.groupby(['date', by])[['newCases', 'totalCases']]
              .sum()
              .unstack(by)
              .sort_index()
              .swaplevel(axis=1)
              .fillna(0)
              .astype(int))


def load_population(by):
    """Load population from IBGE.

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

    """
    assert by in ['state', 'city']

    return (pd.read_csv(IBGE_POPULATION_PATH)
              .rename(columns={'uf': 'state'})
              .assign(city=lambda df: df.city + '/' + df.state)
              .groupby(by)['estimated_population']
              .sum()
              .sort_index())


def get_ibge_code(city, state):
    """Load cases from wcota/covid19br

    Args:
        city (string)
        state (string)

    Returns:
        pandas.DataFrame

    Examples:

        >>> get_ibge_code(city, state)
        3106200

    """
    df = pd.read_csv(IBGE_CODE_PATH)
    code = df[(df['state'] == state) & (df['city'] == city)]['cod_ibge'].values[0]

    return code


def get_ibge_codes_uf(state):
    df = pd.read_csv(IBGE_CODE_PATH)
    return df[(df['state'] == state)]['cod_ibge'].values


def get_ibge_code_list():
    df = pd.read_csv(IBGE_CODE_PATH)
    codes = df['cod_ibge'].to_list()

    return codes

def city_hospitalization(city,state):
    code = get_ibge_code(city,state)
    df = pd.read_excel('simulator/data/taxas_hospitaliz.xlsx')
    return hosp2020_hosp2010*df[(df['codigo_IBGE']==code)]['hospital_rate'].iloc[0]

def state_hospitalization(state):
    code = get_ibge_codes_uf(state)
    code = code.tolist()
    population_df = load_population('city')
    df = pd.read_excel('simulator/data/taxas_hospitaliz.xlsx')
    df_state = df[(df['codigo_IBGE'].isin(code))]
    df_state['pop_hosp'] = df_state['hospital_rate']*df_state['population']

    return hosp2020_hosp2010*df_state['pop_hosp'].sum()/df_state['population'].sum()


def get_city_previous_days(place,date):

    previous = (pd.read_csv(COVID_19_BY_CITY_URL)
          .query("city == '"+place+"' and date <='"+date+"'"))

    previous = previous.reset_index(drop=True)

    
    return previous


def get_state_previous_days(place,date):

    previous = (pd.read_csv(COVID_19_BY_STATE_URL)
          .query("state == '"+place+"' and date <='"+date+"'"))

    previous = previous.reset_index(drop=True)

    
    return previous

def get_brazil_previous_days(date):

    previous = (pd.read_csv(COVID_19_BY_STATE_URL)
          .query("state == 'TOTAL' and date <='"+date+"'"))

    previous = previous.reset_index(drop=True)

    
    return previous


def get_city_deaths(place,date):

    df = previous = (pd.read_csv(COVID_19_BY_CITY_URL)
          .query("city == '"+place+"' and date =='"+date+"'"))['deaths']

    df = df.reset_index()    

    deaths = df['deaths'][0]

    return deaths


def get_state_deaths(place, date):

    df = (pd.read_csv(COVID_19_BY_STATE_URL)
            .query("state == '"+place+"'and date =='"+date+"'"))['deaths']

    df = df.reset_index()

    deaths = df['deaths'][0]

    return deaths


def get_brazil_deaths(date):

    df = (pd.read_csv(COVID_19_BY_STATE_URL)
            .query("state == 'TOTAL' and date =='"+date+"'"))['deaths']

    df = df.reset_index()

    deaths = df['deaths'][0]

    return deaths
