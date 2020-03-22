import pandas as pd
import streamlit as st

ufs = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE',
       'DF', 'ES', 'GO', 'MA', 'MG', 'MS',
       'MT', 'PA', 'PB', 'PE', 'PI', 'PR',
       'RJ', 'RN', 'RO', 'RR', 'RS', 'SC',
       'SE', 'SP', 'TO']


@st.cache
def load_uf_pop_data():
    return (pd.read_csv('data/csv/uf_population/uf_population.csv')
              .set_index('UF'))


@st.cache
def load_uf_covid_data():
    path = 'data/csv/by_uf/'
    return {file.split('.')[0]: pd.read_csv(path+file+'.csv') for file in ufs}


@st.cache
def query_ufs():
    uf_covid_data = load_uf_covid_data()
    valid_uf_list = [key for key in uf_covid_data.keys() if uf_covid_data[key]['cases'].sum() > 0]
    valid_uf_list.insert(0, '(Selecione)')
    return valid_uf_list


def query_dates(uf):
    '''
    Query dates with codiv-19 cases for a given uf
    '''
    if uf == '(Selecione)':
        return ['(Selecione)'], 0
    else:
        uf_covid_data = load_uf_covid_data()
        dates_list =  uf_covid_data[uf].query('cases > 0')['date'].to_list()
        return dates_list, len(dates_list)-1


def query_params(uf: 'query uf',
                 date: 'query uf date',
                 use_capital: True):
    '''
    Query N, I(0), E(0) and R(0) parameters based on historic data
    for a given uf and date

    '''
    uf_pop_data = load_uf_pop_data()
    uf_covid_data = load_uf_covid_data()

    population_col = 'estimated_population'
    if use_capital:
        population_col = 'capital_estimated_population'

    N = uf_pop_data.loc[uf][population_col]
    I0 = uf_covid_data[uf].query('date == @date')['cases'].values[0]
    E0 = I0 #temporary workaround
    R0 = uf_covid_data[uf].query('date == @date')['deaths'].values[0]
    return N, E0, I0, R0