import pandas as pd
import streamlit as st

@st.cache
def load_uf_pop_data():
    return pd.read_csv(f'data/csv/population/by_uf/by_uf.csv')


@st.cache
def load_city_pop_data():
    return pd.read_csv(f'data/csv/population/by_city/by_city.csv')


@st.cache
def load_uf_covid_data():
    return pd.read_csv(f'data/csv/covid_19/by_uf/by_uf.csv')


@st.cache
def load_city_covid_data():
    return pd.read_csv(f'data/csv/covid_19/by_city/by_city.csv')


@st.cache
def query_ufs():
    return load_uf_covid_data()['uf'].unique()


@st.cache
def query_cities():
    return load_city_covid_data()['city'].unique()

def query_uf_city(granularity):
    if granularity == 'Estado':
        query = query_ufs()
    elif granularity == 'Município':
        query = query_cities()
    else:
        query = ['(Selecione Unidade)']
    return query


def query_dates(value,
                granularity):
    '''
    Query dates with codiv-19 cases for a given uf or city
    '''
    dates_list = []
    if granularity == 'Estado':
        dates_list = (load_uf_covid_data()
                .query('uf == @value')
                ['date']
                .unique())
    elif granularity == 'Município':
        dates_list =  (load_city_covid_data()
                .query('city == @value')
                ['date']
                .unique()) 
    else:
        dates_list = [f'(Selecione a data)']
    return dates_list, len(dates_list)-1


def query_N(value: 'query uf/city value',
            granularity):
    if granularity == 'Estado':
        N = (load_uf_pop_data()
             .query('uf == @value')
             [['uf','estimated_population']]
             .values[0][1])
    elif granularity == 'Município':
        N = (load_city_pop_data()
            .query('city == @value')
            [['city', 'estimated_population']]
            .values[0][1])    
    else: 
        N = 13_000_000
    return N


def query_I0(value: 'query uf/city value',
             date: 'query uf date',
             granularity):
    if granularity == 'Estado':
        I0 = (load_uf_covid_data()
              .query('uf == @value')
              .query('date == @date')
              [['uf', 'cases']]
              .values
              [0][1]
             )
    elif granularity == 'Município':
        I0 = (load_city_covid_data()
              .query('city == @value')
              .query('date == @date')
              [['city', 'cases']]
              .values
              [0][1]
             )
    else: 
        I0 = 152
    return I0


def estimate_R0(value: 'query uf/city value',
             date: 'query uf date',
             granularity):
    '''
    Considering: cases(t) = cases(t-1) + new_cases(t) - removed(t)
    ∴
    removed(t) = cases(t-1) + new_cases(t) - cases(t)
    '''

    if granularity == 'Estado':
        R0 = (load_uf_covid_data()
              .query('uf == @value')
              .assign(cases_tminus_1=lambda df: df.cases.shift(1).fillna(0))
              .assign(removed=lambda df: df.cases_tminus_1 + df.new_cases - df.cases)
              .query('date == @date')
              [['uf', 'removed']]
              .values
              [0][1]
             )
    elif granularity == 'Município':
        R0 = (load_city_covid_data()
              .query('city == @value')
              .assign(cases_tminus_1=lambda df: df.cases.shift(1).fillna(0))
              .assign(removed=lambda df: df.cases_tminus_1 + df.new_cases - df.cases)
              .query('date == @date')
              [['city', 'removed']]
              .values
              [0][1]
             )
    else: 
        R0 = 1
    return int(R0)


def estimate_E0(value: 'query uf/city value',
                date: 'query uf date',
                granularity):
    '''
    Premises: Exposed(t)=New_cases(t-avg_incubation_time)
    uses the last valid E for dates which value is null
    '''
    avg_incubation_time = 5
    
    if granularity == 'Estado':
        E0 = (load_uf_covid_data()
              .query('uf == @value')
              .assign(exposed=lambda df: df.new_cases.shift(-avg_incubation_time).fillna(method='ffill'))  
              .query('date == @date')
              [['uf', 'exposed']]
              .values
              [0][1]
             )
    elif granularity == 'Município':
        E0 = (load_city_covid_data()
              .query('city == @value')
              .assign(exposed=lambda df: df.new_cases.shift(-avg_incubation_time).fillna(method='ffill'))  
              .query('date == @date')
              [['uf', 'exposed']]
              .values
              [0][1]
             )
    else: 
        E0 = 152
    return int(E0)      


def query_params(value: 'query uf/city value',
                 date: 'query uf date',
                 granularity):
    '''
    Query N, I(0), E(0) and R(0) parameters based on historic data
    for a given uf and date

    '''
    N = query_N(value, granularity)
    E0 = estimate_E0(value, date, granularity) # temporary workaround
    I0 = query_I0(value, date, granularity)
    R0 = estimate_R0(value, date, granularity)
    return N, E0, I0, R0