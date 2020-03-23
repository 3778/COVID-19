import pandas as pd
import streamlit as st

@st.cache
def load_uf_pop_data(load_local_data=False):
    if load_local_data:
        path=''
    else:
        path='https://raw.githubusercontent.com/3778/COVID-19/master/'
    return pd.read_csv(path+'data/csv/population/by_uf/by_uf.csv')


@st.cache
def load_city_pop_data(load_local_data=False):
    if load_local_data:
        path=''
    else:
        path='https://raw.githubusercontent.com/3778/COVID-19/master/'
    return pd.read_csv(path+'data/csv/population/by_city/by_city.csv')


def load_pop_data(granularity):
    if granularity == 'Estado':
        return (load_uf_pop_data()
                .assign(uf_city = lambda df: df['uf'])
                )
    elif granularity == 'Município':
        return (load_city_pop_data()
                .assign(uf_city = lambda df: df['city'])
                )
    else:
        raise Exception(f'{granularity} is an invalid value for granularity. It must be Estado or Município.')


@st.cache
def load_uf_covid_data(load_local_data=False):
    if load_local_data:
        path=''
    else:
        path='https://raw.githubusercontent.com/3778/COVID-19/master/'
    return pd.read_csv(path+'data/csv/covid_19/by_uf/by_uf.csv')


@st.cache
def load_city_covid_data(load_local_data=False):
    if load_local_data:
        path=''
    else:
        path='https://raw.githubusercontent.com/3778/COVID-19/master/'
    return pd.read_csv(path+'data/csv/covid_19/by_city/by_city.csv')


def load_covid_data(granularity):
    if granularity == 'Estado':
        return (load_uf_covid_data()
                .assign(uf_city = lambda df: df['uf'])
                )
    elif granularity == 'Município':
        return (load_city_covid_data()
                .assign(uf_city = lambda df: df['city'])
                )
    else:
        raise Exception(f'{granularity} is an invalid value for granularity. It must be Estado or Município.')


@st.cache
def query_ufs():
    return list(sorted(load_uf_covid_data()['uf'].unique()))


@st.cache
def query_cities():
    return list(load_city_covid_data()['city'].unique())


def query_uf_city(granularity):
    if granularity == 'Estado':
        query = query_ufs()
        query_index = query.index('SP')
    elif granularity == 'Município':
        query = query_cities()
        query_index = query.index('São Paulo')
    else:
        query = ['(Selecione Unidade)']
    return query, query_index


def query_dates(value,
                granularity):
    '''
    Query dates with codiv-19 cases for a given uf or city
    '''
    dates_list = (load_covid_data(granularity)
                    .query('uf_city == @value')
                    ['date']
                    .unique()
                    )
    return dates_list, len(dates_list)-1


def query_N(value: 'query uf/city value',
            granularity):
    N = (load_pop_data(granularity)
            .query('uf_city == @value')
            [['uf_city','estimated_population']]
            .values[0][1])
    return N


def query_I0(value: 'query uf/city value',
             date: 'query uf date',
             granularity):
    I0 = (load_covid_data(granularity)
            .query('uf_city == @value')
            .query('date == @date')
            [['uf_city', 'cases']]
            .values
            [0][1]
            )
    return int(I0)


def estimate_R0(value: 'query uf/city value',
             date: 'query uf date',
             granularity):
    '''
    Considering: cases(t) = cases(t-1) + new_cases(t) - removed(t)
    ∴
    removed(t) = cases(t-1) + new_cases(t) - cases(t)
    '''
    R0 = (load_covid_data(granularity)
            .query('uf_city == @value')
            .assign(cases_tminus_1=lambda df: df.cases.shift(1).fillna(0))
            .assign(removed=lambda df: df.cases_tminus_1 + df.new_cases - df.cases)
            .query('date == @date')
            [['uf_city', 'removed']]
            .values
            [0][1]
            )
    return int(R0)


def estimate_E0(value: 'query uf/city value',
                date: 'query uf date',
                granularity,
                method='avg_history'):
    '''
    Premises: Exposed(t)>=New_cases(t-avg_incubation_time)
    uses the last valid E for dates which value is null
    '''
    if method == 'avg_history':
        avg_incubation_time = 5
        E0 = (load_covid_data(granularity)
                .query('uf_city == @value')
                .assign(exposed=lambda df: df.cases
                                            .shift(-avg_incubation_time)
                                            .fillna(method='ffill')
                                            .fillna(0))  
                .query('date == @date')
                [['uf_city', 'exposed']]
                .values
                [0][1]
                )
    elif method=='double':
        column = 'uf' if granularity == 'Estado' else 'city'
        E0 = (load_covid_data(granularity)
              .query('uf_city == @value')
              .assign(exposed=lambda df: 2*df.cases)
              .query('date == @date')
              [['uf_city', 'exposed']]
              .values
              [0][1])
    elif method == 'gradient_transient':
        g = 1/10
        a = 1/5.2
        b = g*3
        E0_func = lambda S, N: (g - a + np.sqrt((g - a)**2 + 4*a*b*S/N)/(2*a))
    else: 
        E0 = 152
    return int(E0)      


def query_params(value: 'query uf/city value',
                 date: 'query uf date',
                 granularity,
                 E0_method='avg history'):
    '''
    Query N, I(0), E(0) and R(0) parameters based on historic data
    for a given uf and date

    '''
    N = query_N(value, granularity)
    E0 = estimate_E0(value, date, granularity, E0_method) # temporary workaround
    I0 = query_I0(value, date, granularity)
    R0 = estimate_R0(value, date, granularity)
    return N, E0, I0, R0
