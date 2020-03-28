import streamlit as st
import texts
import base64
import pandas as pd
import numpy as np
from covid19 import data
from covid19.models import SEIRBayes
from formats import global_format_func

MIN_CASES_TH = 10
DEFAULT_CITY = 'São Paulo/SP'
DEFAULT_STATE = 'SP'
DEFAULT_PARAMS = {
    'fator_subr': 40.0,
    'gamma_inv_intervals': (7.0, 14.0, 0.95),
    'alpha_inv_intervals': (4.1, 7.0, 0.95),
    'r0_intervals': (2.5, 6.0, 0.95),
}


@st.cache
def make_place_options(cases_df, population_df):
    return (cases_df
            .swaplevel(0,1, axis=1)
            ['totalCases']
            .pipe(lambda df: df >= MIN_CASES_TH)
            .any()
            .pipe(lambda s: s[s & s.index.isin(population_df.index)])
            .index)

@st.cache
def make_date_options(cases_df, place):
    return (cases_df
            [place]
            ['totalCases']
            .pipe(lambda s: s[s >= MIN_CASES_TH])
            .index
            .strftime('%Y-%m-%d'))

@st.cache
def run_model(NEIR0, intervals, sample_size, t_max):
    pass


def make_param_widgets(NEIR0, defaults=DEFAULT_PARAMS):
    _N0, _E0, _I0, _R0 = map(int, NEIR0)
    interval_density = 0.95

    fator_subr = st.sidebar.number_input(
            ('Fator de subreportagem. Este número irá multiplicar'
             'o número de infectados e expostos.'),
            min_value=1.0, max_value=200.0, step=1.0,
            value=defaults['fator_subr'])

    st.sidebar.markdown('#### Condições iniciais')
    N = st.sidebar.number_input('População total (N)',
                                min_value=0, max_value=1_000_000_000, step=500_000,
                                value=_N0)

    E0 = st.sidebar.number_input('Indivíduos expostos inicialmente (E0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_E0)

    I0 = st.sidebar.number_input('Indivíduos infecciosos inicialmente (I0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_I0)

    R0 = st.sidebar.number_input('Indivíduos removidos com imunidade inicialmente (R0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_R0)

    st.sidebar.markdown('#### R0, período de infecção (1/γ) e tempo incubação (1/α)') 

    r0_inf = st.sidebar.number_input(
             'Limite inferior do número básico de reprodução médio (R0)',
             min_value=0.01, max_value=10.0, step=0.25,
             value=defaults['r0_intervals'][0])

    r0_sup = st.sidebar.number_input(
            'Limite superior do número básico de reprodução médio (R0)',
            min_value=0.01, max_value=10.0, step=0.25,
            value=defaults['r0_intervals'][1])

    gamma_inf = st.sidebar.number_input(
            'Limite inferior do período infeccioso médio em dias (1/γ)',
            min_value=1.0, max_value=60.0, step=1.0,
            value=defaults['gamma_inv_intervals'][0])

    gamma_sup = st.sidebar.number_input(
            'Limite superior do período infeccioso médio em dias (1/γ)',
            min_value=1.0, max_value=60.0, step=1.0,
            value=defaults['gamma_inv_intervals'][1])

    alpha_inf = st.sidebar.number_input(
            'Limite inferior do tempo de incubação médio em dias (1/α)',
            min_value=0.1, max_value=60.0, step=1.0,
            value=defaults['alpha_inv_intervals'][0])

    alpha_sup = st.sidebar.number_input(
            'Limite superior do tempo de incubação médio em dias (1/α)',
            min_value=0.1, max_value=60.0, step=1.0,
            value=defaults['alpha_inv_intervals'][1])

    st.sidebar.markdown('#### Parâmetros gerais') 

    t_max = st.sidebar.number_input('Período de simulação em dias (t_max)',
                                    min_value=1, max_value=8*30, step=15,
                                    value=180)

    return {'fator_subr': fator_subr,
            'alpha_inv_interval': (alpha_inf, alpha_sup, interval_density),
            'gamma_inv_interval': (gamma_inf, gamma_sup, interval_density),
            'r0_interval': (r0_inf, r0_sup, interval_density),
            't_max': t_max}

@st.cache
def make_NEIR0(cases_df, population_df, place, date):
    N0 = population_df[place]
    I0 = cases_df[place]['totalCases'][date]
    E0 = 2*I0
    R0 = 0
    return (N0, E0, I0, R0)


def make_download_df(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    size = (3*len(b64)/4)/(1_024**2)
    href = f"""
    <a download='covid-simulator.3778.care.csv'
       href="data:file/csv;base64,{b64}">
       Download Results as CSV ({size:.02} MB)
    </a>
    """
    st.markdown(href, unsafe_allow_html=True)


def make_EI_df(model, sample_size):
    _, E, I, _, t = model.sample(sample_size)
    size = sample_size*model.params['t_max']
    return pd.DataFrame({'Exposed': E.reshape(size),
                         'Infected': I.reshape(size),
                         'day': np.arange(size) % model.params['t_max'],
                         'run': np.arange(size) // (sample_size-2)})


if __name__ == '__main__':
    st.markdown(texts.INTRODUCTION)
    st.sidebar.markdown(texts.PARAMETER_SELECTION)
    w_granularity = st.sidebar.selectbox('Unidade',
                                         options=['state', 'city'],
                                         index=1,
                                         format_func=global_format_func)

    cases_df = data.load_cases(w_granularity)
    population_df = data.load_population(w_granularity)

    DEFAULT_PLACE = (DEFAULT_CITY if w_granularity == 'city' else
                     DEFAULT_STATE)

    options_place = make_place_options(cases_df, population_df)
    w_place = st.sidebar.selectbox('Município',
                                   options=options_place,
                                   index=options_place.get_loc(DEFAULT_PLACE),
                                   format_func=global_format_func)

    options_date = make_date_options(cases_df, w_place)
    w_date = st.sidebar.selectbox('Data',
                                  options=options_date,
                                  index=len(options_date)-1)
    NEIR0 = make_NEIR0(cases_df, population_df, w_place, w_date)
    w_params = make_param_widgets(NEIR0)
    sample_size = st.sidebar.number_input(
            'Qtde. de iterações da simulação (runs)',
            min_value=1, max_value=3_000, step=100,
            value=1_000)

    st.markdown(texts.MODEL_INTRO)
    w_scale = st.selectbox('Escala do eixo Y',
                           ['log', 'linear'],
                           index=1)
    w_show_uncertainty = st.checkbox('Mostrar intervalo de confiança', 
                                     value=True)
    model = SEIRBayes.init_from_intervals(NEIR0=NEIR0, **w_params)
    ei_df = make_EI_df(model, sample_size)
    if st.button('Convert to CSV file (might make browser slow)'):
        make_download_df(ei_df)
    intervals = [w_params['alpha_inv_interval'],
                 w_params['gamma_inv_interval'],
                 w_params['r0_interval']]
    SEIR0 = model._params['init_conditions']
    st.markdown(texts.make_SIMULATION_PARAMS(SEIR0, intervals))
    st.button('Simular novamente')
    st.markdown(texts.SIMULATION_CONFIG)
    st.markdown(texts.DATA_SOURCES)
