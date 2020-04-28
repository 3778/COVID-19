import streamlit as st

from st_utils import texts
from covid19 import data

import pandas as pd
import numpy as np
import math
import base64
from datetime import timedelta
from json import dumps

from covid19.models import SEIRBayes
from st_utils.viz import prep_tidy_data_to_plot, make_combined_chart, plot_r0

MIN_DEATH_SUBN = 3
MIN_DATA_BRAZIL = '2020-03-26'
FATAL_RATE_BASELINE = 0.00657 #Verity R, Okell LC, Dorigatti I et al. Estimates of the severity of covid-19 disease. medRxiv 2020.

SAMPLE_SIZE = 300

DEFAULT_PARAMS = {
    'fator_subr': 10,
    'gamma_inv_dist': (7.0, 14.0, 0.95, 'lognorm'),
    'alpha_inv_dist': (4.1, 7.0, 0.95, 'lognorm'),
    'r0_dist': (2.5, 6.0, 0.95, 'lognorm'),
    't_max': 180
}

def plot_EI(model_output, scale, start_date):
    _, E, I, _, t = model_output
    source = prep_tidy_data_to_plot(E, I, t, start_date)
    return make_combined_chart(source, 
                               scale=scale, 
                               show_uncertainty=True)

@st.cache(show_spinner=False, suppress_st_warning=True)
def make_EI_df(model_output, sample_size, t_max, date):
    _, E, I, R, t = model_output
    size = sample_size*t_max

    NI = np.add(pd.DataFrame(I).apply(lambda x: x - x.shift(1)).values,
                pd.DataFrame(R).apply(lambda x: x - x.shift(1)).values)

    df = (pd.DataFrame({'Exposed': E.reshape(size),
                        'Infected': I.reshape(size),
                        'Recovered': R.reshape(size),
                        'Newly Infected': NI.reshape(size),
                        'Run': np.arange(size) % sample_size}
                        )
              .assign(Day=lambda df: (df['Run'] == 0).cumsum() - 1))

    return df.assign(
        Date=df['Day']
            .apply(lambda x: pd.to_datetime(date) + timedelta(days=(x))))

def uf(sd,mean):
    u = math.log((math.pow(mean,2))/(math.sqrt(math.pow(sd,2)+math.pow(mean,2))))
    return u

def sf(sd,mean):
    s = math.sqrt(math.log(1+(math.pow(sd/mean,2))))
    return s

def death_distrib_integral(x,u,s):
    fx = (-1/2)*math.erf((u-math.log(x))/(s*(math.sqrt(2))))
    return fx

def day_death_prob(day,mean,sd):
    init = day
    final = day+1
    if init == 0:
        init = 0.001
    u = uf(sd, mean)
    s = sf(sd,mean)
    prob = death_distrib_integral(final,u,s)-death_distrib_integral(init,u,s)
    return prob

def calculateCCFR(cases):

    mean = 13  # Linton NM, Kobayashi T, Yang Y et al. Incubation period and other
    sd = 12.7  # epidemiological characteristics of 2019 novel coronavirus infections
    # with right truncation: A statistical analysis of publicly available case data.
    # Journal of Clinical Medicine 2020;9:538.

    estim_deaths = 0
    cases_confirm = 0
    deaths_confirm = 0

    for i in range(cases.shape[0]):
        cases_confirm += cases['newCases'].iloc[i]
        deaths_confirm += cases['deaths'].iloc[i]
        for j in range(cases.shape[0] - i + 1):
            estim_deaths += cases['newCases'].iloc[i + j - 1] * day_death_prob(j, mean, sd)

    u = estim_deaths / cases_confirm
    cCFR = deaths_confirm / (cases_confirm * u)

    return cCFR

def subnotification(cases):

    cCFR_place = calculateCCFR(cases)
    subnotification_rate = FATAL_RATE_BASELINE/cCFR_place
    return subnotification_rate, cCFR_place


def estimate_subnotification(cases_df, place, date,w_granularity):

    if w_granularity == 'city':
        city_deaths, city_cases = data.get_city_deaths(place,date)
        state = city_cases['state'][0]
        if city_deaths < MIN_DEATH_SUBN:
            place = state
            w_granularity = 'state'

    if w_granularity == 'state':
        state_deaths, state_cases = data.get_state_cases_and_deaths(place,date)
        if state_deaths < MIN_DEATH_SUBN:
            w_granularity = 'brazil'

    if w_granularity == 'city':

        previous_cases = cases_df[place][:date]
        # Formatting previous dates
        all_dates = pd.date_range(start=MIN_DATA_BRAZIL, end=date).strftime('%Y-%m-%d')

        all_dates_df = pd.DataFrame(index=all_dates,
                                    data={"dummy": np.zeros(len(all_dates))})

        previous_cases = all_dates_df.join(previous_cases, how='outer')['newCases']
        #cut_after = previous_cases.shape[0]

        previous_cases = previous_cases.fillna(0)
        previous_cases = pd.DataFrame(previous_cases, columns=['newCases'])
        deaths,cases_df = data.get_city_deaths(place,date)

        previous_cases['deaths'] = 0
        previous_cases['deaths'][0] = deaths
        previous_cases = previous_cases.sort_index(ascending=False)

        return subnotification(previous_cases)

    if w_granularity == 'state':

        state_deaths, cases_df = data.get_state_cases_and_deaths(place,date)
        previous_cases = cases_df.sort_index(ascending=False)
        previous_cases = previous_cases.reset_index()
        total_deaths = previous_cases['deaths'][0]
        previous_cases['deaths'] = 0
        previous_cases['deaths'][0] = total_deaths

        return subnotification(previous_cases)

    if w_granularity == 'brazil':

        brazil_deaths, cases_df = data.get_brazil_cases_and_deaths(date)
        previous_cases = cases_df.sort_index(ascending=False)
        previous_cases = previous_cases.reset_index()
        total_deaths = previous_cases['deaths'][0]
        previous_cases['deaths'] = 0
        previous_cases['deaths'][0] = total_deaths

        return subnotification(previous_cases)

def make_NEIR0(cases_df, population_df, place, date,reported_rate):

    N0 = population_df[place]
    EIR = cases_df[place]['totalCases'][date]
    
    return (N0, EIR)

def make_download_href(df, params, should_estimate_r0, r0_dist):
    _params = {
        'subnotification_factor': params['fator_subr'],
        'incubation_period': {
            'lower_bound': params['alpha_inv_dist'][0],
            'upper_bound': params['alpha_inv_dist'][1],
            'density_between_bounds': params['alpha_inv_dist'][2]
         },
        'infectious_period': {
            'lower_bound': params['gamma_inv_dist'][0],
            'upper_bound': params['gamma_inv_dist'][1],
            'density_between_bounds': params['gamma_inv_dist'][2]
         },
    }
    if should_estimate_r0:
        _params['reproduction_number'] = {
            'samples': list(r0_dist)
        }
    else:
        _params['reproduction_number'] = {
            'lower_bound': r0_dist[0],
            'upper_bound': r0_dist[1],
            'density_between_bounds': r0_dist[2]
        }
    csv = df.to_csv(index=False)
    b64_csv = base64.b64encode(csv.encode()).decode()
    b64_params = base64.b64encode(dumps(_params).encode()).decode()
    size = (3*len(b64_csv)/4)/(1_024**2)
    return f"""
    <a download='covid-simulator.3778.care.csv'
       href="data:file/csv;base64,{b64_csv}">
       Clique para baixar os resultados da simulação em format CSV ({size:.02} MB)
    </a><br>
    <a download='covid-simulator.3778.care.json'
       href="data:file/json;base64,{b64_params}">
       Clique para baixar os parâmetros utilizados em formato JSON.
    </a>
    """

def make_param_widgets(NEIR0, reported_rate, r0_samples=None, defaults=DEFAULT_PARAMS):
    _N0, _EIR0 = map(int, NEIR0)
    interval_density = 0.95
    family = 'lognorm'

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Parâmetro Previsão de infectados**")
    if st.sidebar.checkbox('Parâmetros de infecção'):
        
        fator_subr = st.sidebar.number_input(
            ('Taxa de reportagem de infectados (%)'),
            min_value=0.0, max_value=100.0, step=1.0,
            value=reported_rate)

        EIR0 = st.sidebar.number_input('Indivíduos que já foram infectados e confirmados',
                                min_value=0, max_value=1_000_000_000,
                                value=_EIR0)

        N = st.sidebar.number_input('População total (N)',
                                    min_value=0, max_value=1_000_000_000, step=500_000,
                                    value=_N0)
        
    else:
        fator_subr, N, EIR0 = reported_rate, _N0, _EIR0

    if st.sidebar.checkbox('Parâmetros avançados', key='checkbox_parameters_seir'):

        gamma_inf = st.sidebar.number_input(
                'Limite inferior do período infeccioso médio em dias (1/γ)',
                min_value=1.0, max_value=60.0, step=1.0,
                value=defaults['gamma_inv_dist'][0])

        gamma_sup = st.sidebar.number_input(
                'Limite superior do período infeccioso médio em dias (1/γ)',
                min_value=1.0, max_value=60.0, step=1.0,
                value=defaults['gamma_inv_dist'][1])

        alpha_inf = st.sidebar.number_input(
                'Limite inferior do tempo de incubação médio em dias (1/α)',
                min_value=0.1, max_value=60.0, step=1.0,
                value=defaults['alpha_inv_dist'][0])

        alpha_sup = st.sidebar.number_input(
                'Limite superior do tempo de incubação médio em dias (1/α)',
                min_value=0.1, max_value=60.0, step=1.0,
                value=defaults['alpha_inv_dist'][1])
        
        t_max = st.sidebar.number_input('Período de simulação em dias (t_max)',
                                        min_value=1, max_value=8*30, step=15,
                                        value=defaults['t_max'])
        sample_size = st.sidebar.number_input('Qtde. de iterações da simulação (runs)',
            min_value=1, max_value=3000, step=100,
            value=SAMPLE_SIZE)
    else:
        gamma_inf, gamma_sup, alpha_inf, alpha_sup = (defaults['gamma_inv_dist'][0],
                                                      defaults['gamma_inv_dist'][1],
                                                      defaults['alpha_inv_dist'][0],
                                                      defaults['alpha_inv_dist'][1])
        t_max, sample_size = defaults['t_max'], 300

    return {'fator_subr': fator_subr,
            'alpha_inv_dist': (alpha_inf, alpha_sup, interval_density, family),
            'gamma_inv_dist': (gamma_inf, gamma_sup, interval_density, family),
            't_max': t_max,
            'sample_size': sample_size,
            'NEIR0': (N, EIR0)}

@st.cache(show_spinner=False, suppress_st_warning=True)
def run_seir(w_date,
             w_location,
             cases_df,
             population_df,
             w_location_granulariy,
             r0_dist,
             w_params = DEFAULT_PARAMS,
             sample_size = SAMPLE_SIZE,
             reported_rate = None,
             NEIR0 = None):

    if reported_rate is None:
        calc_reported_rate, cCFR = estimate_subnotification(cases_df,
                                                            w_location,
                                                            w_date,
                                                            w_location_granulariy)
        new_reported_rate = calc_reported_rate*100
    else:
        new_reported_rate = reported_rate

    if NEIR0 is None:
        new_NEIR0 = make_NEIR0(cases_df, population_df, w_location, w_date, reported_rate)
    else:
        new_NEIR0 = NEIR0

    w_params['r0_dist'] = r0_dist
    model = SEIRBayes(new_NEIR0,
                    w_params['r0_dist'],
                    w_params['gamma_inv_dist'],
                    w_params['alpha_inv_dist'],
                    new_reported_rate,
                    w_params['t_max'])

    model_output = model.sample(sample_size)

    return (model, model_output, sample_size, w_params['t_max']), new_reported_rate, new_NEIR0

def build_seir(w_date,
               w_location,
               cases_df,
               population_df,
               w_location_granulariy,
               r0_samples):

    reported_rate, cCFR = estimate_subnotification(cases_df,
                                                   w_location,
                                                   w_date,
                                                   w_location_granulariy)
    NEIR0 = make_NEIR0(cases_df, population_df, w_location, w_date, reported_rate)
    reported_rate = reported_rate * 100

    w_params = make_param_widgets(NEIR0, reported_rate)
    sample_size = w_params.pop('sample_size')
    r0_dist = r0_samples[:, -1] 
    
    st.markdown(texts.MODEL_INTRO)
    st.markdown("---")
    st.markdown(texts.SEIR_SIMULATION_SOURCE_EXPLAIN)
    r0_personalized = st.checkbox('Utilizar Número Básico de Reprodução personalizado',value=False)
    st.markdown("---")

    if r0_personalized:
        r0_values = st.slider('Defina o intervalo para o Número Básico de Reprodução',min_value=0.0,max_value=10.0,value=(2.0,3.0),step=0.01)
        r0_dist = r0_values[0], r0_values[1], .95, 'lognorm'

    with st.spinner("Calculando SEIR..."):
        model_info, _, _ = run_seir(w_date,
                                    w_location,
                                    cases_df,
                                    population_df,
                                    w_location_granulariy,
                                    r0_dist,
                                    w_params = w_params,
                                    sample_size = sample_size,
                                    reported_rate = w_params['fator_subr'],
                                    NEIR0=NEIR0)

    model, model_output, _ , _ = model_info

    with st.spinner("Criando visualização SEIR..."):
        ei_df = make_EI_df(model_output, sample_size, w_params['t_max'], w_date)
        st.write(texts.SEIRBAYES_DESC)
        w_scale = st.selectbox('Escala do eixo Y',
                            ['log', 'linear'],
                            index=1)
        fig = plot_EI(model_output, w_scale,w_date)
        st.altair_chart(fig)

    download_placeholder = st.empty()

    if download_placeholder.button('Preparar dados para download em CSV'):
        href = make_download_href(ei_df, w_params, True, r0_dist)
        st.markdown(href, unsafe_allow_html=True)
        download_placeholder.empty()

    dists = [w_params['alpha_inv_dist'],
            w_params['gamma_inv_dist'],
            r0_dist]

    SEIR0 = model._params['init_conditions']
    st.markdown(texts.make_SIMULATION_PARAMS(SEIR0, dists, True))
    st.markdown("---")

    return (model, model_output, sample_size, w_params['t_max']), reported_rate