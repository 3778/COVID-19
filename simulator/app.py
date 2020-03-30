import altair as alt
import streamlit as st
import texts
import base64
import pandas as pd
import numpy as np
from covid19 import data
from covid19.models import SEIRBayes
from hospital_queue.queue_simulation import run_queue_simulation
from viz import prep_tidy_data_to_plot, make_combined_chart
from formats import global_format_func


MIN_CASES_TH = 10
DEFAULT_CITY = 'São Paulo/SP'
DEFAULT_STATE = 'SP'
DEFAULT_PARAMS = {
    'fator_subr': 40.0,
    'gamma_inv_intervals': (7.0, 14.0, 0.95),
    'alpha_inv_intervals': (4.1, 7.0, 0.95),
    'r0_intervals': (2.5, 6.0, 0.95),

    #Simulations params
    'bge_code': 355030,
    'length_of_stay_covid': 10,
    'length_of_stay_covid_uti': 7,
    'icu_rate': .1,
    'icu_rate_after_bed': .115,

    'total_beds': 12222,
    'total_beds_icu': 2421,
    'occupation_rate': .8,
    'occupation_rate_icu': .8
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

#TODO: refactor seir_for_queue method
def seir_for_queue(model):
        for reduce_by in reduce_r0: #remove

                S, E, I, R, t = model.sample(sample_size)
                pred = pd.DataFrame(index=(pd.date_range(start=date, periods=t.shape[0])
                                                .strftime('%Y-%m-%d')),
                                        data={'S': S.mean(axis=1),
                                        'E': E.mean(axis=1),
                                        'I': I.mean(axis=1),
                                        'R': R.mean(axis=1)})

                df = (pred
                        .join(cases, how='outer')
                        .assign(cases=lambda df: df.totalCases.fillna(df.I))
                        .assign(newly_infected=lambda df: df.cases - df.cases.shift(1) + df.R - df.R.shift(1))
                        .assign(newly_R=lambda df: df.R.diff())
                        .rename(columns={'cases': 'totalCases OR I'}))

                df = df.assign(days=range(1, len(df) + 1))
        return df

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
            't_max': t_max,
            'NEIR0': (N, E0, I0, R0)}

def make_param_widgets_hospital_queue(defaults=DEFAULT_PARAMS):

    st.sidebar.markdown('#### Parâmetros da simulação hospitalar')

    los_covid = st.sidebar.number_input(
            'Tempo de estadia médio no leito comum (dias)',
             step=1,
             min_value=1,
             max_value=100,
             value=DEFAULT_PARAMS['length_of_stay_covid'])

    los_covid_icu = st.sidebar.number_input(
             'Tempo de estadia médio na UTI (dias)',
             step=1,
             min_value=1,
             max_value=100,
             value=DEFAULT_PARAMS['length_of_stay_covid_uti'])

    icu_rate = st.sidebar.number_input(
             'Taxa de pacientes encaminhados para UTI diretamente',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['icu_rate'])

    icu_after_bed = st.sidebar.number_input(
             'Taxa de pacientes encaminhados para UTI a partir dos leitos',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['icu_rate_after_bed'])
    
#     total_beds = st.sidebar.number_input(
#              'Quantidade de leitos',
#              step=1,
#              min_value=0,
#              max_value=int(1e7),
#              value=DEFAULT_PARAMS['total_beds'])
    
#     total_beds_icu = st.sidebar.number_input(
#              'Quantidade de leitos de UTI',
#              step=1,
#              min_value=0,
#              max_value=int(1e7),
#              value=DEFAULT_PARAMS['total_beds_icu'])

    occupation_rate = st.sidebar.number_input(
             'Proporção de leitos disponíveis',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['occupation_rate'])

    icu_occupation_rate = st.sidebar.number_input(
             'Proporção de leitos de UTI disponíveis',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['occupation_rate_icu'])

    ibge_code = data.get_ibge_code(place_string[0], place_string[1])
    return {"ibge_code": ibge_code,
            "los_covid": los_covid,
            "los_covid_icu": los_covid_icu,
            "icu_rate": icu_rate,
            "icu_after_bed": icu_after_bed,
            #"total_beds": total_beds,
            #"total_beds_icu": total_beds_icu,
            "occupation_rate": occupation_rate,
            "icu_occupation_rate": icu_occupation_rate}

@st.cache
def make_NEIR0(cases_df, population_df, place, date):
    N0 = population_df[place]
    I0 = cases_df[place]['totalCases'][date]
    E0 = 2*I0
    R0 = 0
    return (N0, E0, I0, R0)


def make_download_df_href(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    size = (3*len(b64)/4)/(1_024**2)
    return f"""
    <a download='{filename}'
       href="data:file/csv;base64,{b64}">
       Clique para baixar ({size:.02} MB)
    </a>
    """

def make_EI_df(model_output, sample_size):
    _, E, I, _, t = model_output
    size = sample_size*model.params['t_max']
    return (pd.DataFrame({'Exposed': E.reshape(size),
                          'Infected': I.reshape(size),
                          'run': np.arange(size) % sample_size})
              .assign(day=lambda df: (df['run'] == 0).cumsum() - 1))

def plot(model_output, scale, show_uncertainty):
    _, E, I, _, t = model_output
    source = prep_tidy_data_to_plot(E, I, t)
    return make_combined_chart(source, 
                               scale=scale, 
                               show_uncertainty=show_uncertainty)

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

    place_string = (str(w_place).split("/"))

    options_date = make_date_options(cases_df, w_place)
    w_date = st.sidebar.selectbox('Data',
                                  options=options_date,
                                  index=len(options_date)-1)
    NEIR0 = make_NEIR0(cases_df, population_df, w_place, w_date)

    w_params = make_param_widgets(NEIR0)

    sample_size = st.sidebar.number_input(
            'Qtde. de iterações da simulação (runs)',
            min_value=1, max_value=3_000, step=100,
            value=300)

    use_hospital_queue = st.sidebar.checkbox('Simular fila hospitalar')

    st.markdown(texts.MODEL_INTRO)
    w_scale = st.selectbox('Escala do eixo Y',
                           ['log', 'linear'],
                           index=1)
    w_show_uncertainty = st.checkbox('Mostrar intervalo de confiança', 
                                     value=True)
    model = SEIRBayes.init_from_intervals(**w_params)
    model_output = model.sample(sample_size)
    ei_df = make_EI_df(model_output, sample_size)
    fig = plot(model_output, w_scale, w_show_uncertainty)
    st.altair_chart(fig)

    download_placeholder = st.empty()
    if download_placeholder.button('Preparar dados para download em CSV'):
        href = make_download_df_href(ei_df, 'covid-simulator.3778.care.csv')
        st.markdown(href, unsafe_allow_html=True)
        download_placeholder.empty()

    intervals = [w_params['alpha_inv_interval'],
                 w_params['gamma_inv_interval'],
                 w_params['r0_interval']]
    SEIR0 = model._params['init_conditions']
    st.markdown(texts.make_SIMULATION_PARAMS(SEIR0, intervals))
    st.button('Simular novamente')
    st.markdown(texts.SIMULATION_CONFIG)

    if use_hospital_queue:
        params_simulation = make_param_widgets_hospital_queue()

        _, E, I, _, t = model_output
        dataset = prep_tidy_data_to_plot(E, I, t)
        dataset_mean = dataset[['day', 'Infected_mean']].copy()
        dataset_mean = dataset_mean.assign(hospitalizados=round(dataset_mean['Infected_mean']*0.14))
        st.write(dataset_mean.head())
        st.write(dataset_mean.count())
        st.write(dataset_mean.shape)
        simulation_mean = run_queue_simulation(dataset_mean, params_simulation)
        simulation_mean = simulation_mean.loc[:,['Occupied_beds', 'Queue', 'ICU_Occupied_beds', 'ICU_Queue']]

        st.markdown(texts.HOSPITAL_QUEUE_SIMULATION)
        st.area_chart(simulation_mean)

        href = make_download_df_href(simulation_mean, 'queue-simulator.3778.care.csv')
        st.markdown(href, unsafe_allow_html=True)

    st.markdown(texts.DATA_SOURCES)