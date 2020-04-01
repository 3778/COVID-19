import altair as alt
import streamlit as st
import texts
import base64
import pandas as pd
import numpy as np
from covid19 import data
from covid19.models import SEIRBayes
from hospital_queue.queue_simulation import run_queue_simulation
from viz import prep_tidy_data_to_plot, make_combined_chart, make_simulation_chart
from formats import global_format_func
from hospital_queue.confirmation_button import cache_on_button_press
from datetime import datetime
from viz import prep_tidy_data_to_plot, make_combined_chart, plot_r0
from formats import global_format_func
from json import dumps
from covid19.estimation import ReproductionNumber

MIN_CASES_TH = 10
MIN_DAYS_r0_ESTIMATE = 8
DEFAULT_CITY = 'Rio de Janeiro/RJ'
DEFAULT_STATE = 'RJ'
DEFAULT_PARAMS = {
    'fator_subr': 1.0,
    'gamma_inv_dist': (7.0, 14.0, 0.95, 'lognorm'),
    'alpha_inv_dist': (4.1, 7.0, 0.95, 'lognorm'),
    'r0_dist': (2.5, 6.0, 0.95, 'lognorm'),

    #Simulations params
    'length_of_stay_covid': 10,
    'length_of_stay_covid_uti': 8,
    'icu_rate': .1,
    'icu_rate_after_bed': .08,

    'total_beds': 12222,
    'total_beds_icu': 2421,
    'available_rate': .36,
    'available_rate_icu': .36
}

def prepare_for_r0_estimation(df):
    return (
        df
        ['newCases']
        .asfreq('D')
        .fillna(0)
        .rename('incidence')
        .reset_index()
        .rename(columns={'date': 'dates'})
        .set_index('dates')
    )

@st.cache
def make_brazil_cases(cases_df):
    return (cases_df
            .stack(level=1)
            .sum(axis=1)
            .unstack(level=1))



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

def make_param_widgets(NEIR0, r0_samples=None, defaults=DEFAULT_PARAMS):
    _N0, _E0, _I0, _R0 = map(int, NEIR0)
    interval_density = 0.95
    family = 'lognorm'

    fator_subr = st.sidebar.number_input(
            ('Fator de subnotificação. Este número irá multiplicar'
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

    if r0_samples is None:
        r0_inf = st.sidebar.number_input(
                 'Limite inferior do número básico de reprodução médio (R0)',
                 min_value=0.01, max_value=10.0, step=0.25,
                 value=defaults['r0_dist'][0])

        r0_sup = st.sidebar.number_input(
                'Limite superior do número básico de reprodução médio (R0)',
                min_value=0.01, max_value=10.0, step=0.25,
                value=defaults['r0_dist'][1])
        r0_dist = (r0_inf, r0_sup, interval_density, family)
    else:
        r0_inf = None
        r0_sup = None
        r0_dist = r0_samples[:, -1]

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

    st.sidebar.markdown('#### Parâmetros gerais') 

    t_max = st.sidebar.number_input('Período de simulação em dias (t_max)',
                                    min_value=1, max_value=8*30, step=15,
                                    value=180)

    return {'fator_subr': fator_subr,
            'alpha_inv_dist': (alpha_inf, alpha_sup, interval_density, family),
            'gamma_inv_dist': (gamma_inf, gamma_sup, interval_density, family),
            'r0_dist': r0_dist,
            't_max': t_max,
            'NEIR0': (N, E0, I0, R0)}

def make_param_widgets_hospital_queue(city, defaults=DEFAULT_PARAMS):
     
    def load_beds(ibge_code):
        # leitos
        beds_data = pd.read_csv('simulator/hospital_queue/data/ibge_leitos.csv', sep = ';')
        beds_data_filtered = beds_data[beds_data['cod_ibge']==ibge_code]
        beds_data_filtered.head()

        return beds_data_filtered['qtd_leitos'].values[0], beds_data_filtered['qtd_uti'].values[0]

    city, uf = city.split("/")
    qtd_beds, qtd_beds_uci = load_beds(data.get_ibge_code(city, uf))

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
    
    total_beds = st.sidebar.number_input(
             'Quantidade de leitos',
             step=1,
             min_value=0,
             max_value=int(1e7),
             value=qtd_beds)
        
    total_beds_icu = st.sidebar.number_input(
             'Quantidade de leitos de UTI',
             step=1,
             min_value=0,
             max_value=int(1e7),
             value=qtd_beds_uci)

    available_rate = st.sidebar.number_input(
             'Proporção de leitos disponíveis',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['available_rate'])

    available_rate_icu = st.sidebar.number_input(
             'Proporção de leitos de UTI disponíveis',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['available_rate_icu'])
    
    return {"los_covid": los_covid,
            "los_covid_icu": los_covid_icu,
            "icu_rate": icu_rate,
            "icu_after_bed": icu_after_bed,
            "total_beds": total_beds,
            "total_beds_icu": total_beds_icu,
            "available_rate": available_rate,
            "available_rate_icu": available_rate_icu}

@st.cache
def make_NEIR0(cases_df, population_df, place, date):
    N0 = population_df[place]
    I0 = cases_df[place]['totalCases'][date]
    E0 = 2*I0
    R0 = 0
    return (N0, E0, I0, R0)

def make_download_href(df, params, should_estimate_r0):
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
            'samples': list(params['r0_dist'])
        }
    else:
        _params['reproduction_number'] = {
            'lower_bound': params['r0_dist'][0],
            'upper_bound': params['r0_dist'][1],
            'density_between_bounds': params['r0_dist'][2]
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
    
#TODO: method for file download
#     <a download='{filename}'
#        href="data:file/csv;base64,{b64}">
#        Clique para baixar ({size:.02} MB)

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

@cache_on_button_press('Simular Modelo de Filas')
def run_queue_model(dataset, params_simulation):
        bar_text = st.empty()
        bar = st.progress(0)
        bar_text.text('Processando filas...')
        simulation_output = run_queue_simulation(dataset, bar, bar_text, params_simulation)

        bar.progress(1.)
        bar_text.text("Processamento finalizado.")
        st.markdown("### Resultados")

        return simulation_output

def calculate_input_hospital_queue(model_output, place, date):

    S, E, I, R, t = model_output

    pred = pd.DataFrame(index=(pd.date_range(start=date, periods=t.shape[0])
                                    .strftime('%Y-%m-%d')),
                            data={'S': S.mean(axis=1),
                                    'E': E.mean(axis=1),
                                    'I': I.mean(axis=1),
                                    'R': R.mean(axis=1)})

    df = (pred
            .assign(cases=lambda df: df.I.fillna(df.I))
            .assign(newly_infected=lambda df: df.cases - df.cases.shift(1) + df.R - df.R.shift(1))
            .assign(newly_R=lambda df: df.R.diff())
            .rename(columns={'cases': 'totalCases OR I'})) 

    df = df.assign(day=df.index)
    df = df[pd.notna(df.newly_infected)]

    return df

def estimate_r0(cases_df, place, sample_size, min_days, w_date):
    used_brazil = False

    incidence = (
        cases_df
        [place]
        .query("totalCases > @MIN_CASES_TH")
        .pipe(prepare_for_r0_estimation)
        [:w_date]
    )

    if len(incidence) < MIN_DAYS_r0_ESTIMATE:
        used_brazil = True
        incidence = (
            make_brazil_cases(cases_df)
            .query(f"totalCases > {10*MIN_CASES_TH}")
            .pipe(prepare_for_r0_estimation)
            [:w_date]
        )

    Rt = ReproductionNumber(incidence=incidence,
                            prior_shape=5.12, prior_scale=0.64,
                            si_pars={'mean': 4.89, 'sd': 1.48},
                            window_width=6)
    Rt.compute_posterior_parameters()
    samples = Rt.sample_from_posterior(sample_size=sample_size)
    return samples, used_brazil


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
    # w_show_uncertainty = st.checkbox('Mostrar intervalo de confiança', 
    #                                  value=True)
    w_show_uncertainty = True
    sample_size = st.sidebar.number_input(
            'Qtde. de iterações da simulação (runs)',
            min_value=1, max_value=3_000, step=100,
            value=300)

    st.markdown(texts.r0_ESTIMATION_TITLE)
    should_estimate_r0 = st.checkbox(
            'Estimar R0 a partir de dados históricos',
            value=True)
    if should_estimate_r0:
        r0_samples, used_brazil = estimate_r0(cases_df,
                                              w_place,
                                              sample_size, 
                                              MIN_DAYS_r0_ESTIMATE, 
                                              w_date)
        if used_brazil:
            st.write(texts.r0_NOT_ENOUGH_DATA(w_place, w_date))
                       
        _place = 'Brasil' if used_brazil else w_place
        st.markdown(texts.r0_ESTIMATION(_place, w_date))
                      
        st.altair_chart(plot_r0(r0_samples, w_date, 
                                _place, MIN_DAYS_r0_ESTIMATE))
        st.markdown(texts.r0_CITATION)
    else:
        st.markdown(texts.r0_ESTIMATION_DONT)
        r0_samples = None

#     w_params = make_param_widgets(NEIR0, r0_samples)
    model = SEIRBayes(**w_params)
    model_output = model.sample(sample_size)
    ei_df = make_EI_df(model_output, sample_size)
    st.markdown(texts.MODEL_INTRO)
    st.write(texts.SEIRBAYES_DESC)
    w_scale = st.selectbox('Escala do eixo Y',
                           ['log', 'linear'],
                           index=1)
    fig = plot(model_output, w_scale, w_show_uncertainty)
    st.altair_chart(fig)

    download_placeholder = st.empty()

    use_hospital_queue = st.sidebar.checkbox('Simular fila hospitalar')
    if download_placeholder.button('Preparar dados para download em CSV'):
        href = make_download_href(ei_df, w_params, should_estimate_r0)
        st.markdown(href, unsafe_allow_html=True)
        download_placeholder.empty()

    dists = [w_params['alpha_inv_dist'],
             w_params['gamma_inv_dist'],
             w_params['r0_dist']]
    SEIR0 = model._params['init_conditions']
    st.markdown(texts.make_SIMULATION_PARAMS(SEIR0, dists,
                                             should_estimate_r0))
    st.button('Simular novamente')
    st.markdown(texts.SIMULATION_CONFIG)

    #Begining of the queue simulation
    def make_download_simulation_df(df, filename):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        size = (3*len(b64)/4)/(1_024**2)
        return f"""
        <a download='{filename}'
        href="data:file/csv;base64,{b64}">
        Clique para baixar ({size:.02} MB)
        </a>
        """

    if use_hospital_queue:
        st.markdown(texts.HOSPITAL_QUEUE_SIMULATION)

        params_simulation = make_param_widgets_hospital_queue(w_place)
        dataset = calculate_input_hospital_queue(model_output ,w_place, w_date)
        dataset = dataset[['day', 'newly_infected']].copy()
        dataset = dataset.assign(hospitalizados=round(dataset['newly_infected']*0.14))
        simulation_output = run_queue_model(dataset, params_simulation)

        st.altair_chart(make_simulation_chart(simulation_output, "Occupied_beds", "Ocupação de leitos comuns"))
        st.altair_chart(make_simulation_chart(simulation_output, "ICU_Occupied_beds", "Ocupação de leitos de UTI"))
        st.altair_chart(make_simulation_chart(simulation_output, "Queue", "Fila de pacientes"))
        st.altair_chart(make_simulation_chart(simulation_output, "ICU_Queue", "Fila de pacientes UTI"))

        #TODO: change download method
        href = make_download_simulation_df(simulation_output, 'queue-simulator.3778.care.csv')
        st.markdown(href, unsafe_allow_html=True)

    st.markdown(texts.DATA_SOURCES)