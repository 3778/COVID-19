import altair as alt
import streamlit as st
import texts
import base64
from collections import defaultdict
import pandas as pd
import numpy as np
from covid19 import data
from covid19.models import SEIRBayes
from covid19.de_simulation import run_de_simulation, strip_accents, load_age_data, load_capacity_by_city
from viz import prep_tidy_data_to_plot, make_combined_chart, plot_r0
from formats import global_format_func
from json import dumps
from covid19.estimation import ReproductionNumber


MIN_CASES_TH = 10
DEFAULT_CITY = 'São Paulo/SP'
DEFAULT_STATE = 'SP'
DEFAULT_PARAMS = {
    'fator_subr': 25.0,
    'gamma_inv_dist': (7.0, 14.0, 0.95, 'lognorm'),
    'alpha_inv_dist': (4.1, 7.0, 0.95, 'lognorm'),
    'r0_dist': (2.5, 6.0, 0.95, 'lognorm'),
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


def make_param_widgets(NEIR0, r0_samples=None, defaults=DEFAULT_PARAMS):
    _N0, _E0, _I0, _R0 = map(int, NEIR0)
    interval_density = 0.95
    family = 'lognorm'

    #st.sidebar.markdown('#### Condições iniciais')
    # N = st.sidebar.number_input('População total (N)',
    #                             min_value=0, max_value=1_000_000_000, step=500_000,
    #                             value=_N0)
    N = _N0

    E0 = st.sidebar.number_input('Indivíduos expostos inicialmente (E0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_E0)

    I0 = st.sidebar.number_input('Indivíduos infecciosos inicialmente (I0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_I0)

    R0 = st.sidebar.number_input('Indivíduos removidos com imunidade inicialmente (R0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_R0)

    st.sidebar.markdown('#### Subnotificação') 

    fator_subr = st.sidebar.number_input(
            ('Fator de subnotificação. Este número irá multiplicar'
             'o número de infectados e expostos.'),
            min_value=1.0, max_value=200.0, step=1.0,
            value=defaults['fator_subr'])

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


def estimate_r0(cases_df, place, sample_size):
    incidence = (
        cases_df
        [place]
        ['newCases']
        .asfreq('D')
        .fillna(0)
        .rename('incidence')
        .reset_index()
        .rename(columns={'date': 'dates'})
        .set_index('dates')
    )

    Rt = ReproductionNumber(incidence=incidence,
                            prior_shape=5.12, prior_scale=0.64,
                            si_pars={'mean': 7.5, 'sd': 3.4},
                            window_width=6)
    Rt.compute_posterior_parameters()
    return Rt.sample_from_posterior(sample_size=sample_size)


@st.cache
def _load_capacity_by_city():
    return load_capacity_by_city()


@st.cache
def _load_age_data():
    return load_age_data()


def _run_de_simulation(
        len_t_space,
        new_cases,
        empty_dict,
        my_bar,
        ward_capacity_times_availability_ward,
        icu_capacity_times_availability_icu,
        pdiw,
        pdii,
        fator_severidade,
        fator_mortalidade
        ):
    return run_de_simulation(
        len_t_space,
        new_cases,
        empty_dict,
        my_bar,
        ward_capacity_times_availability_ward,
        icu_capacity_times_availability_icu,
        pdiw,
        pdii,
        fator_severidade,
        fator_mortalidade)
    

if __name__ == '__main__':
    st.markdown(texts.INTRODUCTION)
    st.sidebar.markdown(texts.PARAMETER_SELECTION)
    w_granularity = 'city'

    cases_df = data.load_cases(w_granularity)
    population_df = data.load_population(w_granularity)

    DEFAULT_PLACE = (DEFAULT_CITY if w_granularity == 'city' else
                     DEFAULT_STATE)

    options_place = make_place_options(cases_df, population_df)
    w_place = st.selectbox('Município',
                                   options=options_place,
                                   index=options_place.get_loc(DEFAULT_PLACE),
                                   format_func=global_format_func)

    options_date = make_date_options(cases_df, w_place)
    w_date = options_date[len(options_date)-1]
    NEIR0 = make_NEIR0(cases_df, population_df, w_place, w_date)
    _N0, _E0, _I0, _R0 = map(int, NEIR0)
    st.write(f'a populacao é {_N0}')
    w_show_uncertainty = st.sidebar.checkbox('Mostrar intervalo de confiança', 
                                     value=True)
    sample_size = 300

#    st.markdown(texts.r0_ESTIMATION_TITLE)
    should_estimate_r0 = st.sidebar.checkbox(
            'Estimar R0 a partir de dados históricos',
            value=True)
    if should_estimate_r0:
        r0_samples = estimate_r0(cases_df[:w_date], w_place, sample_size)
#        st.markdown(texts.r0_ESTIMATION(w_place, w_date))
#        st.altair_chart(plot_r0(r0_samples, w_date, w_place));
    else:
#        st.markdown(texts.r0_ESTIMATION_DONT)
        r0_samples = None
#    st.markdown(texts.r0_CITATION)

    w_params = make_param_widgets(NEIR0, r0_samples)
    model = SEIRBayes(**w_params)
    model_output = model.sample(sample_size)
    ei_df = make_EI_df(model_output, sample_size)
#    st.markdown(texts.MODEL_INTRO)
#    st.write(texts.SEIRBAYES_DESC)
    w_scale = st.sidebar.selectbox('Escala do eixo Y',
                           ['log', 'linear'],
                           index=1)
    fig = plot(model_output, w_scale, w_show_uncertainty)
    st.altair_chart(fig)
    # download_placeholder = st.empty()
    # if download_placeholder.button('Preparar dados para download em CSV'):
    #     href = make_download_href(ei_df, w_params, should_estimate_r0)
    #     st.markdown(href, unsafe_allow_html=True)
    #     download_placeholder.empty()

    dists = [w_params['alpha_inv_dist'],
             w_params['gamma_inv_dist'],
             w_params['r0_dist']]
    SEIR0 = model._params['init_conditions']
    #st.markdown(texts.make_SIMULATION_PARAMS(SEIR0, dists,
    #                                         should_estimate_r0))
    st.button('Simular novamente')
    #st.markdown(texts.SIMULATION_CONFIG)
    #st.markdown(texts.DATA_SOURCES)

    # Giglio
    st.title('Uso de recursos e capacidade')

    st.sidebar.title('Uso de recursos e capacidade')

    city_c = strip_accents(w_place.split('/')[0]).upper()

    age_data = _load_age_data()
    age_data_c = age_data[age_data['municipio'] == city_c]
    age_data_c = age_data_c.drop(['Total', 'municipio'], axis=1)
    age_groups_to_consider = st.sidebar.multiselect(
        ('age_groups_to_consider'),
        [c for c in age_data_c.columns],
        default=[c for c in age_data_c.columns])
    age_share = age_data_c[age_groups_to_consider].sum().sum() / age_data_c.sum().sum()
    
    ward_capacity_by_city, icu_capacity_by_city = _load_capacity_by_city()

    ward_capacity = st.number_input(
        ('ward_capacity'),
        min_value=0, max_value=500000, step=1,
        value=int(ward_capacity_by_city[city_c]))

    availability_ward = st.number_input(
        ('availability ward'),
        min_value=0, max_value=100, step=1,
        value=20) / 100

    new_ward = st.number_input(
        ('new ward'),
        min_value=0, max_value=500000, step=1,
        value=0)

    icu_capacity = st.number_input(
        ('icu_capacity'),
        min_value=0, max_value=50000, step=1,
        value=int(icu_capacity_by_city[city_c]))

    availability_icu = st.number_input(
        ('availability icu'),
        min_value=0, max_value=100, step=1,
        value=50) / 100

    new_icu = st.number_input(
        ('new icu'),
        min_value=0, max_value=500000, step=1,
        value=0)

    mk_share = st.number_input(
        ('mk_share'),
        min_value=1, max_value=100, step=1,
        value=70) / 100

    fator_internacao = st.sidebar.number_input(
        ('fator internacao (ward)'),
        min_value=1, max_value=100, step=1,
        value=12) / 100

    pdiw = st.sidebar.number_input(
        ('Tempo médio ward'),
        min_value=1, max_value=50, step=1,
        value=5)

    pdii = st.sidebar.number_input(
        ('Tempo médio ICU'),
        min_value=1, max_value=50, step=1,
        value=12)

    fator_severidade = st.sidebar.number_input(
        ('fator severidade (icu)'),
        min_value=1, max_value=100, step=1,
        value=10) / 100

    fator_mortalidade = st.sidebar.number_input(
        ('fator mortalidade'),
        min_value=1, max_value=1000, step=1,
        value=15) / 1000

    discount = fator_internacao * mk_share * (1/w_params['fator_subr']) * age_share

    real_new_cases = cases_df[w_place]['newCases']
    real_total_cases = cases_df[w_place]['totalCases']

    S, E, I, R, t_space = model_output
    time_index = pd.date_range(start=w_date, periods=len(t_space))

    st.write('população infectada')
    pop_infected = pd.Series(I.mean(axis=1), index=time_index)
    st.line_chart(pop_infected)

    st.write('Casos notificados')
    total_not_cases = pd.Series((I.mean(axis=1) + R.mean(axis=1)) * (1/w_params['fator_subr']), index=time_index, name='model')
    total_not_cases_table = pd.merge(total_not_cases,
                    real_total_cases, how='outer', right_index=True, left_index=True)
    st.line_chart(total_not_cases_table)

    st.write('Casos novos notificados')
    new_not_cases = total_not_cases.diff().fillna(method='bfill').rename('model')
    new_not_cases_table = pd.merge(new_not_cases,
                            real_new_cases, how='outer', right_index=True, left_index=True)
    st.line_chart(new_not_cases_table)

    st.write('Casos totais internação')
    total_cases = pd.Series((I.mean(axis=1) + R.mean(axis=1)) * discount, index=time_index, name='total cases ward')
    st.line_chart(total_cases)

    st.write('Casos novos internação')
    new_cases = total_cases.diff().fillna(method='bfill')
    st.line_chart(new_cases)

    st.title('Simulação de capacidade')
    my_bar = st.progress(1)

    simrun = st.button('run simulation')

    if simrun:
        ward_capacity_times_availability_ward = (ward_capacity * availability_ward) + new_ward
        icu_capacity_times_availability_icu = (icu_capacity * availability_icu) + new_icu
        logger = _run_de_simulation(
            len(t_space),
            new_cases,
            defaultdict(list),
            my_bar,
            ward_capacity_times_availability_ward,
            icu_capacity_times_availability_icu,
            pdiw,
            pdii,
            fator_severidade,
            fator_mortalidade)

        st.write('Ocupação Ward')
        ward_oc = pd.Series(logger['count_ward'], index=time_index, name='Ocupação ward')
        st.line_chart(ward_oc)

        st.write('Ocupação Icu')
        icu_oc = pd.Series(logger['count_icu'], index=time_index, name='Ocupação icu')
        st.line_chart(icu_oc)
    #
        df = (
            pd
            .concat(
                [ward_oc, icu_oc],
                axis=1
            )
#            .resample('W').sum()
            .assign(**
                {'ward_capacity': int(ward_capacity_times_availability_ward),
                 'icu_capacity': int(icu_capacity_times_availability_icu)}
            )
            .assign(**
                {'saldo_ward': lambda x: x['ward_capacity'] - x['Ocupação ward'],
                 'saldo_icu': lambda x: x['icu_capacity'] - x['Ocupação icu']}
            )
        )
        df = df[['Ocupação ward', 'ward_capacity', 'saldo_ward', 'Ocupação icu', 'saldo_icu', 'icu_capacity']]

        df = pd.merge(df, real_new_cases.astype(int), right_index=True, left_index=True, how='outer')
        df = pd.merge(df, new_not_cases.astype(int), right_index=True, left_index=True, how='outer')
        
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'''
        <a download='3778care.csv'
        href="data:file/csv;base64,{b64}">
        Clique para baixar os resultados da simulação em formato CSV
        </a>
        '''
        st.markdown(href, unsafe_allow_html=True)

        mask = df['saldo_ward'] <= 0
        if mask.mean() != 0:
            ward_doomsday = pd.to_datetime(df[mask].iloc[0].name)
            ward_alert = ward_doomsday - pd.DateOffset(7)
            st.write(f'''
            ward vai acabar em {ward_doomsday.date()}, uma semana antes é {ward_alert.date()}
            ''')
        mask = df['saldo_icu'] <= 0
        if mask.mean() != 0:
            icu_doomsday = pd.to_datetime(df[df['saldo_icu'] <= 0].iloc[0].name)
            icu_alert = icu_doomsday - pd.DateOffset(7)
            st.write(f'''
            icu vai acabar em {icu_doomsday.date()}, uma semana antes é {icu_alert.date()}
            ''')
        st.table(df)
