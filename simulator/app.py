import altair as alt
import streamlit as st
import texts
import base64
from collections import defaultdict
import pandas as pd
import numpy as np
from covid19.data import load_cases, load_population
from data import (load_age_data, load_capacity,
    load_cnes_options, load_cnes_map, load_unid_map, load_unid_options,
    translate_cnes_code, translate_unid_code, fix_city)
from covid19.models import SEIRBayes
from covid19.de_simulation import run_de_simulation
from viz import prep_tidy_data_to_plot, make_combined_chart, plot_r0
from formats import global_format_func
from json import dumps
from covid19.estimation import ReproductionNumber
from constants import initial2state


SAMPLE_SIZE=500
MIN_CASES_TH = 10
MIN_DAYS_r0_ESTIMATE = 14
MIN_DATA_BRAZIL = '2020-03-26'
DEFAULT_CITY = 'São Paulo/SP'
DEFAULT_STATE = 'SP'
DEFAULT_PARAMS = {
    'fator_subr': 40.0,
    'gamma_inv_dist': (7.0, 14.0, 0.95, 'lognorm'),
    'alpha_inv_dist': (4.1, 7.0, 0.95, 'lognorm'),
    'r0_dist': (2.5, 6.0, 0.95, 'lognorm'),
}

@st.cache
def make_brazil_cases(cases_df):
    return (cases_df
            .stack(level=1)
            .sum(axis=1)
            .unstack(level=1))

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
    if r0_samples is None:
        r0_mid = st.sidebar.number_input(
                 'Número básico de reprodução médio (R0) (1/γ)',
                 min_value=0.01, max_value=10.0, step=0.25,
                 value=4.5)
        r0_inf = r0_mid - 1.5
        r0_sup = r0_mid + 1.5

        r0_dist = (r0_inf, r0_sup, interval_density, family)
    else:
        r0_inf = None
        r0_sup = None
        r0_dist = r0_samples[:, -1]

    st.sidebar.markdown('### Período infeccioso (duração da doença)') 

    gamma_mid = st.sidebar.number_input(
            'Período infeccioso médio em dias (1/γ)',
            min_value=1.0, max_value=60.0, step=1.0,
            value=10.0)
    gamma_inf = gamma_mid - 3
    gamma_sup = gamma_mid + 3

    st.sidebar.markdown('### Período de incubação') 

    alpha_mid = st.sidebar.number_input(
            'Tempo de incubação médio em dias (1/α)',
            min_value=0.1, max_value=60.0, step=1.0,
            value=5.5)
    alpha_inf = alpha_mid - 1.5
    alpha_sup = alpha_mid + 1.5

    N = _N0

    st.sidebar.markdown('### Condições iniciais')

    E0 = st.sidebar.number_input('Indivíduos expostos inicialmente (E0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_E0)

    I0 = st.sidebar.number_input('Indivíduos infecciosos inicialmente (I0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_I0)

    R0 = st.sidebar.number_input('Indivíduos removidos com imunidade inicialmente (R0)',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_R0)

    st.sidebar.markdown('### Subnotificação') 

    fator_subr = st.sidebar.number_input(
            ('Fator de subnotificação. Este número irá multiplicar '
             'o número de infectados e expostos.'),
            min_value=1.0, max_value=200.0, step=1.0,
            value=defaults['fator_subr'])

    t_max = 180

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


def plot(model_output, scale, show_uncertainty, time_index):
    _, E, I, _, t = model_output
    source = prep_tidy_data_to_plot(E, I, t, time_index)
    return make_combined_chart(source, 
                               scale=scale, 
                               show_uncertainty=show_uncertainty)


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
            .pipe(prepare_for_r0_estimation)
            [:w_date]
        )

    Rt = ReproductionNumber(incidence=incidence,
                            prior_shape=5.12, prior_scale=0.64,
                            si_pars={'mean': 4.89, 'sd': 1.48},
                            window_width=MIN_DAYS_r0_ESTIMATE - 2)
    Rt.compute_posterior_parameters()
    samples = Rt.sample_from_posterior(sample_size=sample_size)
    return samples, used_brazil





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
    st.sidebar.markdown('### R0 (número básico de reprodução)')
    st.sidebar.markdown(texts.r0_CITATION)

    w_granularity = st.sidebar.selectbox('Unidade',
                                         options=['state', 'city'],
                                         index=1,
                                         format_func=global_format_func)

    # source = 'ms' if w_granularity == 'state' else 'wcota'
    source = 'wcota'

    cases_df = load_cases(w_granularity, source)
    population_df = load_population(w_granularity)

    DEFAULT_PLACE = (DEFAULT_CITY if w_granularity == 'city' else
                     DEFAULT_STATE)

    options_place = make_place_options(cases_df, population_df)
    w_place_box = 'Município' if w_granularity == 'city' else 'Estado'

    w_place = st.selectbox(w_place_box,
                                   options=options_place,
                                   index=options_place.get_loc(DEFAULT_PLACE),
                                   format_func=global_format_func)

    options_date = make_date_options(cases_df, w_place)
    w_date = options_date[len(options_date)-1]

    NEIR0 = make_NEIR0(cases_df, population_df, w_place, w_date)
    _N0, _E0, _I0, _R0 = map(int, NEIR0)
    st.write(f'### A populacao de {w_place} é de {_N0:,} habitantes'.replace(',', '.'))
    w_show_uncertainty = True
    sample_size = 300
    should_estimate_r0 = st.sidebar.checkbox(
            'Estimar R0 a partir de dados históricos',
            value=True)
    if should_estimate_r0:
        r0_samples, used_brazil = estimate_r0(cases_df,
                                              w_place,
                                              SAMPLE_SIZE, 
                                              MIN_DAYS_r0_ESTIMATE, 
                                              w_date)
        if used_brazil:
            st.write(texts.r0_NOT_ENOUGH_DATA(w_place, w_date))
    else:
        r0_samples = None

    w_params = make_param_widgets(NEIR0, r0_samples)
    model = SEIRBayes(**w_params)
    model_output = model.sample(sample_size)

    S, E, I, R, t_space = model_output
    
    time_index = pd.date_range(start=w_date, periods=len(t_space))
    ei_df = make_EI_df(model_output, sample_size)
    
    w_scale = 'linear'
    fig = plot(model_output, w_scale, w_show_uncertainty, time_index)
    st.altair_chart(fig)

    dists = [w_params['alpha_inv_dist'],
             w_params['gamma_inv_dist'],
             w_params['r0_dist']]
    SEIR0 = model._params['init_conditions']

    st.title('Uso de recursos e capacidade')

    cnes_options = load_cnes_options()
    unid_options = load_unid_options()

    st.sidebar.title('CNES: Capacidade informada')
    st.sidebar.markdown('### Leitos de internação')
    tipos_leito_ward = st.sidebar.multiselect(
        'Selecione os tipos de leitos de internação elegíveis',
        cnes_options,
        default=['31','33','35','36','37','38','40','46','87','88'],
        format_func=translate_cnes_code)
    st.sidebar.markdown('### Vagas em CTI')
    tipos_leito_icu = st.sidebar.multiselect(
        'Selecione os tipos de vagas de CTI elegíveis',
        cnes_options,
        default=['74','75','76','77','78','79'],
        format_func=translate_cnes_code)
    st.sidebar.markdown('### Unidades elegíveis')
    unid_codes = st.sidebar.multiselect(
        'Selecione os tipos de unidades elegíveis',
        unid_options,
        default=['20','21','5','7','15'],
        format_func=translate_unid_code)

    st.sidebar.title('Grupos etários')

    if w_granularity == 'city':
        age_unity = 'municipio'
        subject_age = fix_city(w_place)
        subject_capacity = subject_age
        to_drop = 'municipio'
    else:
        age_unity = 'UF'
        subject_age = initial2state[w_place]
        subject_capacity = w_place
        to_drop = 'UF'

    age_data = load_age_data(w_granularity)
    age_data_c = age_data[age_data[age_unity] == subject_age].drop(['Total', to_drop], axis=1)
    age_options = [c for c in age_data_c.columns]

    age_groups_to_consider = st.sidebar.multiselect(
        ('Grupos etários inclusos'),
        age_options,
        default=age_options)
    age_share = age_data_c[age_groups_to_consider].sum().sum() / age_data_c.sum().sum()

    age_share = 1 if np.isnan(age_share) else age_share


    ward_capacity_by_granularity, icu_capacity_by_granularity = load_capacity(
        w_granularity, tipos_leito_ward, tipos_leito_icu, unid_codes)

    try:
        ward_city_cap = ward_capacity_by_granularity.loc[subject_capacity]
    except KeyError:
        ward_city_cap = 0
    st.write(f'### A quantidade informada de leitos SUS é: {ward_city_cap:,}'.replace(',', '.'))
    ward_capacity = st.number_input(
        ('Capacidade de internação instalada (número de leitos contingenciados)'),
        min_value=0, max_value=500000, step=1,
        value=int(ward_city_cap))

    availability_ward = 1
    new_ward = st.number_input(
        ('Novos leitos internação (número de novos leitos abertos)'),
        min_value=0, max_value=500000, step=1,
        value=0)

    try:
        icu_city_cap = icu_capacity_by_granularity.loc[subject_capacity]
    except KeyError:
        icu_city_cap = 0
    st.write(f'### A quantidade informada de vagas de CTI SUS é: {icu_city_cap:,}'.replace(',', '.'))
    icu_capacity = st.number_input(
        ('Capacidade de CTI instalada (número de vagas contingenciadas)'),
        min_value=0, max_value=50000, step=1,
        value=int(icu_city_cap))

    availability_icu = 1
    new_icu = st.number_input(
        ('Novas vagas em CTI (número de novas vagas abertas)'),
        min_value=0, max_value=500000, step=1,
        value=0)

    st.write('### População SUS dependente')
    mk_share = st.number_input(
        ('População SUS dependente (em %, proporção da população que usa apenas o SUS)'),
        min_value=1, max_value=100, step=1,
        value=70) / 100

    st.sidebar.title('Características da doença: COVID-19')
    fator_internacao = st.sidebar.number_input(
        ('Taxa de internacao (em %, proporção de casos que serão internados)'),
        min_value=1, max_value=100, step=1,
        value=12) / 100

    pdiw = st.sidebar.number_input(
        ('Tempo médio (duração) da internação em dias'),
        min_value=1, max_value=50, step=1,
        value=5)

    pdii = st.sidebar.number_input(
        ('Tempo médio (duração) da estadia em CTI em dias'),
        min_value=1, max_value=50, step=1,
        value=12)

    fator_severidade = st.sidebar.number_input(
        ('Proporção de casos internados que necessitam de CTI (em %)'),
        min_value=1, max_value=100, step=1,
        value=10) / 100

    fator_mortalidade = st.sidebar.number_input(
        ('Taxa de mortalidade específica (em %, proporção de casos que evoluem para óbito)'),
        min_value=0.1, max_value=1000.0, step=1.0,
        value=1.5) / 1000

    discount = fator_internacao * mk_share * (1/w_params['fator_subr']) * age_share

    real_new_cases = cases_df[w_place]['newCases'].rename('Casos oficiais (novos)')
    real_total_cases = cases_df[w_place]['totalCases'].rename('Casos oficiais (totais)')

    st.write('### População infectada')
    pop_infected = pd.Series(I.mean(axis=1), index=time_index, name='População infectada')
    st.line_chart(pop_infected, width=900, use_container_width=False)

    st.write('### Casos notificados')
    total_not_cases = pd.Series((I.mean(axis=1) + R.mean(axis=1)) * (1/w_params['fator_subr']),
                                index=time_index, name='Casos previstos (total)')
    total_not_cases_table = pd.merge(total_not_cases,
                    real_total_cases, how='outer', right_index=True, left_index=True)
    st.line_chart(total_not_cases_table, width=900, use_container_width=False)

    st.write('### Casos novos notificados')
    new_not_cases = total_not_cases.diff().fillna(method='bfill').rename('Casos previstos (novos)')
    new_not_cases_table = pd.merge(new_not_cases,
                            real_new_cases, how='outer', right_index=True, left_index=True)
    st.line_chart(new_not_cases_table, width=900, use_container_width=False)

    st.write('### Casos totais internação')
    total_cases = pd.Series((I.mean(axis=1) + R.mean(axis=1)) * discount,
                            index=time_index, name='Casos totais internação')
    st.line_chart(total_cases, width=900, use_container_width=False)

    st.write('### Casos novos internação')
    new_cases = total_cases.diff().fillna(method='bfill')
    st.line_chart(new_cases, width=900, use_container_width=False)

    st.title('Simulação de capacidade')
    simrun = st.button('Simular Capacidade')

    if simrun:
        my_bar = st.progress(1)
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

        st.write('### Ocupação na internação (número de leitos ocupados em cada dia)')
        ward_oc = pd.Series(logger['count_ward'], index=time_index, name='Ocupação na internação')
        st.line_chart(ward_oc, width=900, use_container_width=False)

        st.write('### Ocupação no CTI (número de vagas ocupadas em cada dia)')
        icu_oc = pd.Series(logger['count_icu'], index=time_index, name='Ocupação CTI')
        st.line_chart(icu_oc, width=900, use_container_width=False)
        df = (
            pd
            .concat(
                [ward_oc, icu_oc],
                axis=1
            )
            .assign(**
                {'Capacidade Internação': int(ward_capacity_times_availability_ward),
                 'Capacidade CTI': int(icu_capacity_times_availability_icu)}
            )
            .assign(**
                {'Saldo Internação': lambda x: x['Capacidade Internação'] - x['Ocupação na internação'],
                 'Saldo CTI': lambda x: x['Capacidade CTI'] - x['Ocupação CTI']}
            )
        )
        df = df[['Ocupação na internação', 'Capacidade Internação', 'Saldo Internação',
                 'Ocupação CTI', 'Saldo CTI', 'Capacidade CTI']]

        df = pd.merge(df, real_new_cases.astype(int), right_index=True, left_index=True, how='outer')
        df = pd.merge(df, new_not_cases.astype(int), right_index=True, left_index=True, how='outer')
        

        csv = (df.reset_index()
                 .rename(columns={'index': 'data'})
                 .to_csv(index=False, encoding='utf-8-sign')
                )

        b64 = base64.b64encode(csv.encode()).decode()
        download_time = pd.Timestamp.now().strftime("[%Y-%m-%d]-[%H-%M]")
        filename = f'simulação-{download_time}-covid-sus.csv'
        href = f'''
        <a download='{filename}'
        href="data:file/csv;base64,{b64}">
        Clique para baixar os resultados da simulação em formato CSV
        </a>
        '''
        st.markdown(href, unsafe_allow_html=True)

        mask = df['Saldo Internação'] <= 0
        if mask.mean() != 0:
            ward_doomsday = pd.to_datetime(df[mask].iloc[0].name)
            ward_alert = ward_doomsday - pd.DateOffset(7)
            st.write(f'''
            AVISO: A capacidade máxima de leitos clínicos será atingida em {ward_doomsday.date()}
            (uma semana antes: {ward_alert.date()})
            ''')
        mask = df['Saldo CTI'] <= 0
        if mask.mean() != 0:
            icu_doomsday = pd.to_datetime(df[mask].iloc[0].name)
            icu_alert = icu_doomsday - pd.DateOffset(7)
            st.write(f'''
            AVISO: A capacidade máxima de vagas em CTI será atingida em {icu_doomsday.date()}
            (uma semana antes: {icu_alert.date()})
            ''')
        st.dataframe(df)
