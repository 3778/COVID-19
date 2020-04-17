import streamlit as st
import base64
import pandas as pd
import numpy as np
import math                     # add to requirements
from datetime import timedelta
from json import dumps


from st_utils import texts
from st_utils.viz import make_simulation_chart, prep_tidy_data_to_plot, make_combined_chart, plot_r0
from st_utils.formats import global_format_func
from hospital_queue.confirmation_button import cache_on_button_press
from hospital_queue.queue_simulation import run_queue_simulation
from covid19.utils import get_latest_file
from covid19 import data
from covid19.models import SEIRBayes
from covid19.estimation import ReproductionNumber


FATAL_RATE_BASELINE = 0.0138  # Verity R, Okell LC, Dorigatti I et al. Estimates of the severity of covid-19 disease. \
# medRxiv 2020.
SAMPLE_SIZE=500
MIN_CASES_TH = 10
MIN_DAYS_r0_ESTIMATE = 14
MIN_DEATH_SUBN = 3
MIN_DATA_BRAZIL = '2020-03-26'
DEFAULT_CITY = 'São Paulo/SP'
DEFAULT_STATE = 'SP'
DEFAULT_PARAMS = {
    'fator_subr': 10,
    'gamma_inv_dist': (7.0, 14.0, 0.95, 'lognorm'),
    'alpha_inv_dist': (4.1, 7.0, 0.95, 'lognorm'),
    'r0_dist': (2.5, 6.0, 0.95, 'lognorm'),

    # Simulations params
    'confirm_admin_rate': .07,  # considerando 2,8% a mortalidade do cdc para a pirâmide etária do Brasil
    'length_of_stay_covid': 9,
    'length_of_stay_covid_uti': 8,
    'icu_rate': .0,  # deve ser zero após implementarmos transferência dos mais graves do leito normal p/ a UTI \
    # quando os leitos normais lotarem antes
    'icu_rate_after_bed': .25,

    'icu_death_rate': .78,
    'icu_queue_death_rate': .0,
    'queue_death_rate': .0,

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
            [MIN_DATA_BRAZIL:]
            .index
            .strftime('%Y-%m-%d'))


def make_param_widgets(NEIR0, reported_rate, r0_samples=None, defaults=DEFAULT_PARAMS):
    _N0, _EIR0 = map(int, NEIR0)
    interval_density = 0.95
    family = 'lognorm'

    fator_subr = st.sidebar.number_input(
            'Taxa de reportagem de infectados. Porcentagem dos infectados que testaram positivo',
            min_value=0.0, max_value=100.0, step=1.0,
            value=reported_rate)

    st.sidebar.markdown('#### Condições iniciais')
    N = st.sidebar.number_input('População total (N)',
                                min_value=0, max_value=1_000_000_000, step=500_000,
                                value=_N0)

    EIR0 = st.sidebar.number_input('Indivíduos que já foram infectados e confirmados',
                                 min_value=0, max_value=1_000_000_000,
                                 value=_EIR0)

    
    st.sidebar.markdown('#### Período de infecção (1/γ) e tempo incubação (1/α)') 

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
            't_max': t_max,
            'NEIR0': (N, EIR0)}


def make_param_widgets_hospital_queue(location, w_granularity, defaults=DEFAULT_PARAMS):
     
    def load_beds(ibge_codes):
        # leitos
        beds_data = pd.read_csv(get_latest_file('ibge_leitos'))
        ibge_codes = pd.Series(ibge_codes).rename('codes_to_filter')
        beds_data_filtered = (beds_data[beds_data['codibge'].isin(ibge_codes)]
                              [['qtd_leitos', 'qtd_uti']]
                              .sum())

        return beds_data_filtered['qtd_leitos'], beds_data_filtered['qtd_uti']

    if w_granularity == 'state':
        uf = location
        qtd_beds, qtd_beds_uci = load_beds(data.get_ibge_codes_uf(uf))
    else:
        city, uf = location.split("/")
        qtd_beds, qtd_beds_uci = load_beds([data.get_ibge_code(city, uf)])

    # TODO: Adjust reliable cCFR
    # admiss_rate = FATAL_RATE_BASELINE/cCFR

    st.sidebar.markdown('---')
    st.sidebar.markdown('#### Parâmetros da simulação hospitalar')

    confirm_admin_rate = st.sidebar.number_input(
            'Porcentagem de confirmados que são hospitalizados (%)',
             step=1.0,
             min_value=0.0,
             max_value=100.0,
             value=DEFAULT_PARAMS['confirm_admin_rate']*100)

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

    icu_death_rate = st.sidebar.number_input(
             'Taxa de mortes após estadia na UTI',
             step=.01,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['icu_death_rate'])

    icu_queue_death_rate = st.sidebar.number_input(
             'Taxa de mortes na fila da UTI',
             step=.01,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['icu_queue_death_rate'])

    queue_death_rate = st.sidebar.number_input(
             'Taxa de mortes na fila dos leitos normais',
             step=.01,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['queue_death_rate'])

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
             value=int(qtd_beds))
        
    total_beds_icu = st.sidebar.number_input(
             'Quantidade de leitos de UTI',
             step=1,
             min_value=0,
             max_value=int(1e7),
             value=int(qtd_beds_uci))

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
    
    return {"confirm_admin_rate": confirm_admin_rate,
            "los_covid": los_covid,
            "los_covid_icu": los_covid_icu,
            "icu_rate": icu_rate,

            "icu_death_rate": icu_death_rate,
            "icu_queue_death_rate": icu_queue_death_rate,
            "queue_death_rate": queue_death_rate,
            
            "icu_after_bed": icu_after_bed,
            "total_beds": total_beds,
            "total_beds_icu": total_beds_icu,
            "available_rate": available_rate,
            "available_rate_icu": available_rate_icu
            }


@st.cache
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


def make_EI_df(model_output, sample_size, date):
    _, E, I, R, t = model_output
    size = sample_size*model.params['t_max']

    NI = np.add(pd.DataFrame(I).apply(lambda x: x - x.shift(1)).values,
                pd.DataFrame(R).apply(lambda x: x - x.shift(1)).values)

    df = (pd.DataFrame({'Exposed': E.reshape(size),
                        'Infected': I.reshape(size),
                        'Recovered': R.reshape(size),
                        'Newly Infected': NI.reshape(size),
                        'Run': np.arange(size) % sample_size}
                        ).assign(Day=lambda df: (df['Run'] == 0).cumsum() - 1))

    return df.assign(
        Date=df['Day'].apply(lambda x: pd.to_datetime(date) + timedelta(days=x)))


def plot_EI(model_output, scale):
    _, E, I, _, t = model_output
    source = prep_tidy_data_to_plot(E, I, t)
    return make_combined_chart(source, 
                               scale=scale, 
                               show_uncertainty=True)


@cache_on_button_press('Simular Modelo de Filas')
def run_queue_model(model_output, cases_df, w_place, w_date, params_simulation):

        bar_text = st.empty()
        bar_text.text('Estimando crecscimento de infectados...')
        simulations_outputs = []

        dataset, cut_after = calculate_input_hospital_queue(model_output , cases_df, w_place, w_date)
        
        for execution_columnm, execution_description in [('newly_infected_lower', 'Otimista'),
                                                         ('newly_infected_mean', 'Médio'),
                                                         ('newly_infected_upper', 'Pessimista')]:

            # TODO bug:review order of magnitude of all parameters (make sure it is consistant)

            dataset = dataset.assign(hospitalizados=0)

            for idx, row in dataset.iterrows():

                if idx < cut_after:
                    dataset['hospitalizados'].iloc[idx] = round(dataset[execution_columnm].iloc[idx] * params_simulation['confirm_admin_rate']/reported_rate)
                else:
                    dataset['hospitalizados'].iloc[idx] = round(dataset[execution_columnm].iloc[idx] * (params_simulation['confirm_admin_rate']/100))

            # dataset = dataset.assign(hospitalizados=round(dataset[execution_columnm]\
            # *params_simulation['confirm_admin_rate']*reported_rate/1000))
            # st.write("input modelo")
            # st.write(dataset.tail())

            bar_text = st.empty()
            bar = st.progress(0)

            bar_text.text(f'Processando o cenário {execution_description.lower()}...')
            simulation_output = (run_queue_simulation(dataset, bar, bar_text, params_simulation)
                .join(dataset, how='inner'))

            simulations_outputs.append((execution_columnm, execution_description, simulation_output))

            bar.progress(1.)
            bar_text.text(f"Processamento do cenário {execution_description.lower()} finalizado.")

        return simulations_outputs, cut_after


def calculate_input_hospital_queue(model_output, cases_df, place, date):

    S, E, I, R, t = model_output
    
    previous_cases = cases_df[place]

    # Formatting previous dates
    all_dates = pd.date_range(start=MIN_DATA_BRAZIL, end=date).strftime('%Y-%m-%d')
    all_dates_df = pd.DataFrame(index=all_dates,
                                data={"dummy": np.zeros(len(all_dates))})
    previous_cases = all_dates_df.join(previous_cases, how='left')['newCases']
    cut_after = previous_cases.shape[0]
    
    # Calculating newly infected for all samples
    size = sample_size*model.params['t_max']
    NI = np.add(pd.DataFrame(I).apply(lambda x: x - x.shift(1)).values,
                pd.DataFrame(R).apply(lambda x: x - x.shift(1)).values)
    pred = (pd.DataFrame({'Newly Infected': NI.reshape(size),
                          'Run': np.arange(size) % sample_size,
                          'Day': np.floor(np.arange(size) / sample_size) + 1}))
    pred = pred.assign(day=pred['Day'].apply(lambda x: pd.to_datetime(date) + timedelta(days=(x-1))))

    # Calculating standard deviation and mean
    def droplevel_col_index(df: pd.DataFrame):
        df.columns = df.columns.droplevel()
        return df

    df = (pred[['Newly Infected', 'day']]
          .groupby("day")
          .agg({"Newly Infected": [np.mean, np.std]})
          .pipe(droplevel_col_index)
          .assign(upper=lambda df: df["mean"] + df["std"])
          .assign(lower=lambda df: df["mean"] - df["std"])
          .add_prefix("newly_infected_")
          .join(previous_cases, how='outer')
          )

    # Formatting the final otput
    df = (df
          .assign(newly_infected_mean=df['newly_infected_mean'].combine_first(df['newCases']))
          .assign(newly_infected_upper=df['newly_infected_upper'].combine_first(df['newCases']))
          .assign(newly_infected_lower=df['newly_infected_lower'].combine_first(df['newCases']))
          .assign(newly_infected_lower=lambda df: df['newly_infected_lower'].clip(lower=0))
          .drop(columns=['newCases', 'newly_infected_std'])
          .reset_index()
          .rename(columns={'index':'day'}))

    return df, cut_after


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


def make_r0_widgets(defaults=DEFAULT_PARAMS):
    r0_inf = st.number_input(
             'Limite inferior do número básico de reprodução médio (R0)',
             min_value=0.01, max_value=10.0, step=0.25,
             value=defaults['r0_dist'][0])

    r0_sup = st.number_input(
            'Limite superior do número básico de reprodução médio (R0)',
            min_value=0.01, max_value=10.0, step=0.25,
            value=defaults['r0_dist'][1])
    return r0_inf, r0_sup, .95, 'lognorm'


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
        # cut_after = previous_cases.shape[0]

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

    reported_rate,cCFR = estimate_subnotification(cases_df, w_place, w_date, w_granularity)
    NEIR0 = make_NEIR0(cases_df, population_df, w_place, w_date,reported_rate)
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
                                              SAMPLE_SIZE, 
                                              MIN_DAYS_r0_ESTIMATE, 
                                              w_date)
        if used_brazil:
            st.write(texts.r0_NOT_ENOUGH_DATA(w_place, w_date))
                       
        _place = 'Brasil' if used_brazil else w_place
        st.markdown(texts.r0_ESTIMATION(_place, w_date))
                      
        st.altair_chart(plot_r0(r0_samples, w_date, 
                                _place, MIN_DAYS_r0_ESTIMATE))
        r0_dist = r0_samples[:, -1]
        st.markdown(f'**O $R_{{0}}$ estimado está entre '
                    f'${np.quantile(r0_dist, 0.01):.03}$ e ${np.quantile(r0_dist, 0.99):.03}$**')
        st.markdown(texts.r0_CITATION)
    else:
        r0_dist = make_r0_widgets()
        st.markdown(texts.r0_ESTIMATION_DONT)

    reported_rate = reported_rate*100
    w_params = make_param_widgets(NEIR0,reported_rate)
    model = SEIRBayes(**w_params, r0_dist=r0_dist)

    model_output = model.sample(sample_size)
    ei_df = make_EI_df(model_output, sample_size, w_date)
    st.markdown(texts.MODEL_INTRO)
    st.write(texts.SEIRBAYES_DESC)
    w_scale = st.selectbox('Escala do eixo Y',
                           ['log', 'linear'],
                           index=1)
    fig = plot_EI(model_output, w_scale)
    st.altair_chart(fig)

    download_placeholder = st.empty()

    if download_placeholder.button('Preparar dados para download em CSV'):
        href = make_download_href(ei_df, w_params, should_estimate_r0, r0_dist)
        st.markdown(href, unsafe_allow_html=True)
        download_placeholder.empty()

    dists = [w_params['alpha_inv_dist'],
             w_params['gamma_inv_dist'],
             r0_dist]
    SEIR0 = model._params['init_conditions']
    st.markdown(texts.make_SIMULATION_PARAMS(SEIR0, dists,
                                             should_estimate_r0))
    
    st.markdown(texts.HOSPITAL_QUEUE_SIMULATION)
    use_hospital_queue = st.checkbox('Habilitar simulador de fila hospitalar')
    # st.markdown(texts.SIMULATION_CONFIG)

    # Begining of the queue simulation
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

        params_simulation = make_param_widgets_hospital_queue(w_place, w_granularity)
        # st.write(cases_df.head())
        # st.write(w_place)
        # st.write(w_date)
        # st.write(params_simulation)
        simulation_outputs, cut_after = run_queue_model(model_output , cases_df, w_place, w_date, params_simulation)

        st.markdown(texts.HOSPITAL_GRAPH_DESCRIPTION)
        st.markdown(texts.HOSPITAL_BREAKDOWN_DESCRIPTION)

        def get_breakdown(description, simulation_output):

            simulation_output = simulation_output.assign(is_breakdown=simulation_output["Queue"] >= 1,
                                                         is_breakdown_icu=simulation_output["ICU_Queue"] >= 1)

            def get_breakdown_start(column):

                breakdown_days = simulation_output[simulation_output[column]]

                if (breakdown_days.size >= 1):
                    breakdown_date = breakdown_days.Data.iloc[0].strftime("%d/%m/%Y")
                    return breakdown_date
                else:
                    return "N/A"
            
            return (description,
                    get_breakdown_start('is_breakdown'), 
                    get_breakdown_start('is_breakdown_icu'))

        st.markdown((
            pd.DataFrame(
                data=[get_breakdown(description, simulation_output) for _, description, simulation_output in simulation_outputs],
                columns=['Cenário', 'Leitos comuns', 'Leitos UTI'])
            .set_index('Cenário')
            .to_markdown()))

        st.markdown("*N/A*: não houve formação de filas por falta de leitos")
        st.markdown("### Visualizações")

        plot_output = pd.concat(
            [(simulation_output
                .drop(simulation_output.index[:cut_after])
                .assign(description=description)) 
             for _, description, simulation_output in simulation_outputs])
            
        st.altair_chart(make_simulation_chart(plot_output, "Occupied_beds", "Ocupação de leitos comuns"))
        st.altair_chart(make_simulation_chart(plot_output, "ICU_Occupied_beds", "Ocupação de leitos de UTI"))
        st.altair_chart(make_simulation_chart(plot_output, "Queue", "Fila de pacientes"))
        st.altair_chart(make_simulation_chart(plot_output, "ICU_Queue", "Fila de pacientes UTI"))

        # TODO: change download method
        href = make_download_simulation_df(plot_output, 'queue-simulator.3778.care.csv')
        st.markdown(href, unsafe_allow_html=True)

    st.markdown(texts.DATA_SOURCES)