import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import base64
from datetime import timedelta, datetime

from covid19 import data
from covid19.utils import get_data_dir
from st_utils.viz import make_simulation_chart, make_simulation_chart_ocup_rate
from st_utils import texts  
from hospital_queue.queue_simulation import run_queue_simulation

MIN_DATA_BRAZIL = '2020-02-25'

DEFAULT_PARAMS = {'confirm_admin_rate': .07,  
    'length_of_stay_covid': 9,
    'length_of_stay_covid_uti': 8,
    'icu_rate': .0,  
    'icu_rate_after_bed': .25,

    'icu_death_rate': .78,
    'icu_queue_death_rate': .0,
    'queue_death_rate': .0,

    'total_beds': 12222,
    'total_beds_icu': 2421,
    'available_rate': .36,
    'available_rate_icu': .36}

def get_latest_file(file_name, file_type='csv'):
    """
    Gets most recent file. Useful for files with datetime in their names

    :param file_name: file name substring to get the most recent
    :param file_type: file type for filtering the file list
    :return: full file path to be directly used on functions
    """
    list_of_files = glob.glob(os.path.join(get_data_dir(), '%s*.%s' % (file_name, file_type)))
    latest_file = max(list_of_files, key=os.path.basename)

    return latest_file

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

def make_param_widgets_hospital_queue(location, w_granularity, defaults=DEFAULT_PARAMS):
     
    def load_beds(ibge_codes):
        beds_data = pd.read_csv(get_latest_file('ibge_leitos'))
        ibge_codes = pd.Series(ibge_codes).rename('codes_to_filter')
        beds_data_filtered = (beds_data[beds_data['codibge'].isin(ibge_codes)]
                              [['qtd_leitos', 'qtd_uti']]
                              .sum())

        return beds_data_filtered['qtd_leitos'], beds_data_filtered['qtd_uti']

    confirm_admin_rate = DEFAULT_PARAMS['confirm_admin_rate']

    if w_granularity == 'state':
        uf = location
        qtd_beds, qtd_beds_uci = load_beds(data.get_ibge_codes_uf(uf))
        confirm_admin_rate = data.state_hospitalization(uf)
    else:
        city, uf = location.split("/")
        qtd_beds, qtd_beds_uci = load_beds([data.get_ibge_code(city, uf)])
        confirm_admin_rate = data.city_hospitalization(city,uf)

    st.sidebar.markdown('---')
    st.sidebar.markdown("**Parâmetro de simulação hospitalar**")

    if st.sidebar.checkbox('Parâmetros básicos'):
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
    else:
        total_beds = int(qtd_beds)
        total_beds_icu = int(qtd_beds_uci)
        available_rate = DEFAULT_PARAMS['available_rate']
        available_rate_icu = DEFAULT_PARAMS['available_rate_icu']

    if st.sidebar.checkbox('Parâmetros avançados', key='checkbox_parameters_queue'):
        confirm_admin_rate = st.sidebar.number_input(
                'Porcentagem de confirmados que são hospitalizados (%)',
                step=1.0,
                min_value=0.0,
                max_value=100.0,
                value=100*confirm_admin_rate)

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
        
        
    else:
        confirm_admin_rate = confirm_admin_rate*100
        los_covid = DEFAULT_PARAMS['length_of_stay_covid']
        los_covid_icu = DEFAULT_PARAMS['length_of_stay_covid_uti']
        icu_rate = DEFAULT_PARAMS['icu_rate']
        icu_death_rate = DEFAULT_PARAMS['icu_death_rate']
        icu_queue_death_rate = DEFAULT_PARAMS['icu_queue_death_rate']
        queue_death_rate = DEFAULT_PARAMS['queue_death_rate']
        icu_after_bed = DEFAULT_PARAMS['icu_rate_after_bed']

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
            "available_rate_icu": available_rate_icu}

@st.cache(show_spinner=False)
def calculate_input_hospital_queue(model_output, sample_size, t_max, cases_df,real_cases, place, date):
    S, E, I, R, t = model_output
    real_cases_df = real_cases[0]
    for i in range(real_cases_df['date'].shape[0]):
        real_cases_df['date'].iloc[i] = datetime.strptime(real_cases_df['date'].iloc[i], '%Y-%m-%d')
    real_cases_df = real_cases_df.set_index('date')
    previous_cases = real_cases_df['real_newCases']
    # Formatting previous dates
    all_dates = pd.date_range(start=MIN_DATA_BRAZIL, end=date).strftime('%Y-%m-%d')
    all_dates_df = pd.DataFrame(index=all_dates,
                                data={"dummy": np.zeros(len(all_dates))})
    previous_cases = all_dates_df.join(previous_cases, how='left')['real_newCases']
    cut_after = previous_cases.shape[0]
    
    # Calculating newly infected for all samples
    size = sample_size*t_max
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
          .assign(newly_infected_mean=df['newly_infected_mean'].combine_first(df['real_newCases']))
          .assign(newly_infected_upper=df['newly_infected_upper'].combine_first(df['real_newCases']))
          .assign(newly_infected_lower=df['newly_infected_lower'].combine_first(df['real_newCases']))
          .assign(newly_infected_lower=lambda df: df['newly_infected_lower'].clip(lower=0))
          .drop(columns=['real_newCases', 'newly_infected_std'])
          .reset_index()
          .rename(columns={'index':'day'}))
    df = df.fillna(method='ffill')
    df = df.fillna(0)
    return df, cut_after


def run_queue_simulation_cached(dataset, cut_after, params_simulation, reported_rate, w_place):

    @st.cache(show_spinner=False, suppress_st_warning=True)
    def run_simulation(dataset, execution_columnm, execution_description):
        df = dataset.copy()
        df = df.assign(hospitalizados=0)

        for idx, row in df.iterrows():

            df['hospitalizados'].iloc[idx] = round(df[execution_columnm].iloc[idx] * (params_simulation['confirm_admin_rate']/100))
            
                

        bar_text = st.empty()
        bar = st.progress(0)

        bar_text.text(f'Processando o cenário {execution_description.lower()}...')
        simulation_output = (run_queue_simulation(df, bar, bar_text, params_simulation)
            .join(df, how='inner'))

        bar.progress(1.)
        bar_text.text(f"Processamento do cenário {execution_description.lower()} finalizado.")

        return simulation_output

    return [('newly_infected_lower', 'Otimista', run_simulation(dataset, 'newly_infected_lower', 'Otimista')),
            ('newly_infected_mean', 'Médio', run_simulation(dataset,'newly_infected_mean', 'Médio')),
            ('newly_infected_upper', 'Pessimista', run_simulation(dataset,'newly_infected_upper', 'Pessimista'))]


def run_queue_model(model_output, 
                    sample_size,
                    t_max,
                    reported_rate,
                    cases_df,
                    real_cases,
                    w_place,
                    w_date,
                    params_simulation):


        
        with st.spinner("Estimando crecscimento de infectados...."):
            dataset, cut_after = calculate_input_hospital_queue(model_output,
                                                                sample_size,
                                                                t_max,
                                                                cases_df,
                                                                real_cases,
                                                                w_place,
                                                                w_date)

        simulations_outputs = run_queue_simulation_cached(dataset, cut_after, params_simulation, reported_rate, w_place)
        return simulations_outputs, cut_after

def build_queue_simulator(w_date,
                          w_location,
                          cases_df,
                          w_location_granulariy,
                          real_cases,
                          seir_output,
                          reported_rate):
    
    st.markdown(texts.HOSPITAL_QUEUE_SIMULATION)
    st.markdown(texts.QUEUE_SIMULATION_SOURCE_EXPLAIN)
    st.markdown("---")

    _, model_output, sample_size, t_max = seir_output
    params_simulation = make_param_widgets_hospital_queue(w_location, w_location_granulariy)

    if st.button("Executar simulação"):
        simulation_outputs, cut_after = run_queue_model(model_output,
                                                        sample_size,
                                                        t_max,
                                                        reported_rate,
                                                        cases_df,
                                                        real_cases,
                                                        w_location,
                                                        w_date,
                                                        params_simulation)
        
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
            
        st.altair_chart(make_simulation_chart(plot_output, "Occupied_beds", "Leitos Comuns Ocupados COVID"))
        st.altair_chart(make_simulation_chart(plot_output, "ICU_Occupied_beds", "Leitos UTI Ocupados COVID"))

        st.altair_chart(make_simulation_chart_ocup_rate(plot_output, "Occupied_beds", "Taxa de ocupação de leitos comuns (%)",
                                                        params_simulation["total_beds"],params_simulation["total_beds_icu"],
                                                        params_simulation["available_rate"],params_simulation["available_rate_icu"]))
        st.altair_chart(make_simulation_chart_ocup_rate(plot_output, "ICU_Occupied_beds", "Taxa de ocupação de leitos UTI (%)",
                                                        params_simulation["total_beds"],params_simulation["total_beds_icu"],
                                                        params_simulation["available_rate"],params_simulation["available_rate_icu"]))

        st.altair_chart(make_simulation_chart(plot_output, "Queue", "Fila de pacientes"))
        st.altair_chart(make_simulation_chart(plot_output, "ICU_Queue", "Fila de pacientes UTI"))

        href = make_download_simulation_df(plot_output, 'queue-simulator.3778.care.csv')
        st.markdown(href, unsafe_allow_html=True)
