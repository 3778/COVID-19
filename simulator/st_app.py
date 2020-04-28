import streamlit as st
from st_utils import texts
from covid19 import data

import st_app_r0
import st_app_seir
import st_app_queue

from datetime import datetime

def create_basic_sidebar(): 

    MIN_DATA_BRAZIL = '2020-03-26'
    DEFAULT_CITY = 'São Paulo/SP'
    DEFAULT_STATE = 'SP'
    MIN_CASES_TH = 10

    def format_local(key):
        return {
            'state': 'Estado',
            'city': 'Município'
        }.get(key, key)

    def format_date(date):
        return (datetime
                    .strptime(date, "%Y-%m-%d")
                    .strftime("%d/%m/%Y"))
    @st.cache(show_spinner=False)
    def make_place_options(cases_df, population_df):
        return (cases_df
                .swaplevel(0,1, axis=1)
                ['totalCases']
                .pipe(lambda df: df >= MIN_CASES_TH)
                .any()
                .pipe(lambda s: s[s & s.index.isin(population_df.index)])
                .index)

    @st.cache(show_spinner=False)
    def make_date_options(cases_df, place):
        return (cases_df
                [place]
                ['totalCases']
                .pipe(lambda s: s[s >= MIN_CASES_TH])
                [MIN_DATA_BRAZIL:]
                .index
                .strftime('%Y-%m-%d'))

    st.sidebar.markdown(texts.INTRODUCTION_SIDEBAR)

    w_r0_model = st.sidebar.checkbox(texts.R0_MODEL_DESC)
    w_seir_model = st.sidebar.checkbox(texts.SEIR_MODEL_DESC)
    w_queue_model = st.sidebar.checkbox(texts.QUEUE_MODEL_DESC)
    
    st.sidebar.markdown(texts.BASE_PARAMETERS_DESCRIPTION)

    st.sidebar.markdown("**Parâmetro de UF/Município**")
    
    w_location_granularity = st.sidebar.radio(
        "Unidade",
        options=("state", "city"),
        index=1,
        format_func=format_local)

    cases_df = data.load_cases(w_location_granularity)

    if w_location_granularity == 'city':

        population_df = data.load_population(w_location_granularity)
        options_place = make_place_options(cases_df, population_df)

        index = options_place.get_loc(DEFAULT_CITY)

        w_location = st.sidebar.selectbox(
            'Município',
            options=options_place,
            index=index)

    elif w_location_granularity == "state":

        population_df = data.load_population(w_location_granularity)
        options_place = make_place_options(cases_df, population_df)
        
        index = options_place.get_loc(DEFAULT_STATE)

        w_location = st.sidebar.selectbox(
            'Estado',
            options=options_place,
            index=index)

    options_date = make_date_options(cases_df, w_location)
    w_date = st.sidebar.selectbox('Data',
                                  options=options_date,
                                  index=len(options_date)-1,
                                  format_func=format_date)

    return {"location_granularity": w_location_granularity,
            "date": w_date,
            "location": w_location,
            "cases": cases_df,
            "population": population_df,
            "r0_model": w_r0_model,
            "seir_model": w_seir_model,
            "queue_model": w_queue_model}

if __name__ == '__main__':

    my_placeholder = st.empty()
    my_placeholder.markdown(texts.INTRODUCTION)

    base_parameters = create_basic_sidebar()

    if base_parameters['r0_model']:
        my_placeholder.markdown("")

        r0_samples, used_brasil = st_app_r0.build_r0(base_parameters['date'],
                                                     base_parameters["location"],
                                                     base_parameters["cases"])
    
    if base_parameters['seir_model']:
        my_placeholder.markdown("")

        if not base_parameters['r0_model']:
            r0_samples, _ = st_app_r0.estimate_r0(base_parameters['date'],
                                                  base_parameters["location"],
                                                  base_parameters["cases"])
    
        seir_output, reported_rate = st_app_seir.build_seir(base_parameters['date'],
                                                            base_parameters["location"],
                                                            base_parameters["cases"],
                                                            base_parameters["population"],
                                                            base_parameters["location_granularity"],
                                                            r0_samples)
    
    if base_parameters['queue_model']:
        my_placeholder.markdown("")

        if not base_parameters['seir_model']:
            
            r0_samples, _ = st_app_r0.estimate_r0(base_parameters['date'],
                                                  base_parameters["location"],
                                                  base_parameters["cases"])
            r0_dist = r0_samples[:, -1]
            seir_output, reported_rate, _ = st_app_seir.run_seir(base_parameters['date'],
                                                                 base_parameters["location"],
                                                                 base_parameters["cases"],
                                                                 base_parameters["population"],
                                                                 base_parameters["location_granularity"],
                                                                 r0_dist,
                                                                 w_params=st_app_seir.DEFAULT_PARAMS,
                                                                 sample_size=st_app_seir.SAMPLE_SIZE,
                                                                 reported_rate=None,
                                                                 NEIR0=None)

        st_app_queue.build_queue_simulator(base_parameters['date'],
                                           base_parameters["location"],
                                           base_parameters["cases"],
                                           base_parameters["location_granularity"],
                                           seir_output,
                                           reported_rate)
    st.markdown(texts.DATA_SOURCES)