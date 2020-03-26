import streamlit as st
import texts
from covid19 import data
from formats import global_format_func

if __name__ == '__main__':
    st.markdown(texts.INTRODUCTION)
    st.sidebar.markdown(texts.PARAMETER_SELECTION)
    GRANULARITY = st.sidebar.selectbox('Unidade',
                                       options=['state', 'city'],
                                       index=1,
                                       format_func=global_format_func)
    cases_df = data.load_cases(GRANULARITY)
    population_df = data.load_population(GRANULARITY)
    st.table(cases_df.head())
