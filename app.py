import sys
sys.path.insert(0, "models")
import streamlit as st
from models.seir_bayes import (
    run_SEIR_BAYES_model, seir_bayes_plot, Model
    )
import matplotlib.pyplot as plt


def _run_SEIR_BAYES_model(model):
    model = run_SEIR_BAYES_model(model)
    fig = seir_bayes_plot(model)
    return fig


N = int(st.sidebar.text_input('N', '13000000'))
E0 = int(st.sidebar.text_input('E0', '50'))
I0 = int(st.sidebar.text_input('I0', '152'))
R0 = int(st.sidebar.text_input('R0', '1'))
R0__loc = float(st.sidebar.text_input('R0__loc', '2.2'))
gamma_loc = float(st.sidebar.text_input('gamma_loc', '0.1'))
alpha_loc = float(st.sidebar.text_input('alpha_loc', '0.1923'))
t_max = int(st.sidebar.text_input('t_max', '180'))
runs = int(st.sidebar.text_input('runs', '100'))

model = Model(
    {
        'N': N,
        'E0': E0,
        'I0': I0,
        'R0': R0,
        'R0__loc': R0__loc,
        'gamma_loc': gamma_loc,
        'alpha_loc': alpha_loc,
        't_max': t_max,
        'runs': runs
    }
)

fig = _run_SEIR_BAYES_model(model)
st.pyplot()
