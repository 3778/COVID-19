import streamlit as st
from models.seir_bayes import run_SEIR_BAYES_model, make_normal_scale, seir_bayes_plot
import matplotlib.pyplot as plt



def _run_SEIR_BAYES_model(N, E0, I0, R0, 
                          R0__loc, R0__scale,
                          gamma_loc, gamma_scale,
                          alpha_loc, alpha_scale,
                          t_max, runs):
    S, E, I, R, t_space = run_SEIR_BAYES_model(
                                        N, E0, I0, R0, 
                                        R0__loc, R0__scale,
                                        gamma_loc, gamma_scale,
                                        alpha_loc, alpha_scale,
                                        t_max, runs)
    fig = seir_bayes_plot(N, E0, I0, R0, 
                          R0__loc, R0__scale,
                          gamma_loc, gamma_scale,
                          alpha_loc, alpha_scale,
                          t_max, runs, S, E, I, R, t_space)
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

S0 = N - (E0 + I0 + R0)
R0__scale = make_normal_scale(1.96, 2.55, .95, R0__loc)
gamma_scale = make_normal_scale(1/14, 1/7, .95, gamma_loc)
alpha_scale = make_normal_scale(1/7, 1/4.1, .95, alpha_loc)

fig = _run_SEIR_BAYES_model(N, E0, I0, R0,
                          R0__loc, R0__scale,
                          gamma_loc, gamma_scale,
                          alpha_loc, alpha_scale,
                          t_max, runs)
st.pyplot()