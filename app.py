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


st.markdown(
    """
    # COVID-19
    O objetivo deste projeto é iniciar uma força tarefa conjunta da comunidade científica e tecnológica a fim de criar modelos de previsão de infectados (e talvez outras métricas) pelo COVID-19, focando no Brasil. O projeto é público e pode ser usado por todos.
    
    Acesse [este link](https://github.com/3778/COVID-19) para mais informações.

    ## Previsão de infectados
    **Obs.**: Os resultados apresentados são *preliminares* e estão em fase de validação.
    """)

st.sidebar.title("Parâmetros da simulação")
st.sidebar.markdown("Para simular outros cenários, altere um parâmetro e tecle **Enter**. O novo resultado será calculado e apresentado automaticamente.")

N = int(st.sidebar.text_input('População total (N)', '13000000'))
E0 = int(st.sidebar.text_input('Indivíduos expostos inicialmente (E0)', '50'))
I0 = int(st.sidebar.text_input('Indivíduos infecciosos inicialmente (I0)', '152'))
R0 = int(st.sidebar.text_input('Indivíduos removidos com imunidade inicialmente (R0)', '1'))
R0__loc = float(st.sidebar.text_input('Número básico de reprodução médio (R0__loc)', '2.2'))
inverse_gamma_loc = float(st.sidebar.text_input('Tempo de recuperação em dias (1/gamma_loc)', '10'))
inverse_alpha_loc = float(st.sidebar.text_input('Tempo de incubação em dias (1/alpha_loc)', '5.2'))
t_max = int(st.sidebar.text_input('Período de simulação em dias (t_max)', '180'))
runs = int(st.sidebar.text_input('Qtde. de iterações da simulação (runs)', '100'))

gamma_loc = 1/inverse_gamma_loc
alpha_loc = 1/inverse_alpha_loc

S0 = N - (E0 + I0 + R0)
R0__scale = make_normal_scale(1.96, 2.55, .95, R0__loc)
gamma_scale = make_normal_scale(1/14, 1/7, .95, gamma_loc)
alpha_scale = make_normal_scale(1/7, 1/4.1, .95, alpha_loc)

fig = _run_SEIR_BAYES_model(N, E0, I0, R0,
                          R0__loc, R0__scale,
                          gamma_loc, gamma_scale,
                          alpha_loc, alpha_scale,
                          t_max, runs)

st.markdown(
    """
    ### Modelo SEIR-Bayes
    O gráfico abaixo mostra o resultado da simulação da evolução de pacientes infectados para os parâmetros escolhidos no menu da barra à esquerda. Mais informações sobre este modelo [aqui](https://github.com/3778/COVID-19#seir-bayes).
    """)
st.pyplot()