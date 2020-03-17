
COVID-19
====
Ciência de Dados aplicada a pandemia COVID-19.


# Índice
<!--ts-->
   * [COVID-19](#covid-19)
   * [Índice](#índice)
   * [Setup para rodar os modelos](#setup-para-rodar-os-modelos)
   * [Modelos](#modelos)
      * [SEIR (ODE)](#seir-ode)
   * [Recursos didáticos](#recursos-didáticos)

<!-- Added by: severo, at: Mon Mar 16 21:11:08 -03 2020 -->

<!--te-->

# Setup para rodar os modelos
1. Instale python 3.6 ou superior;
2. (Opcional) Crie um ambiente virtual;
3. Instale as dependências com `pip install -r requirements.txt`

# Modelos
## SEIR (ODE)
Este modelo deterministico separa a população em 3 compartimentos: Suscetíveis, Expostos, Infectados e Removidos; cujo equacionamento é dado por uma equação differencial ordinária.

Para rodar: `python models/SEIR-ode.py` (a forma de rodar provavelmente vai mudar no futuro)

[[Codigo]](/models/SEIR-ode.py) [[Equacionamento]](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model)

# Recursos didáticos
- [Compartmental models in epidemiology](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
- [Modelling the coronavirus epidemic spreading in a city with Python](https://towardsdatascience.com/modelling-the-coronavirus-epidemic-spreading-in-a-city-with-python-babd14d82fa2)
- [Social Distancing to Slow the Coronavirus](https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296)
- [The MATH of Epidemics | Intro to the SIR Model](https://youtu.be/Qrp40ck3WpI)
- [The SIR epidemic model (SciPy)](https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/)
- [Understanding Unreported Cases in the COVID-19 Epidemic Outbreak in Wuhan, China, and the Importance of Major Public Health Interventions](https://www.mdpi.com/2079-7737/9/3/50/htm)
