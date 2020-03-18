
COVID-19
====
O objetivo deste repositório é iniciar uma força tarefa conjunta da comunidade científica e tecnológica a fim de criar modelos de previsão de infectados (e talvez outras métricas) pelo COVID-19. O projeto é público e pode ser usado por todxs.

Toda e qualquer comunicação deve ser feita publicamente via [GitHub Issues](https://github.com/3778/COVID-19/issues) (fique a vontade para criar uma issue nova).

No momento, as principais contribuições são o modelo [SEIR-Bayes](#seir-bayes) que pode ser visualizado interativamente com o [Simulador](#simulador)

# Índice
<!--ts-->
   * [COVID-19](#covid-19)
   * [Índice](#índice)
   * [Informações rápidas](#informações-rápidas)
      * [Qual o modelo que acreditamos ser melhor?](#qual-o-modelo-que-acreditamos-ser-melhor)
   * [Setup para rodar os modelos](#setup-para-rodar-os-modelos)
   * [Modelos](#modelos)
      * [Modelos Compartimentados](#modelos-compartimentados)
         * [SEIR-ODE](#seir-ode)
         * [SEIR-SDE](#seir-sde)
         * [SEIR-Bayes](#seir-bayes)
            * [Resultado](#resultado)
   * [Simulador](#simulador)
      * [Com pip](#com-pip)
      * [Com Docker](#com-docker)
   * [Como contribuir?](#como-contribuir)
      * [Tipos de contribuições](#tipos-de-contribuições)
   * [Recursos didáticos](#recursos-didáticos)
      * [Introdução aos modelos SEIR e variantes](#introdução-aos-modelos-seir-e-variantes)
      * [Implementações](#implementações)
      * [Efeito das intervenções públicas](#efeito-das-intervenções-públicas)
   * [Referências](#referências)

<!-- Added by: severo, at: Tue Mar 17 23:50:23 -03 2020 -->

<!--te-->

# Informações rápidas
## Qual o modelo que acreditamos ser melhor?
[SEIR-Bayes](#seir-bayes)

# Setup para rodar os modelos
1. Instale python 3.6 ou superior;
2. (Opcional) Crie um ambiente virtual;
3. Instale as dependências com `pip install -r requirements.txt`

# Modelos
Estes modelos são testes iniciais e não são bons exemplos de como se deve programar em Python.

## Modelos Compartimentados
https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model

Buscamos na [literatura](#referências) e temos as seguintes estimativas para os parâmetros desses modelos.

|             Parâmetro            | Limite inferior | Valor típico | Limite superior | Referências |
|:--------------------------------:|:---------------:|--------------|-----------------|:-----------:|
|     Tempo de incubação (1/α)     |       4.1       | 5.2 dias     | 7.0             |   1, 2, 4   |
| Número básico de reprodução (R0) |       1.4       | 2.2          | 3.9             |   2, 3, 4   |
|  Período infeccioso médio (1/γ)  |        ?        | 14 dias      | ?               |      1      |

### SEIR-ODE
Este modelo deterministico separa a população em 4 compartimentos: Suscetíveis, Expostos, Infectados e Removidos; cujo equacionamento é dado por uma equação differencial ordinária.

Para rodar: `python models/seir_ode.py` (a forma de rodar provavelmente vai mudar no futuro)

[[Codigo]](/models/seir_ode.py) [[Equacionamento]](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model)

### SEIR-SDE
Modelo similar ao [SEIR-ODE](#seir-ode), porem com dinâmica de transição de estados estabelecida por uma binomial.

Para rodar: `python models/seir_sde.py` (a forma de rodar provavelmente vai mudar no futuro)

[[Codigo]](/models/seir_sde.py)

### SEIR-Bayes
Modelo similar ao [SEIR-SDE](#seir-sde), porém com os parâmetros alpha, gamma e beta amostrados de uma distribuição à priori para cada rodada de simulação.

Para rodar: `python models/seir_bayes.py` (a forma de rodar provavelmente vai mudar no futuro)

[[Codigo]](/models/seir_bayes.py)

#### Resultado
Este resultado é preliminar, favor ver [issue 13](https://github.com/3778/COVID-19/issues/13). O objetivo era simular a cidade de São Paulo.
![](/figures/seir-bayes-0.png)

# Simulador 

Este simulador usa o [Streamlit](https://www.streamlit.io/). No momento, ele permite simular o [SEIR-Bayes](#seir-bayes) variando os parâmetros. Estamos trabalhando para melhorar este simulador (veja as issues).

## Com pip
1. Faça o [Setup para rodar os modelos](#setup-para-rodar-os-modelos)
2. Execute `make launch`

## Com Docker
1. Instale [docker](https://docs.docker.com/install/);
2. Na raiz do projeto execute `make image` para construir a imagem;
3. Em seguida, execute `make covid-19` e aponte seu navegador para [http://localhost:8501](http://localhost:8501).

# Como contribuir?
Fique a vontade para abrir uma issue nova, ou trabalhar em uma já existente. Discussões e sugestões, além de código e modelagem, são bem vindas.

## Quero entender os modelos, mas não sei por onde começar!
- [The MATH of Epidemics | Intro to the SIR Model](https://youtu.be/Qrp40ck3WpI)

## Tipos de contribuições
Toda contribuição é bem vinda. Estamos gerenciando via GitHub Issues. Existem algumas categorias de contribuições:
- [modelagem](https://github.com/3778/COVID-19/issues?q=is%3Aopen+is%3Aissue+label%3Amodelagem) - relacionados a modelagem matemática (discussões e implementações) dos modelos;
- [bug](https://github.com/3778/COVID-19/issues?q=is%3Aopen+is%3Aissue+label%3Abug) - problemas encontrados no código;
- [documentação](https://github.com/3778/COVID-19/issues?q=is%3Aopen+is%3Aissue+label%3Adocumenta%C3%A7%C3%A3o);
- [dev](https://github.com/3778/COVID-19/issues?q=is%3Aopen+is%3Aissue+label%3Adev) - tudo que é relacionado a código (sem ser a modelagem ou bugs);
- `modelo: $NOME_DO_MODELO`- para modelos específicos (por exemplo, [modelo: SEIR-Bayes](https://github.com/3778/COVID-19/issues?q=is%3Aissue+is%3Aopen+label%3A%22modelo%3A+SEIR-Bayes%22)).

# Recursos didáticos
## Introdução aos modelos SEIR e variantes
- [Compartmental models in epidemiology](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
- [The MATH of Epidemics | Intro to the SIR Model](https://youtu.be/Qrp40ck3WpI)

## Implementações
- [Modelling the coronavirus epidemic spreading in a city with Python](https://towardsdatascience.com/modelling-the-coronavirus-epidemic-spreading-in-a-city-with-python-babd14d82fa2)
- [Social Distancing to Slow the Coronavirus](https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296)
- [The SIR epidemic model (SciPy)](https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/)

## Efeito das intervenções públicas
- [Understanding Unreported Cases in the COVID-19 Epidemic Outbreak in Wuhan, China, and the Importance of Major Public Health Interventions](https://www.mdpi.com/2079-7737/9/3/50/htm)

# Referências
1. [Report of the WHO-China Joint Mission on Coronavirus Disease 2019 (COVID-19)](https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf)
2. [Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus–Infected Pneumonia](nejm.org/doi/full/10.1056/NEJMoa2001316)
3. [Estimation of the reproductive number of novel coronavirus (COVID-19) and the probable outbreak size on the Diamond Princess cruise ship: A data-driven analysis](https://www.ijidonline.com/article/S1201-9712(20)30091-6/fulltext)
4. [MIDAS Online Portal for COVID-19 Modeling Research](https://midasnetwork.us/covid-19/#resources)
