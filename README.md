
COVID-19
====
Ciência de Dados aplicada a pandemia COVID-19.


# Índice
<!--ts-->
   * [COVID-19](#covid-19)
   * [Índice](#índice)
   * [Informações rápidas](#informações-rápidas)
      * [Qual o modelo que acreditamos ser melhor?](#qual-o-modelo-que-acreditamos-ser-melhor)
   * [Setup para rodar os modelos](#setup-para-rodar-os-modelos)
   * [Modelos](#modelos)
      * [SEIR-ODE](#seir-ode)
      * [SEIR-SDE](#seir-sde)
      * [SEIR-Bayes](#seir-bayes)
         * [Resultado](#resultado)
   * [Como contribuir?](#como-contribuir)
      * [Tipos de contribuições](#tipos-de-contribuições)
   * [Recursos didáticos](#recursos-didáticos)
      * [Introdução aos modelos SEIR e variantes](#introdução-aos-modelos-seir-e-variantes)
      * [Implementações](#implementações)
      * [Efeito das intervenções públicas](#efeito-das-intervenções-públicas)

<!-- Added by: severo, at: Tue Mar 17 20:49:30 -03 2020 -->

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

## SEIR-ODE
Este modelo deterministico separa a população em 4 compartimentos: Suscetíveis, Expostos, Infectados e Removidos; cujo equacionamento é dado por uma equação differencial ordinária.

Para rodar: `python models/SEIR-ode.py` (a forma de rodar provavelmente vai mudar no futuro)

[[Codigo]](/models/SEIR-ode.py) [[Equacionamento]](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model)

## SEIR-SDE
Modelo similar ao [SEIR-ODE](#seir-ode), porem com dinâmica de transição de estados estabelecida por uma binomial.

Para rodar: `python models/SEIR-sde.py` (a forma de rodar provavelmente vai mudar no futuro)

[[Codigo]](/models/SEIR-sde.py) [[Equacionamento]](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model)

## SEIR-Bayes
Modelo similar ao [SEIR-SDE](#seir-sde), porem com os parâmetros alpha, gamma e beta são amostrados de uma distribuição à priori para cada rodada de simulação
[[Codigo]](/models/SEIR-Bayes.py) [[Equacionamento]](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model)

### Como levamos em conta a varianção dos parâmetros?

### Resultado
**Este resultado é preliminar, favor ver** a [issue 13](https://github.com/3778/COVID-19/issues/13)
![](/figures/seir-bayes-0.png)

# Como contribuir?
## Tipos de contribuições
Toda contribuição é bem vinda. Estamos gerenciando via GitHub Issues. Existem algumas categorias de contribuições:
- [modelagem](https://github.com/3778/COVID-19/issues?q=is%3Aopen+is%3Aissue+label%3Amodelagem) - relacionados a modelagem matemática (discussões e implementações) dos modelos;
- [bug](https://github.com/3778/COVID-19/issues?q=is%3Aopen+is%3Aissue+label%3Abug) - problemas encontrados no código;
- [documentação](https://github.com/3778/COVID-19/issues?q=is%3Aopen+is%3Aissue+label%3Adocumenta%C3%A7%C3%A3o);
- [dev](https://github.com/3778/COVID-19/issues?q=is%3Aopen+is%3Aissue+label%3Adev) - tudo que é relacionado a código (sem ser a modelagem ou bugs);
- modelo: <nome do modelo> - para modelos específicos (por exemplo, [modelo: SEIR-Bayes](https://github.com/3778/COVID-19/issues?q=is%3Aissue+is%3Aopen+label%3A%22modelo%3A+SEIR-Bayes%22)).

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
