INTRODUCTION = '''
# COVID-19

## Simulador de Capacidade instalada para o SUS

<TEXTO INTRODUTÓRIO>

Acesse [este link](https://github.com/3778/COVID-19) para mais informações.

---
'''

PARAMETER_SELECTION='''
# Seleção de parâmetros (avançado)
'''

MODEL_INTRO='''
## Modelo SEIR-Bayes
O gráfico abaixo mostra o resultado da simulação da evolução de pacientes infectados para os parâmetros escolhidos no menu da barra à esquerda. Mais informações sobre este modelo [aqui](https://github.com/3778/COVID-19#seir-bayes).

### Previsão de infectados
**(!) Importante**: Os resultados apresentados são *preliminares* e estão em fase de validação.
'''

def make_SIMULATION_PARAMS(SEIR0, intervals, should_estimate_r0):
    alpha_inv_inf, alpha_inv_sup, _, _ = intervals[0]
    gamma_inv_inf, gamma_inv_sup, _, _ = intervals[1]

    if not should_estimate_r0:
        r0_inf, r0_sup, _, _ = intervals[2]
        r0_txt = f'- $${r0_inf:.03} < R_{{0}} < {r0_sup:.03}$$'
    else:
        r0_txt = '- $$R_{{0}}$$ está sendo estimado com dados históricos.'

    S0, E0, I0, R0 = map(int, SEIR0)
    txt = f'''
    ### Parâmetros da simulação
    - $SEIR(0) = ({S0}, {E0}, {I0}, {R0})$

    Os intervalos abaixo definem 95% do intervalo de confiança de uma distribuição LogNormal
    - $${alpha_inv_inf:.03} < T_{{incub}} = 1/\\alpha < {alpha_inv_sup:.03}$$
    - $${gamma_inv_inf:.03} < T_{{infec}} = 1/\gamma < {gamma_inv_sup:.03}$$
    ''' 
    return txt + r0_txt

SIMULATION_CONFIG = '''
### Configurações da  simulação (menu à esquerda)

### Seleção de Unidade
 possível selecionar o tipo de unidade (Estado ou Município).
#### Seleção de UF/Município
Baseado na seleção anterior, é possível selecionar uma unidade da federação ou município para utilizar seus parâmetros nas condições inicias de *População total* (N), *Indivíduos infecciosos inicialmente* (I0), *Indivíduos removidos com imunidade inicialmente* (R0) e *Indivíduos expostos inicialmente (E0)*.

#### Limites inferiores e superiores dos parâmetros
Também podem ser ajustados limites superior e inferior dos parâmetros *Período infeccioso*, *Tempo de incubação* e *Número básico de reprodução*. Estes limites definem um intervalo de confiança de 95% de uma distribuição log-normal para cada parâmetro.\n\n\n
'''

DATA_SOURCES = '''
### Fontes dos dados

* Casos confirmados: [Número de casos confirmados de COVID-19 no Brasil](https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv) (de https://github.com/wcota/covid19br)
* População: Estimativa IBGE de 01/07/2019 (disponível em: [IBGE - Estimativas da população](https://www.ibge.gov.br/estatisticas/sociais/populacao/9103-estimativas-de-populacao.html))
'''

r0_ESTIMATION_TITLE = '## Número de reprodução básico $R_{{0}}$'

def r0_ESTIMATION(place, date): return  f'''
O valor do número de reprodução básico $R_{0}$ está sendo estimado com os dados históricos de {place}. Caso você queria especificar o valor manualmente, desabilite a opção acima e insira os valores desejados no menu à esquerda.

O $R_{{0}}$ utilizado no modelo SEIR-Bayes é o do dia {date}, que é o mais recente.
'''

SEIRBAYES_DESC = '''
O eixo do tempo do modelo abaixo considera que $0$ é o dia em que foram observadas as condições iniciais inseridas no menu à esquerda. É possível selecionar o estado ou município, além da data, desejado. Neste caso, o dia $0$ será a data escolhida.
'''

r0_ESTIMATION_DONT = '''
Utilize o menu à esquerda para configurar o parâmetro.
'''

r0_CITATION = '''
A metodologia utilizada para estimação foi baseada no artigo [*Thompson, R. N., et al. "Improved inference of time-varying reproduction numbers during infectious disease outbreaks." Epidemics 29 (2019): 100356*](https://www.sciencedirect.com/science/article/pii/S1755436519300350). O código da implementação pode ser encontrado [aqui](https://github.com/3778/COVID-19/blob/master/covid19/estimation.py).

'''


def r0_NOT_ENOUGH_DATA(w_place, w_date): return f'''
**{w_place} não possui dados suficientes na data 
{w_date} para fazer a estimação do R0 (número básico de reprodução). 
Logo, foram utilizados os dados agregados Brasil**
'''
