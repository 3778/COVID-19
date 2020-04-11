INTRODUCTION = '''
# Simulador COVID-19

_Ciência de Dados aplicada à pandemia do novo coronavírus_

---
'''

ABOUT = '''
Este projeto é uma força tarefa das comunidades científica e tecnológica a fim de criar modelos de previsão de infectados pelo COVID-19 - e outras métricas relacionadas -, para o Brasil. O projeto é público e pode ser usado por todos. 

Acesse [este link](https://github.com/3778/COVID-19) para informações detalhadas e instruções sobre como contribuir.
'''

PARAMETER_SELECTION='''
# Seleção de parâmetros
Para simular outros cenários, altere um parâmetro e tecle **Enter**. O novo resultado será calculado e apresentado automaticamente.
#### Parâmetros de UF/Município
'''

MODEL_INTRO='''
### Previsão de expostos e infectados
O gráfico abaixo mostra o resultado da simulação da evolução de pacientes expostos e infectados para os parâmetros selecionados. Mais informações sobre este modelo estão disponíveis [aqui](https://github.com/3778/COVID-19#seir-bayes).

**(!) Importante**: Os resultados apresentados são *preliminares* e estão em fase de validação.
'''

def make_SIMULATION_PARAMS(SEIR0, intervals, should_estimate_r0):
    alpha_inv_inf, alpha_inv_sup, _, _ = intervals[0]
    gamma_inv_inf, gamma_inv_sup, _, _ = intervals[1]

    if not should_estimate_r0:
        r0_inf, r0_sup, _, _ = intervals[2]
        r0_txt = f'- $${r0_inf:.03} < R_{{0}} < {r0_sup:.03}$$'
    else:
        r0_txt = '- $$R_{{0}}$$ está sendo estimado com dados históricos'

    intro_txt = '''
    ---

    ### Parâmetros da simulação
    
    Valores iniciais dos compartimentos:
    '''
    
    seir0_labels = [
        "Suscetíveis",
        "Expostos",
        "Infectados",
        "Removidos",
    ]
    seir0_values = list(map(int, SEIR0))
    seir0_dict = {
        "Compartimento": seir0_labels, 
        "Valor inicial": seir0_values,
    }
    
    other_params_txt = f'''
    Demais parâmetros:
    - $${alpha_inv_inf:.03} < T_{{incub}} = 1/\\alpha < {alpha_inv_sup:.03}$$
    - $${gamma_inv_inf:.03} < T_{{infec}} = 1/\gamma < {gamma_inv_sup:.03}$$
    {r0_txt}

    Os intervalos de $$T_{{incub}}$$ e $$T_{{infec}}$$ definem 95% do intervalo de confiança de uma distribuição LogNormal.
    ''' 
    return intro_txt, seir0_dict, other_params_txt

SIMULATION_CONFIG = '''
---

### Configurações da  simulação (menu à esquerda)

#### Seleção de UF/Município
É possível selecionar uma unidade da federação ou município para utilizar seus parâmetros nas condições inicias de *População total* (N), *Indivíduos infecciosos inicialmente* (I0), *Indivíduos removidos com imunidade inicialmente* (R0) e *Indivíduos expostos inicialmente (E0)*.

#### Limites inferiores e superiores dos parâmetros
Também podem ser ajustados limites superior e inferior dos parâmetros *Período infeccioso*, *Tempo de incubação* e *Número básico de reprodução*. Estes limites definem um intervalo de confiança de 95% de uma distribuição log-normal para cada parâmetro.\n\n\n
'''

DATA_SOURCES = '''
---

### Fontes dos dados

* Casos confirmados por município: [Número de casos confirmados de COVID-19 no Brasil](https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv) (de https://github.com/wcota/covid19br)
* Casos confirmados por estado: [Painel de casos de doença pelo coronavírus 2019 (COVID-19) no Brasil pelo Ministério da Saúde](https://covid.saude.gov.br/)
* População: Estimativa IBGE de 01/07/2019 (disponível em: [IBGE - Estimativas da população](https://www.ibge.gov.br/estatisticas/sociais/populacao/9103-estimativas-de-populacao.html))
'''

r0_ESTIMATION_TITLE = '### Número de reprodução básico $R_{{0}}$'

def r0_ESTIMATION(place, date): return  f'''
O número de reprodução básico $R_{0}$ está sendo estimado com os dados históricos de {place}. O valor utilizado no modelo SEIR-Bayes é o do dia {date}, que é o mais recente.

Caso você queria especificar o valor manualmente, desabilite a opção acima e insira os valores desejados.

**(!) Importante**: A estimação é sensível à qualidade das notificações dos casos positivos.
'''

r0_ESTIMATION_DONT = '''
Utilize o menu à esquerda para configurar o parâmetro.
'''

r0_CITATION = '''
A metodologia utilizada para estimação foi baseada no artigo [*Thompson, R. N., et al. "Improved inference of time-varying reproduction numbers during infectious disease outbreaks." Epidemics 29 (2019): 100356*](https://www.sciencedirect.com/science/article/pii/S1755436519300350). O código da implementação pode ser encontrado [aqui](https://github.com/3778/COVID-19/blob/master/covid19/estimation.py).
'''

def r0_NOT_ENOUGH_DATA(w_place, w_date): return f'''
**{w_place} não possui dados suficientes na data 
{w_date} para fazer a estimação. Logo, foram
utilizados os dados agregados Brasil**
'''
