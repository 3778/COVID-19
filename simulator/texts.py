INTRODUCTION = '''
# COVID-19
O objetivo deste projeto é iniciar uma força tarefa conjunta da comunidade científica e tecnológica a fim de criar modelos de previsão de infectados (e talvez outras métricas) pelo COVID-19, focando no Brasil. O projeto é público e pode ser usado por todos.

Acesse [este link](https://github.com/3778/COVID-19) para mais informações.

---

## Previsão de infectados
**(!) Importante**: Os resultados apresentados são *preliminares* e estão em fase de validação.
'''

PARAMETER_SELECTION='''
# Seleção de parâmetros
Para simular outros cenários, altere um parâmetro e tecle **Enter**. O novo resultado será calculado e apresentado automaticamente.
#### Parâmetros de UF/Município
'''

MODEL_INTRO='''
### Modelo SEIR-Bayes
O gráfico abaixo mostra o resultado da simulação da evolução de pacientes infectados para os parâmetros escolhidos no menu da barra à esquerda. Mais informações sobre este modelo [aqui](https://github.com/3778/COVID-19#seir-bayes).
'''

def make_SIMULATION_PARAMS(SEIR0, intervals):
    alpha_inv_inf, alpha_inv_sup, _ = intervals[0]
    gamma_inv_inf, gamma_inv_sup, _ = intervals[1]
    r0_inf, r0_sup, _ = intervals[2]
    S0, E0, I0, R0 = map(int, SEIR0)
    return f'''
    ### Parâmetros da simulação
    - $SEIR(0) = ({S0}, {E0}, {I0}, {R0})$

    Os intervalos abaixo definem 95% do intervalo de confiança de uma distribuição LogNormal
    - $${alpha_inv_inf:.03} < T_{{incub}} = 1/\\alpha < {alpha_inv_sup:.03}$$
    - $${gamma_inv_inf:.03} < T_{{infec}} = 1/\gamma < {gamma_inv_sup:.03}$$
    - $${r0_inf:.03} < R_{{0}} < {r0_sup:.03}$$
    ''' 

SIMULATION_CONFIG = '''
>### Configurações da  simulação (menu à esquerda)
>
>### Seleção de Unidade
É possível selecionar o tipo de unidade (Estado ou Município).
>#### Seleção de UF/Município
>Baseado na seleção anterior, é possível selecionar uma unidade da federação ou município para utilizar seus parâmetros nas condições inicias de *População total* (N), *Indivíduos infecciosos inicialmente* (I0), *Indivíduos removidos com imunidade inicialmente* (R0) e *Indivíduos expostos inicialmente (E0)*.
>
>#### Limites inferiores e superiores dos parâmetros
>Também podem ser ajustados limites superior e inferior dos parâmetros *Período infeccioso*, *Tempo de incubação* e *Número básico de reprodução*. Estes limites definem um intervalo de confiança de 95% de uma distribuição log-normal para cada parâmetro.\n\n\n
'''


HOSPITAL_QUEUE_SIMULATION= '''
###SIMULAÇÃO FILA HOSPITALAR

'''

DATA_SOURCES = '''
### Fontes dos dados

* Casos confirmados: [Número de casos confirmados de COVID-19 no Brasil](https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv) (de https://github.com/wcota/covid19br)
* População: Estimativa IBGE de 01/07/2019 (disponível em: [IBGE - Estimativas da população](https://www.ibge.gov.br/estatisticas/sociais/populacao/9103-estimativas-de-populacao.html))
'''
