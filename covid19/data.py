import pandas as pd
from pathlib import Path
import unicodedata


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
COVID_19_BY_CITY_URL=('https://raw.githubusercontent.com/wcota/covid19br/'
                      'master/cases-brazil-cities-time.csv')
IBGE_POPULATION_PATH=DATA_DIR / 'ibge_population.csv'


def load_cases(by):
    '''Load cases from wcota/covid19br

    Args:
        by (string): either 'state' or 'city'.

    Returns:
        pandas.DataFrame

    Examples:
        
        >>> cases_city = load_cases('city')
        >>> cases_city['São Paulo/SP']['newCases']['2020-03-20']
        99.0

        >>> cases_state = load_cases('state')
        >>> cases_state['SP']['newCases']['2020-03-20']
        109.0
        
    '''
    assert by in ['state', 'city']

    return (pd.read_csv(COVID_19_BY_CITY_URL, parse_dates=['date'])
              .query("state != 'TOTAL'")
              .groupby(['date', by])
              [['newCases', 'totalCases']]
              .sum()
              .unstack(by)
              .sort_index()
              .swaplevel(axis=1)
              .fillna(0))


def load_population(by):
    '''Load cases from wcota/covid19br

    Args:
        by (string): either 'state' or 'city'.

    Returns:
        pandas.DataFrame

    Examples:
        
        >>> load_population('state').head()
        state
        AC      881935
        AL     3337357
        AM     4144597
        AP      845731
        BA    14873064
        Name: estimated_population, dtype: int64

        >>> load_population('city').head()
        city
        Abadia de Goiás/GO          8773
        Abadia dos Dourados/MG      6989
        Abadiânia/GO               20042
        Abaetetuba/PA             157698
        Abaeté/MG                  23237
        Name: estimated_population, dtype: int64
        
    '''
    assert by in ['state', 'city']

    return (pd.read_csv(IBGE_POPULATION_PATH)
              .rename(columns={'uf': 'state'})
              .assign(city=lambda df: df.city + '/' + df.state)
              .groupby(by)
              ['estimated_population']
              .sum()
              .sort_index())


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def load_age_data():
    df = pd.read_excel('data/Tabela 5918.xlsx')
    df.columns = df.iloc[3].fillna('municipio')
    df = df.iloc[4:-1]
    df['municipio'] = df['municipio'].apply(lambda x: strip_accents(x.split('(')[0].strip()).upper())
    return df


def translate_cnes_code(code):
    m = load_cnes_map()
    return m[code]


def translate_unid_code(code):
    m = load_unid_map()
    return m[code]


def load_cnes_options():
    m = pd.read_csv('data/dict.tsv', delimiter='\t')
    m['cod'] = m['cod'].astype(str)
    return list(m['cod'].unique())


def load_unid_options():
    m = pd.read_csv('data/unmap.csv', delimiter=',')
    m['TP_UNIDADE (Tipo de unidade)'] = m['TP_UNIDADE (Tipo de unidade)'].astype(str)
    return list(m['TP_UNIDADE (Tipo de unidade)'].unique())


def load_unid_map():
    m = pd.read_csv('data/unmap.csv', delimiter=',')
    m['TP_UNIDADE (Tipo de unidade)'] = m['TP_UNIDADE (Tipo de unidade)'].astype(str)
    return (m[['TP_UNIDADE (Tipo de unidade)', 'Tipo de unidade (traducao)']]
           .set_index('TP_UNIDADE (Tipo de unidade)')['Tipo de unidade (traducao)'].to_dict())


def load_cnes_map():
    m = pd.read_csv('data/dict.tsv', delimiter='\t')
    m['cod'] = m['cod'].astype(str)
    return m[['cod', 'desc']].set_index('cod')['desc'].to_dict()


def load_capacity_by_city(tipos_leito_ward, tipos_leito_icu, unid_codes):
    ward_codes = tipos_leito_ward
    icu_codes = tipos_leito_icu
    df = (pd.read_csv('data/basecnes.csv', delimiter=';')
        [['TP_UNIDADE (Tipo de unidade)', 'Tipo de unidade (traducao)',
          'CO_LEITO', 'Tipo de leito (traducao)', 'QT_EXIST', 'QT_SUS',
          'Codigo municipio (traducao)']])
    ward_capacity_by_city = (
        df[(df['CO_LEITO'].isin(ward_codes)) & 
           (df['TP_UNIDADE (Tipo de unidade)'].isin(unid_codes))]
        .groupby('Codigo municipio (traducao)')['QT_SUS'].sum())
    uci_capacity_by_city = (
        df[(df['CO_LEITO'].isin(icu_codes)) &
           (df['TP_UNIDADE (Tipo de unidade)'].isin(unid_codes))]
        .groupby('Codigo municipio (traducao)')['QT_SUS'].sum())
    return ward_capacity_by_city, uci_capacity_by_city


def fix_city(w_place):
    return strip_accents(w_place.split('/')[0]).upper()
