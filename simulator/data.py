import pandas as pd
from pathlib import Path
import unicodedata
import streamlit as st
import requests

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


@st.cache
def prepare_age_data(level, old_col, new_col):
    BASE_URL = 'http://api.sidra.ibge.gov.br/values/t/5918/p/201904/v/606/C58/all/f/n'
    url = f'{BASE_URL}{level}'
    r = requests.get(url)
    df = pd.read_json(r.text)
    df = (df.rename(columns=df.iloc[0])
            .drop(df.index[0])
            .drop(columns=['Nível Territorial',
                        'Trimestre',
                        'Variável',
                        'Unidade de Medida'])
            .rename(columns={'Grupo de idade': 'g_idade', old_col: new_col}))
    df = df.pivot(new_col, columns='g_idade')['Valor'].reset_index()
    df.columns.name = None
    return df




@st.cache
def load_age_data(granularity):
    assert granularity in ['state', 'city']

    if granularity == 'state':
        df = prepare_age_data('/N3/all', 'Unidade da Federação', 'UF')
    else:
        df = prepare_age_data('/N6/all', 'Município', 'municipio')
        df['municipio'] = df['municipio'].apply(lambda x: strip_accents(x.split(' -')[0].strip()).upper())

    return df


@st.cache
def translate_cnes_code(code):
    m = load_cnes_map()
    return m[code]

@st.cache
def translate_unid_code(code):
    m = load_unid_map()
    return m[code]

@st.cache
def load_cnes_options():
    m = pd.read_csv('data/dict.tsv', delimiter='\t')
    m['cod'] = m['cod'].astype(str)
    return list(m['cod'].unique())

@st.cache
def load_unid_options():
    m = pd.read_csv('data/unmap.csv', delimiter=',')
    m['TP_UNIDADE (Tipo de unidade)'] = m['TP_UNIDADE (Tipo de unidade)'].astype(str)
    return list(m['TP_UNIDADE (Tipo de unidade)'].unique())

@st.cache
def load_unid_map():
    m = pd.read_csv('data/unmap.csv', delimiter=',')
    m['TP_UNIDADE (Tipo de unidade)'] = m['TP_UNIDADE (Tipo de unidade)'].astype(str)
    return (m[['TP_UNIDADE (Tipo de unidade)', 'Tipo de unidade (traducao)']]
           .set_index('TP_UNIDADE (Tipo de unidade)')['Tipo de unidade (traducao)'].to_dict())

@st.cache
def load_cnes_map():
    m = pd.read_csv('data/dict.tsv', delimiter='\t')
    m['cod'] = m['cod'].astype(str)
    return m[['cod', 'desc']].set_index('cod')['desc'].to_dict()


@st.cache
def load_capacity(granularity, ward_codes, icu_codes, unid_codes):
    assert granularity in ['state', 'city']
    to_agg = 'Codigo municipio (traducao)' if granularity == 'city' else 'UF municipio (traducao)'

    df = (pd.read_csv('data/basecnes.csv', delimiter=';')
        [['TP_UNIDADE (Tipo de unidade)', 'Tipo de unidade (traducao)',
          'CO_LEITO', 'Tipo de leito (traducao)', 'QT_EXIST', 'QT_SUS',
          to_agg]])

    ward_capacity = (
        df[(df['CO_LEITO'].isin(ward_codes)) & 
           (df['TP_UNIDADE (Tipo de unidade)'].isin(unid_codes))]
        .groupby(to_agg)['QT_SUS'].sum())

    uci_capacity = (
        df[(df['CO_LEITO'].isin(icu_codes)) &
           (df['TP_UNIDADE (Tipo de unidade)'].isin(unid_codes))]
        .groupby(to_agg)['QT_SUS'].sum())

    return ward_capacity, uci_capacity


@st.cache
def load_capacity_by_city(ward_codes, icu_codes, unid_codes):
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

@st.cache
def load_capacity_by_state(ward_codes, icu_codes, unid_codes):
    df = (pd.read_csv('data/basecnes.csv', delimiter=';')
        [['TP_UNIDADE (Tipo de unidade)', 'Tipo de unidade (traducao)',
          'CO_LEITO', 'Tipo de leito (traducao)', 'QT_EXIST', 'QT_SUS',
          'UF municipio (traducao)']])

    ward_capacity_by_state = (
        df[(df['CO_LEITO'].isin(ward_codes)) & 
           (df['TP_UNIDADE (Tipo de unidade)'].isin(unid_codes))]
        .groupby('UF municipio (traducao)')['QT_SUS'].sum())

    uci_capacity_by_state = (
        df[(df['CO_LEITO'].isin(icu_codes)) &
           (df['TP_UNIDADE (Tipo de unidade)'].isin(unid_codes))]
        .groupby('UF municipio (traducao)')['QT_SUS'].sum())
    return ward_capacity_by_state, uci_capacity_by_state

@st.cache
def fix_city(w_place):
    return strip_accents(w_place.split('/')[0]).upper()