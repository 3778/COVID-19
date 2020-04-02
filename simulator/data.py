import pandas as pd
from pathlib import Path
import unicodedata
import streamlit as st

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

@st.cache
def load_age_data():
    df = pd.read_excel('data/Tabela 5918.xlsx')
    df.columns = df.iloc[3].fillna('municipio')
    df = df.iloc[4:-1]
    df['estado'] = df['municipio'].apply(lambda x: (x.split('(')[1].strip().split(')')[0]).upper())
    df['municipio'] = df['municipio'].apply(lambda x: strip_accents(x.split('(')[0].strip()).upper())
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