import pandas as pd
import streamlit as st

from collectors import load_ms_data


df = load_ms_data()

top_ufs = (
    df
    .groupby('UF')
    ['cases']
    .max()
    .sort_values(ascending=False)
    .index
    .tolist()
)

st.title('Parte 1 - Explorar as variáveis de uma UF')

uf = (
    st
    .selectbox(
        'UF',
        top_ufs
    )
)

fields = (
    st
    .multiselect(
        'Fields to plot',
        ['suspects', 'refuses', 'cases', 'deaths'],
        default=['cases']
    )
)

st.line_chart(
    df
    [df['UF'] == uf]
    .sort_values(by='date', ascending=False)
    .drop(['UF'], axis=1)
    .reset_index(drop=True)
    .set_index('date')
    [fields]
)

st.title('Parte 2 - Explorar dias em particular para todas as UFs')

date = (
    st
    .date_input('date')
    .strftime(format='%d_%m_%Y')
)

order_by = (
    st
    .selectbox(
        'Order',
        ['suspects', 'refuses', 'cases', 'deaths']
    )
)

st.table(
    df
    [df['date'].dt.strftime(date_format='%d_%m_%Y') == date]
    .sort_values(by=order_by, ascending=False)
    .drop(['date'], axis=1)
    .reset_index(drop=True)
    .set_index('UF')
)

st.title('Parte 3 - Comparar casos entre UFs')

ufs = (
    st
    .multiselect(
        'UFs to plot',
        top_ufs,
        default=top_ufs[:2]
    )
)

pvt = (
    pd.pivot_table(
        df,
        values='cases',
        index='date',
        columns='UF',
        fill_value=0
    )
)
pvt.columns = [pair for pair in pvt.columns]
st.line_chart(
    pvt[ufs]
)
