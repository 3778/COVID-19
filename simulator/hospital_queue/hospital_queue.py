from enum import Enum
from io import BytesIO, StringIO
from typing import Union

import altair as alt
import base64
import streamlit as st
import pandas as pd
import numpy as np
from queue_simulation import run_queue_simulation

STYLE = """
### MODELO DE FILAS HOSPITALARES
"""

FILE_TYPES = ["csv"]

DEFAULT_PARAMS = {

    'length_of_stay_covid': 10,
    'length_of_stay_covid_uti': 7,
    'icu_rate': .1,
    'icu_rate_after_bed': .115,

    'total_beds': 12222,
    'total_beds_icu': 2421,
    'occupation_rate': .8,
    'occupation_rate_icu': .8

}

class FileType(Enum):
    """Used to distinguish between file types"""

    IMAGE = "Image"
    CSV = "csv"
    PYTHON = "Python"

def make_download_df_href(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    size = (3*len(b64)/4)/(1_024**2)
    return f"""
    <a download='covid-simulator.3778.care.csv'
       href="data:file/csv;base64,{b64}">
       Clique para baixar ({size:.02} MB)
    </a>
    """

def get_file_type(file: Union[BytesIO, StringIO]) -> FileType:
    """The file uploader widget does not provide information on the type of file uploaded so we have
    to guess using rules or ML

    I've implemented rules for now :-)

    Arguments:
        file {Union[BytesIO, StringIO]} -- The file uploaded

    Returns:
        FileType -- A best guess of the file type
    """

    if isinstance(file, BytesIO):
        return FileType.IMAGE
    content = file.getvalue()
    if (
        content.startswith('"""')
        or "import" in content
        or "from " in content
        or "def " in content
        or "class " in content
        or "print(" in content
    ):
        return FileType.PYTHON

    return FileType.CSV


def main():
    """Run this function to display the Streamlit app"""
    #st.info(__doc__)
    st.markdown(STYLE)

    st.sidebar.markdown('# Parâmetros simulação')
    los_covid = st.sidebar.number_input(
            'Tempo de estadia médio no leito comum (horas)',
             step=1,
             min_value=1,
             max_value=100,
             value=DEFAULT_PARAMS['length_of_stay_covid'])

    los_covid_icu = st.sidebar.number_input(
             'Tempo de estadia médio na UTI (horas)',
             step=1,
             min_value=1,
             max_value=100,
             value=DEFAULT_PARAMS['length_of_stay_covid_uti'])

    icu_rate = st.sidebar.number_input(
             'Taxa de pacientes encaminhados para UTI diretamente',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['icu_rate'])

    icu_after_bed = st.sidebar.number_input(
             'Taxa de pacientes encaminhados para UTI a partir dos leitos',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['icu_rate_after_bed'])
    
    total_beds = st.sidebar.number_input(
             'Quantidade de leitos',
             step=1,
             min_value=0,
             max_value=int(1e7),
             value=DEFAULT_PARAMS['total_beds'])
    
    total_beds_icu = st.sidebar.number_input(
             'Quantidade de leitos de UTI',
             step=1,
             min_value=0,
             max_value=int(1e7),
             value=DEFAULT_PARAMS['total_beds_icu'])

    occupation_rate = st.sidebar.number_input(
             'Proporção de leitos disponíveis',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['occupation_rate'])

    icu_occupation_rate = st.sidebar.number_input(
             'Proporção de leitos de UTI disponíveis',
             step=.1,
             min_value=.0,
             max_value=1.,
             value=DEFAULT_PARAMS['occupation_rate_icu'])
    
    file = st.file_uploader("Upload file", type=FILE_TYPES)
    show_file = st.empty()
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
        return

    file_type = get_file_type(file)
    if file_type == FileType.IMAGE:
        show_file.image(file)
    elif file_type == FileType.PYTHON:
        st.code(file.getvalue())
    else:
        data = pd.read_csv(file)
        data = data[['day', 'Infected']]
        hospitalized = round(data['Infected']*0.14)
        data['hospitalizados'] = hospitalized
        dat = data.rename(columns={"day": "", "Infected": "infectados"})
        st.dataframe(data.head(10))

    file.close()

    if st.button("Simular modelo de fila"):

        result = run_queue_simulation(data, {"los_covid": los_covid,
                                             "los_covid_icu": los_covid_icu,
                                             "icu_rate": icu_rate,
                                             "icu_after_bed": icu_after_bed,
                                             "total_beds": total_beds,
                                             "total_beds_icu": total_beds_icu,
                                             "occupation_rate": occupation_rate,
                                             "icu_occupation_rate": icu_occupation_rate})

        result = result.loc[:,['Occupied_beds', 'Queue', 'ICU_Occupied_beds', 'ICU_Queue']]
        #st.write(result.head())
        #model = run_queue_simulation.Model()
        st.write("Done")
        st.area_chart(result)
        download_placeholder = st.empty()
        href = make_download_df_href(result)
        st.markdown(href, unsafe_allow_html=True)
        download_placeholder.empty()

main()