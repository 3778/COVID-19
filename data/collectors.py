import json
import pandas as pd
import requests


def load_dump_uf_pop():
    IBGE_POPULATION_EXCEL_URL = 'ftp://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2019/estimativa_dou_2019.xls'
    
    def _load_uf_codes():
        return (
            pd.read_html(
                'https://www.oobj.com.br/bc/article/'
                'quais-os-c%C3%B3digos-de-cada-uf-no-brasil-465.html'
            )
            [0]
            .replace('\s\(\*\)', '', regex=True)
            .rename(columns={'UF': 'uf'})
            [['Unidade da Federação', 'uf']]
        )

    def _load_uf_capitals():
        return (
            pd.read_html(
                'https://www.estadosecapitaisdobrasil.com/'
            )
            [0]
            .rename(columns={'Sigla': 'uf', 'Capital': 'city'})
            [['uf', 'city']]
        )

    def _load_city_pop():
        print('Scraping city population')
        return (
            pd.read_excel(IBGE_POPULATION_EXCEL_URL, sheet_name='Municípios', header=1)
            .rename(columns={
                'COD. UF': 'UF_code',
                'COD. MUNIC': 'city_code',
                'NOME DO MUNICÍPIO': 'city',
                'POPULAÇÃO ESTIMADA': 'estimated_population'
            })
            .dropna(how='any')
            .assign(estimated_population=lambda df: df.estimated_population
                                                    .replace('\.', '', regex=True)
                                                    .replace('\-', ' ', regex=True)
                                                    .replace('\(\d+\)', '', regex=True)
                                                    .astype('int')
            )
            .assign(  UF_code=lambda df: df.UF_code.astype(int))
            .assign(city_code=lambda df: df.city_code.astype(int))
            .rename(columns={'UF': 'uf'})
            [['uf', 'city', 'estimated_population']]
        )
    
    def _load_uf_pop():
        print('Scraping UF population')
        uf_codes = _load_uf_codes()
        return (
            pd.read_excel(IBGE_POPULATION_EXCEL_URL, header=1)
            .drop(columns=['Unnamed: 1'])
            .rename(columns={'POPULAÇÃO ESTIMADA': 'estimated_population'})
            .dropna(how='any')
            .assign(estimated_population=lambda df: df.estimated_population
                                                    .replace('\.', '', regex=True)
                                                    .replace('\-', ' ', regex=True)
                                                    .replace('\(\d\)', '', regex=True)
                                                    .astype('int')
            )
            .pipe(lambda df: pd.merge(df,
                                    uf_codes,
                                    left_on='BRASIL E UNIDADES DA FEDERAÇÃO',
                                    right_on='Unidade da Federação',
                                    how='inner'))
            [['uf', 'estimated_population']]
        )
        
    uf_pop, city_pop, uf_capitals = (_load_uf_pop(),
                                     _load_city_pop(),
                                     _load_uf_capitals())

    print('Combining uf and city data')
    uf_pop = (
        uf_pop
        # Add capital city name
        .merge(
            uf_capitals, 
            how='left', 
            on='uf'
        )
        # Add capital population
        .merge(
            city_pop,
            how='left',
            on=['uf', 'city']
        )
        .rename(
            columns={
                'estimated_population_x': 'estimated_population', 
                'estimated_population_y': 'capital_estimated_population'
            }
        )
    )

    dfs = [uf_pop, city_pop]
    filenames = ['by_uf', 'by_city']
    for df, filename in zip(dfs, filenames):
        output_path = f'data/population_ibge_2019.csv'
        df.to_csv(output_path, index=False)
