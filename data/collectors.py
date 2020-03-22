import json
import pandas as pd
import requests


def load_dump_covid_19_data():

    COVID_19_BY_CITY_URL='https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv'

    by_city=(pd.read_csv(COVID_19_BY_CITY_URL)
               .query('country == "Brazil"')
               .drop(columns=['country'])
               .pipe(lambda df: df[df.state!='TOTAL'])
               .assign(city=lambda df: df.city.apply(lambda x: x.split('/')[0]))
               .rename(columns={'totalCases': 'cases',
                                'newCases': 'new_cases',
                                'state': 'uf'})
               .sort_values(by=['city', 'date'])
            )

    by_uf = (by_city
             .groupby(['date', 'uf'])
             ['new_cases', 'cases']
             .sum()
             .reset_index())
    
    dfs = [by_uf, by_city]
    filenames = ['by_uf', 'by_city']
    for df, filename in zip(dfs, filenames):
        output_path = f'data/csv/covid_19/{filename}/{filename}.csv'
        df.to_csv(output_path, index=False)
        print(f'{filename} data exported to {output_path}')


def load_dump_uf_pop():
    IBGE_POPULATION_EXCEL_URL = 'ftp://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2019/estimativa_dou_2019.xls'
    
    def _load_uf_codes():
        print('Scraping UF codes')
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
        print('Scraping UF capital names')
        return (
            pd.read_html(
                'https://www.estadosecapitaisdobrasil.com/'
            )
            [0]
            .rename(columns={'Sigla': 'uf', 'Capital': 'city'})
            [['uf', 'city']]
        )

    # TODO: download excel file only once
    def _download_ibge_excel_file(url):
        pass
    
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
        output_path = f'data/csv/population/{filename}/{filename}.csv'
        df.to_csv(output_path, index=False)
        print(f'{filename} data exported to {output_path}')

def load_jh_df(csv):
    '''
    Loads a CSV file from JH repository and make some transforms
    '''
    jh_data_path = (
    'https://raw.githubusercontent.com/'
    'CSSEGISandData/COVID-19/master/'
    'csse_covid_19_data/csse_covid_19_time_series/'
    )

    return (
        pd.read_csv(
            jh_data_path
            + csv[1]
        )
        .drop(['Lat', 'Long'], axis=1)
        .groupby('Country/Region')
        .sum()
        .reset_index()
        .rename(
            columns={'Country/Region':'country'}
        )
        .melt(
            id_vars=['country'],
            var_name='date',
            value_name=csv[0]
        )
        .assign(
            date=lambda x: pd.to_datetime(
                x['date'],
                format='%m/%d/%y'
            )
        )
    )

def load_jh_data():
    '''
    Loads the latest COVID-19 global data from
    Johns Hopkins University repository
    '''
    cases_csv = ('cases', 'time_series_19-covid-Confirmed.csv')
    deaths_csv = ('deaths', 'time_series_19-covid-Deaths.csv')
    recovered_csv = ('recoveries', 'time_series_19-covid-Recovered.csv')

    return (
        pd.merge(
            pd.merge(
                load_jh_df(cases_csv),
                load_jh_df(deaths_csv)
            ),
             load_jh_df(recovered_csv)
        )
        .reindex(
            columns = ['date',
                       'cases',
                       'deaths',
                       'recoveries',
                       'country']
        )
    )

if __name__ == '__main__':
    # try:
    #     load_dump_covid_19_data()
    # except Exception as e:
    #     print(f'Error when collecting COVID-19 cases data: {repr(e)}')
    
    try:
        load_dump_uf_pop()
    except Exception as e:
        print(f'Error when collecting population data: {repr(e)}')
