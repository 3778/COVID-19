import json
import pandas as pd
import requests


def load_uf_codes():
    return (
        pd.read_html(
            'https://www.oobj.com.br/bc/article/'
            'quais-os-c%C3%B3digos-de-cada-uf-no-brasil-465.html'
        )
        [0]
        .set_index('Código UF')
        ['UF']
    )


def load_ms_db():
    return (
        json.loads(
            requests
            .get(
                'http://plataforma.saude.gov.br'
                '/novocoronavirus/resources/scripts/database.js'
            )
            .text
            .replace('\n', '')
            .replace('var database=', '')
        )
        ['brazil']
    )


def ms_db_to_df(db):
    rows = []
    for d in db:
        for v in d['values']:
            rows.append(
                [
                    d['date'],
                    v['uid'],
                    v.get('suspects', 0),
                    v.get('refuses', 0),
                    v.get('cases', 0),
                    v.get('deaths', 0)
                ]
            )
    df = (
        pd.DataFrame(
            rows,
            columns=[
                'date', 'uid', 'suspects',
                'refuses', 'cases', 'deaths'
            ]
        )
        .assign(
            date=lambda x: pd.to_datetime(
                x['date'],
                format='%d/%m/%Y'
            )
        )
    )
    return (
        pd.merge(
            df,
            load_uf_codes(),
            how='left',
            left_on='uid',
            right_index=True
        )
        .drop(['uid'], axis=1)
    )


def load_ms_data():
    db = load_ms_db()
    df = ms_db_to_df(db)
    return df


def dump_by_uf(df):
    ufs = (
        df['UF']
        .dropna()
        .unique()
    )

    for uf in ufs:
        print(f'Saving data for UF {uf}')
        (
            df
            [df['UF'] == uf]
            .sort_values(by='date')
            .drop(['UF'], axis=1)
            .to_csv(
                f'data/csv/by_uf/{uf}.csv',
                index=False
            )
        )


def dump_by_day(df):
    days = (
        df['date']
        .dropna()
        .dt.strftime(date_format='%d-%m-%Y')
        .unique()
    )
    for day in days:
        print(f'Saving data for day {day}')
        (
            df
            [df['date'].dt.strftime(date_format='%d-%m-%Y') == day]
            .sort_values(by='UF')
            .drop(['date'], axis=1)
            .to_csv(
                f'data/csv/by_day/{day}.csv',
                index=False
            )
        )


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
            [['Unidade da Federação', 'UF']]
        )

    def _load_uf_capitals():
        print('Scraping UF capital names')
        return (
            pd.read_html(
                'https://www.estadosecapitaisdobrasil.com/'
            )
            [0]
            .rename(columns={'Sigla': 'UF', 'Capital': 'city_name'})
            [['UF', 'city_name']]
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
                'NOME DO MUNICÍPIO': 'city_name',
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
            [['UF', 'city_name', 'estimated_population']]
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
            [['UF', 'estimated_population']]
        )
        
    uf_pop, city_pop, uf_capitals = (_load_uf_pop(),
                                     _load_city_pop(),
                                     _load_uf_capitals())

    print('Combining UF and city data')
    uf_pop = (
        uf_pop
        # Add capital city name
        .merge(
            uf_capitals, 
            how='left', 
            on='UF'
        )
        # Add capital population
        .merge(
            city_pop,
            how='left',
            on=['UF', 'city_name']
        )
        .rename(
            columns={
                'estimated_population_x': 'estimated_population', 
                'estimated_population_y': 'capital_estimated_population'
            }
        )
    )

    dfs = [uf_pop, city_pop]
    filenames = ['uf_population', 'city_population']
    for df, filename in zip(dfs, filenames):
        output_path = f'data/csv/{filename}/{filename}.csv'
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
    try:
        df = load_ms_data()
        dump_by_uf(df)
        dump_by_day(df)
    except Exception as e:
        print(f'Error when collecting COVID-19 cases data: {repr(e)}')
    
    try:
        load_dump_uf_pop()
    except Exception as e:
        print(f'Error when collecting population data: {repr(e)}')
