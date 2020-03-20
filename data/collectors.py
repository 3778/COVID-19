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
    def _load_uf_codes():
        return (
            pd.read_html(
                'https://www.oobj.com.br/bc/article/'
                'quais-os-c%C3%B3digos-de-cada-uf-no-brasil-465.html'
            )
            [0]
            .replace('\s\(\*\)', '', regex=True)
            [['Unidade da Federação', 'UF']]
        )

    uf_codes = _load_uf_codes()

    uf_pop = (pd.read_excel('ftp://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2019/estimativa_dou_2019.xls',
                        header=1)
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
    uf_pop.to_csv('data/csv/uf_population/uf_population.csv', index=False)


if __name__ == '__main__':

    df = load_ms_data()
    dump_by_uf(df)
    dump_by_day(df)
    load_dump_uf_pop()
