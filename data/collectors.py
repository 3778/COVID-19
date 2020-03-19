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
        .dt.strftime(date_format='%d_%m_%Y')
        .unique()
    )
    for day in days:
        (
            df
            [df['date'].dt.strftime(date_format='%d_%m_%Y') == day]
            .sort_values(by='UF')
            .drop(['date'], axis=1)
            .to_csv(
                f'data/csv/by_day/{day}.csv',
                index=False
            )
        )


if __name__ == '__main__':

    df = load_ms_data()
    dump_by_uf(df)
    dump_by_day(df)
