import numpy as np
import pandas as pd
import simpy
import unicodedata


def get_capacity():
    
    def get_map():
        df = (
            pd
            .read_csv('data/dict.tsv', delimiter='\t', header=None)
            .drop([1], axis=1)
        )
        df.columns = ['nome', 'tipo']
        df = df.drop_duplicates()
        return df
    
    df = (
        pd
        .read_csv(
            'data/basecnes.csv',
            delimiter=';'
        )
        [['QT_EXIST', 'QT_SUS', 'UF municipio (traducao)',
          'Codigo municipio (traducao)', 'Tipo de leito (traducao)']]
        .rename(
            columns={
                'Codigo municipio (traducao)': 'Municipio',
                'Tipo de leito (traducao)': 'Tipo leito'
            }
        )
        .drop(['UF municipio (traducao)'], axis=1)
        .groupby(['Municipio', 'Tipo leito'])['QT_EXIST', 'QT_SUS'].sum().reset_index()
    )
    df = (
        pd
        .merge(
            df,
            get_map(),
            how='left',
            left_on='Tipo leito',
            right_on='nome',
        )
        .drop(['Tipo leito', 'nome'], axis=1)
        .rename(columns={'tipo': 'Tipo leito'})
        # .drop(['Tipo leito'], axis=1)
        # .rename(columns={'nome': 'Tipo leito'})
        .groupby(['Municipio', 'Tipo leito'])['QT_EXIST', 'QT_SUS'].sum().reset_index()
    )
    return df


def gen_time_between_arrival(p):
    return 1/p


def gen_days_in_ward(pdiw):
    return np.random.poisson(pdiw)


def gen_days_in_icu(pdii):
    return np.random.poisson(pdii)


def gen_go_to_icu(gti):
    return np.random.random() < gti


def gen_recovery(fty, gti):
    return np.random.random() > (fty / gti)


def request_icu(env, icu, logger, pdii):
    logger['requested_icu'].append(env.now)
    with icu.request() as icu_request:
        time_of_arrival = env.now
        yield icu_request
        logger['time_waited_icu'].append(env.now - time_of_arrival)
        yield env.timeout(gen_days_in_ward(pdii))


def request_ward(env, ward, logger, pdiw):
    logger['requested_ward'].append(env.now)
    with ward.request() as ward_request:
        time_of_arrival = env.now
        yield ward_request
        logger['time_waited_ward'].append(env.now - time_of_arrival)
        yield env.timeout(gen_days_in_icu(pdiw))


def patient(env, ward, icu, logger, pdiw, pdii, gti, fty):
    if gen_go_to_icu(gti):
        yield env.process(request_icu(env, icu, logger, pdii))
        if not gen_recovery(fty, gti):
            logger['deaths'].append(env.now)
        else:
            yield env.process(request_ward(env, ward, logger, pdiw))
    else:
        yield env.process(request_ward(env, ward, logger, pdiw))
        if not gen_go_to_icu(gti):
            logger['recovered_from_ward'].append(env.now)
        else:
            yield env.process(request_icu(env, icu, logger, pdii))
            if not gen_recovery(fty, gti):
                logger['deaths'].append(env.now)
            else:
                yield env.process(request_ward(env, ward, logger, pdiw))


def generate_patients(env, ward, icu, new_cases, logger, pdiw, pdii, gti, fty):
    while True:
        p = new_cases[int(env.now)]
        env.process(patient(env, ward, icu, logger, pdiw, pdii, gti, fty))
        yield env.timeout(gen_time_between_arrival(p))


def observer(env, ward, icu, logger, my_bar, nsim):
    while True:
        my_bar.progress(int(env.now) / nsim)
        logger['queue_ward'].append(len(ward.queue))
        logger['queue_icu'].append(len(icu.queue))
        logger['count_ward'].append(ward.count)
        logger['count_icu'].append(icu.count)
        yield env.timeout(1)


def run_de_simulation(nsim, new_cases, logger, my_bar, ward_capacity, icu_capacity, pdiw, pdii, gti, fty):

    env = simpy.Environment()
    ward = simpy.Resource(env, ward_capacity)
    icu = simpy.Resource(env, icu_capacity)
    env.process(generate_patients(env, ward, icu, new_cases, logger, pdiw, pdii, gti, fty))
    env.process(observer(env, ward, icu, logger, my_bar, nsim))
    env.run(until=nsim)

    return logger

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def load_age_data():
    df = pd.read_excel('data/Tabela 5918.xlsx')
    df.columns = df.iloc[3].fillna('municipio')
    df = df.iloc[4:-1]
    df['municipio'] = df['municipio'].apply(lambda x: strip_accents(x.split('(')[0].strip()).upper())
    df['60 anos ou mais'] = df['60 anos ou mais'] / df['Total']
    df = df[['municipio', '60 anos ou mais']]
    return df
