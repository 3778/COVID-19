import numpy as np
import pandas as pd
import simpy
import unicodedata


def load_capacity_by_city():
    m = pd.read_csv('data/dict.tsv', delimiter='\t')
    ward_codes = m[m['ward'] == 1]['desc'].values
    icu_codes = m[m['icu'] == 1]['desc'].values
    df = (
        pd
        .read_csv(
            'data/basecnes.csv',
            delimiter=';'
        )
        [['TP_UNIDADE (Tipo de unidade)', 'Tipo de unidade (traducao)',
          'CO_LEITO', 'Tipo de leito (traducao)', 'QT_EXIST', 'QT_SUS',
          'Codigo municipio (traducao)']]
    )
    ward_capacity_by_city = (
        df[df['Tipo de leito (traducao)'].isin(ward_codes)]
        .groupby('Codigo municipio (traducao)')['QT_SUS'].sum())
    uci_capacity_by_city = (
        df[df['Tipo de leito (traducao)'].isin(icu_codes)]
        .groupby('Codigo municipio (traducao)')['QT_SUS'].sum())
    return ward_capacity_by_city, uci_capacity_by_city


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
    return df
