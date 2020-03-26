import numpy as np
import pandas as pd
import simpy


def gen_time_between_arrival(p):
    return 1/p


def gen_days_in_ward():
    return np.random.randint(15, 20)


def gen_days_in_icu():
    return np.random.randint(15, 45)


def gen_go_to_icu():
    return np.random.random() < 0.10


def gen_recovery():
    return np.random.random() < 0.90


def gen_priority():
    return np.random.randint(1, 5)


def gen_icu_max_wait():
    return np.random.randint(2, 5)


def gen_ward_max_wait():
    return np.random.randint(5, 15)


def request_icu(env, icu, logger):
    logger['requested_icu'].append(env.now)
    priority = gen_priority()
    with icu.request(priority=priority) as icu_request:
        time_of_arrival = env.now
        final = yield icu_request | env.timeout(gen_icu_max_wait())
        logger['time_waited_icu'].append(env.now - time_of_arrival)
        logger['priority_icu'].append(priority)
        if icu_request in final:
            yield env.timeout(gen_days_in_icu())
        else:
            logger['lost_patients_icu'].append(env.now)


def request_ward(env, ward, logger):
    logger['requested_ward'].append(env.now)
    with ward.request() as ward_request:
        time_of_arrival = env.now
        final = yield ward_request | env.timeout(gen_ward_max_wait())
        if ward_request in final:
            logger['time_waited_ward'].append(env.now - time_of_arrival)
            yield env.timeout(gen_days_in_ward())
        else:
            logger['lost_patients_ward'].append(env.now)


def patient(env, ward, icu, logger):
    if gen_go_to_icu():
        yield env.process(request_icu(env, icu, logger))
        if not gen_recovery():
            logger['deaths'].append(env.now)
        else:
            yield env.process(request_ward(env, ward, logger))
    else:
        yield env.process(request_ward(env, ward, logger))
        if not gen_go_to_icu():
            logger['recovered_from_ward'].append(env.now)
        else:
            yield env.process(request_icu(env, icu, logger))
            if not gen_recovery():
                logger['deaths'].append(env.now)
            else:
                yield env.process(request_ward(env, ward, logger))


def generate_patients(env, ward, icu, I, mk_share, logger):
    while True:
        p = I[int(env.now)].mean() * mk_share
        print(p)
        env.process(patient(env, ward, icu, logger))
        yield env.timeout(gen_time_between_arrival(p))


def observer(env, ward, icu, logger):
    while True:
        logger['queue_ward'].append(len(ward.queue))
        logger['queue_icu'].append(len(icu.queue))
        logger['count_ward'].append(ward.count)
        logger['count_icu'].append(icu.count)
        yield env.timeout(1)


def run_de_simulation(nsim, I, mk_share, logger):

    env = simpy.Environment()
    ward = simpy.Resource(env, 3000)
    icu = simpy.PriorityResource(env, 300)
    env.process(generate_patients(env, ward, icu, I, mk_share, logger))
    env.process(observer(env, ward, icu, logger))
    env.run(until=nsim)

    return logger
