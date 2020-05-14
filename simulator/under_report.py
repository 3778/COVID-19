import math
import pandas as pd
import numpy as np
import streamlit as st

from covid19 import data
from covid19.regressions import spline_poisson

MIN_DEATH_SUBN = 5
MIN_DATA_BRAZIL = '2020-03-26'
FATAL_RATE_BASELINE = 0.00657 

def uf(sd,mean):
    u = math.log((math.pow(mean,2))/(math.sqrt(math.pow(sd,2)+math.pow(mean,2))))
    return u

def sf(sd,mean):
    s = math.sqrt(math.log(1+(math.pow(sd/mean,2))))
    return s

def death_distrib_integral(x,u,s):
    fx = (-1/2)*math.erf((u-math.log(x))/(s*(math.sqrt(2))))
    return fx

def day_death_prob(day,mean,sd):
    init = day
    final = day+1
    if init == 0:
        init = 0.001
    u = uf(sd, mean)
    s = sf(sd,mean)
    prob = death_distrib_integral(final,u,s)-death_distrib_integral(init,u,s)
    return prob

def calculatecCFR(cases,period=False):

    mean = 13  # Linton NM, Kobayashi T, Yang Y et al. Incubation period and other
    sd = 12.7  # epidemiological characteristics of 2019 novel coronavirus infections
    # with right truncation: A statistical analysis of publicly available case data.
    # Journal of Clinical Medicine 2020;9:538.

    estim_deaths = 0
    cases_confirm = 0
    deaths_confirm = 0
    cases['cCFR'] = 0

    for i in range(cases.shape[0]):
        cases_confirm += cases['newCases'].iloc[i]
        deaths_confirm += cases['newDeaths'].iloc[i]
        for j in range(cases.shape[0] - i + 1):
            estim_deaths += cases['newCases'].iloc[i + j - 1] * day_death_prob(j, mean, sd)
        cases['cCFR'].iloc[cases.shape[0] - i - 1] = deaths_confirm / estim_deaths

    cCFR = deaths_confirm / estim_deaths

    if not period:
        return cCFR

    else:
        return cases

def subnotification(cases,period=False):
    
    if not period:
        cCFR_place = calculatecCFR(cases,period)
        subnotification_rate = FATAL_RATE_BASELINE/cCFR_place
        return subnotification_rate, cCFR_place
    else:
        cases = calculatecCFR(cases,period)
        befor_min_deaths = cases[cases['deaths']<MIN_DEATH_SUBN]
        befor_min_deaths = befor_min_deaths.reset_index()
        cCFR_befor = befor_min_deaths['cCFR'][0]
        if cCFR_befor == 0:
            cCFR_befor = FATAL_RATE_BASELINE
        cases.loc[(cases['deaths']<MIN_DEATH_SUBN),'cCFR'] = cCFR_befor
        cases['cum_subn'] = FATAL_RATE_BASELINE/cases['cCFR']
        cases = cases.sort_index()
        cases = cases[:-1]
        #cases = spline_poisson(cases,'cum_subn')  #regressão ruim
        cases = spline_poisson(cases,'newCases')  
        cases['real_newCases'] = 0
        
        for i in range(1,cases.shape[0]):
            subn_i = cases['cum_subn'].iloc[i]
            subn_i_less_1 = cases['cum_subn'].iloc[i-1]
            incid_i = cases['newCases_regression'].iloc[i]
            incid_til_i_less_1 = cases['newCases_regression'].cumsum().iloc[i-1]
            cases['real_newCases'].iloc[i] = np.random.poisson((incid_i+incid_til_i_less_1*(1-subn_i/subn_i_less_1))/subn_i)
        
        return cases


def estimate_subnotification(place, date,w_granularity,period=False):

    if w_granularity == 'city':
        city_deaths = data.get_city_deaths(place,date)
        state = place.split('/')[1]
        if city_deaths < MIN_DEATH_SUBN:
            place = state
            w_granularity = 'state'

    if w_granularity == 'state':
        state_deaths = data.get_state_deaths(place,date)
        if state_deaths < MIN_DEATH_SUBN:
            place = 'Brasil'
            w_granularity = 'brazil'

    if w_granularity == 'city':

        previous_days = data.get_city_previous_days(place,date)
        previous_days = previous_days.sort_index(ascending=False)

        if not period:
            return subnotification(previous_days)
        else:
            return subnotification(previous_days,period=True),place

    if w_granularity == 'state':

        previous_days = data.get_state_previous_days(place,date)
        previous_days = previous_days.sort_index(ascending=False)

        if not period:
            return subnotification(previous_days)
        else:
            return subnotification(previous_days,period=True),place

    if w_granularity == 'brazil':

        previous_days = data.get_brazil_previous_days(date)
        previous_days = previous_days.sort_index(ascending=False)

        if not period:
            return subnotification(previous_days)
        else:
            return subnotification(previous_days,period=True),place


