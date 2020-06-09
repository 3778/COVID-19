import streamlit as st
import numpy as np

from st_utils import texts
from st_utils.viz import plot_r0
from models.reproduction_number import ReproductionNumber
from models.reproduction_number import RNDefaults as rnd

from covid19.regressions import spline_poisson
import under_report as ur

import altair as alt
import matplotlib.pyplot as plt

SAMPLE_SIZE = 500

def prepare_for_r0_estimation(df):


    return (
            df
            [['date','real_newCases']]
            .rename(columns={'real_newCases': 'incidence'})
            .reset_index()
            .rename(columns={'date': 'dates'})
            .set_index('dates')
    )

def load_incidence_by_location(cases_df, date, location, real_cases,min_casest_th):
    
    df,place = real_cases
    df = df.reset_index()
    
    #df.plot(kind='line',x='index',y=['newCases','newCases_regression'])
    #plt.savefig("newCases.png")

    #df.plot(kind='line',x='index',y=['newCases','newCases_regression','real_newCases'])
    #plt.savefig("real_newCases.png")

    #df.plot(kind='line',x='index',y=['cum_subn'])
    #plt.savefig("cum_subn.png")


    return df.pipe(prepare_for_r0_estimation) ,place
    

#def load_incidence(cases_df):
#    return (cases_df
#            .stack(level=1)
#            .sum(axis=1)
#            .unstack(level=1))

@st.cache(show_spinner=False)
def estimate_r0(w_date,
                w_location,
                cases_df,
                real_cases):
    
    incidence,place = load_incidence_by_location(cases_df,
                                                w_date,
                                                w_location,
                                                real_cases,
                                                rnd.MIN_CASES_TH)


    Rt = ReproductionNumber(incidence=incidence,
                            prior_shape=rnd.PRIOR_SHAPE,
                            prior_scale=rnd.PRIOR_SCALE,
                            si_pars={'mean': rnd.SI_PARS_MEAN, 
                                     'sd': rnd.SI_PARS_SD},
                            window_width=rnd.MIN_DAYS_r0_ESTIMATE - 2)

    Rt.compute_posterior_parameters()
    samples = Rt.sample_from_posterior(sample_size=rnd.SAMPLE_SIZE)

    return samples, place

def build_r0(w_date,
             w_location,
             cases_df,
             real_cases):
    
    st.markdown("# Número de reprodução básico")
    r0_samples, place = estimate_r0(w_date,
                                          w_location,
                                          cases_df,
                                          real_cases)

    if place != w_location:
        st.write(texts.r0_NOT_ENOUGH_DATA(w_location, w_date,place))
        location = place
    else:
        location = place

    #st.markdown(texts.r0_WARNING)
    st.altair_chart(plot_r0(r0_samples,
                            w_date, 
                            location,
                            rnd.MIN_DAYS_r0_ESTIMATE))

    st.markdown(texts.r0_ESTIMATION(location, w_date))
    r0_dist = r0_samples[:, -1]
    st.markdown(f'**O $R_{{t}}$ estimado está entre '
                f'${np.quantile(r0_dist, 0.025):.03}$ e ${np.quantile(r0_dist, 0.975):.03}$**')
    st.markdown(texts.r0_CITATION)
    st.markdown("---")

    #re_mean = np.mean(r0_samples,axis=0)
    #real_cases[0]['Re'] = 0

    #for i in range(np.shape(re_mean)[0]):
    #    real_cases[0]['Re'].iloc[real_cases[0].shape[0]-i-1] = re_mean[np.shape(re_mean)[0]-i-1]
    #st.write(np.mean(r0_samples,axis=0))
    #st.write(real_cases[0])

    #real_cases[0].to_csv('Re.csv')

    return r0_samples, place