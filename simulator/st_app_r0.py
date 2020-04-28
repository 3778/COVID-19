import streamlit as st
import numpy as np

from st_utils import texts
from st_utils.viz import plot_r0
from models.reproduction_number import ReproductionNumber
from models.reproduction_number import RNDefaults as rnd

SAMPLE_SIZE = 500

def prepare_for_r0_estimation(df):
    return (
            df
            ['newCases']
            .asfreq('D')
            .fillna(0)
            .rename('incidence')
            .reset_index()
            .rename(columns={'date': 'dates'})
            .set_index('dates')
    )

def load_incidence_by_location(cases_df, date, location, min_casest_th):

    return (
        cases_df
        [location]
        .query("totalCases > @min_casest_th")
        .pipe(prepare_for_r0_estimation)
        [:date]
    )

def load_incidence(cases_df):
    return (cases_df
            .stack(level=1)
            .sum(axis=1)
            .unstack(level=1))

@st.cache(show_spinner=False)
def estimate_r0(w_date,
                w_location,
                cases_df):
    
    incidence = load_incidence_by_location(cases_df,
                                           w_date,
                                           w_location,
                                           rnd.MIN_CASES_TH)

    if len(incidence) < rnd.MIN_DAYS_r0_ESTIMATE:
        used_brazil = True
        incidence = (
            load_incidence(cases_df)
            .pipe(prepare_for_r0_estimation)
            [:w_date]
        )
    else:
        used_brazil = False

    Rt = ReproductionNumber(incidence=incidence,
                            prior_shape=rnd.PRIOR_SHAPE,
                            prior_scale=rnd.PRIOR_SCALE,
                            si_pars={'mean': rnd.SI_PARS_MEAN, 
                                     'sd': rnd.SI_PARS_SD},
                            window_width=rnd.MIN_DAYS_r0_ESTIMATE - 2)

    Rt.compute_posterior_parameters()
    samples = Rt.sample_from_posterior(sample_size=rnd.SAMPLE_SIZE)

    return samples, used_brazil

def build_r0(w_date,
             w_location,
             cases_df):
    
    st.markdown("# Número de reprodução básico")
    r0_samples, used_brazil = estimate_r0(w_date,
                                          w_location,
                                          cases_df)

    if used_brazil:
        st.write(texts.r0_NOT_ENOUGH_DATA(w_location, w_date))
        location = 'Brasil'
    else:
        location =  w_location

    st.markdown(texts.r0_WARNING)
    st.altair_chart(plot_r0(r0_samples,
                            w_date, 
                            location,
                            rnd.MIN_DAYS_r0_ESTIMATE))

    st.markdown(texts.r0_ESTIMATION(location, w_date))
    r0_dist = r0_samples[:, -1]
    st.markdown(f'**O $R_{{0}}$ estimado está entre '
                f'${np.quantile(r0_dist, 0.01):.03}$ e ${np.quantile(r0_dist, 0.99):.03}$**')
    st.markdown(texts.r0_CITATION)
    st.markdown("---")

    return r0_samples, used_brazil