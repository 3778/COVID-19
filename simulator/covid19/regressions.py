import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.gam.tests.test_penalized import df_autos
import pandas as pd
import streamlit as st
import numpy as np

def spline_poisson(data,column_name,integer=False):
    
    if data.shape[0] < 1:
        column_regression = column_name + '_regression'
        data[column_regression] = data[column_name]
        return data
    else:
        index_origin = data.index.name
        data.index.name = 'init_index'
        data = data.reset_index()
        data['index'] = data.index
        x_spline = data[['index']]
        bs = BSplines(x_spline, df=[4], degree=[3])
        gam_bs = GLMGam.from_formula(f'{column_name} ~ index', data=data[['index', column_name]],
                                    smoother=bs, family=sm.families.Poisson())
        res_bs = gam_bs.fit()
        column_regression = column_name + '_regression'
        if integer:
            data[column_regression] = np.random.poisson(res_bs.predict())
        else:
            data[column_regression] = res_bs.predict()
        data = data.drop(columns='index')
        data = data.set_index('init_index')
        data.index.name = index_origin
        return data