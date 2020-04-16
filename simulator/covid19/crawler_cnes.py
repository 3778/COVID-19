import pandas as pd
import sys
import time
import os

from data import get_ibge_code_list
from crawler_utils import get_city_beds
from crawler_utils import get_bed_codes

list_city = get_ibge_code_list()

df_beds = pd.DataFrame(columns=['codibge', 'Codigo', 'Descrição', 'Existente', 'Sus', 'Não Sus'])

i = 0
start_time = time.time()
for codibge in list_city:
    df_beds = pd.concat([df_beds, get_city_beds(str(codibge))], ignore_index=True)

    # fetching status
    i += 1
    elapsed_time = (time.time()-start_time)/60
    sys.stdout.write('\r Fetching data: %.2f%% %.2f minutes, Estimated: %.2f minutes' % (100*(i / len(list_city)),
                                                                                         elapsed_time,
                                                                                         (elapsed_time/i)*len(list_city)
                                                                                         )
                     )
    sys.stdout.flush()

    # speed down the crawler
    time.sleep(0.1)

# convert columns
df_beds['Existente'] = pd.to_numeric(df_beds['Existente'])
df_beds['Sus'] = pd.to_numeric(df_beds['Sus'])
df_beds['Não Sus'] = pd.to_numeric(df_beds['Não Sus'])

# assumptions normal beds
codes_normal_beds = get_bed_codes('normal')
codes_icu_beds = get_bed_codes('icu')
codes_icu_beds_covid = get_bed_codes('covid')

# df beds
df_normal_beds = df_beds[df_beds.Codigo.isin(codes_normal_beds)]['Existente']. \
    groupby(df_beds['codibge']). \
    sum(). \
    to_frame(name='qtd_leitos'). \
    reset_index()

# df beds icu
df_icu_beds = df_beds[df_beds.Codigo.isin(codes_icu_beds)]['Existente']. \
    groupby(df_beds['codibge']). \
    sum(). \
    to_frame(name='qtd_uti'). \
    reset_index()

# df beds icu covid
df_icu_beds_covid = df_beds[df_beds.Codigo.isin(codes_icu_beds_covid)]['Existente']. \
    groupby(df_beds['codibge']). \
    sum(). \
    to_frame(name='qtd_uti_covid'). \
    reset_index()

# concat df_normal_beds, df_icu_beds
df_all = pd.merge(df_normal_beds, df_icu_beds, on='codibge', how='outer')

# concat df_all, df_icu_beds_covid
df_all = pd.merge(df_all, df_icu_beds_covid, on='codibge', how='outer')

# replace NaN
df_all.fillna(0, inplace=True)

# save csv
df_all.to_csv(os.path.join(os.getcwd(), 'simulator/data/ibgeleitos.csv'), index=False)
