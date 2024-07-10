#%% import
import pandas as pd
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt
import os

import functions.mult_lin_reg_utils as mlr_utils
import functions.mult_lin_reg_utils.model_reduction as mod_red
import functions.mult_lin_reg_utils.terms as terms
import functions.math_utils as math_utils
import functions.helper_funcs as helper_funcs
from functions.helper_funcs import load_data_xlsx


#%%

try:
    src_dir = os.path.dirname(os.path.abspath(__file__))
except:
    src_dir = os.getcwd()
    
output_dir = helper_funcs.create_dir('Output',src_dir)
boxcox_dir = helper_funcs.create_dir('Box-Cox',output_dir)
    
#%% load data

data_xlsx = load_data_xlsx(os.path.join(src_dir,'Data.xlsx'))

data = data_xlsx['data']
design_parameters = data_xlsx['design_parameters']
response_parameters = data_xlsx['response_parameters']
features = data_xlsx['features']
levels = data_xlsx['levels']
term_types = data_xlsx['term_types']
responses = data_xlsx['responses']
lambdas = data_xlsx['lambdas']
rescalers = data_xlsx['rescalers']

#%%

reduced_models = {}
df_all_models,df_best_models = pd.DataFrame(),pd.DataFrame()

for response in responses:
    reduced_models[response] = mod_red.auto_model_reduction(data,
                                                            ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(C**2)'],
                                                            term_types = {'A':'Process','B':'Process','C':'Process'},
                                                            response = response,
                                                            key_stat = 'aicc_bic',
                                                            direction ='forwards_backwards',
                                                            lambdas = lambdas[response])
    
    for lmbda in reduced_models[response]['model_stats'].keys():
        df_all_models = pd.concat([df_all_models,reduced_models[response]['model_stats'][lmbda]]).reset_index(drop=True)
        df_best_models = pd.concat([df_best_models,reduced_models[response]['best_models'][lmbda]]).reset_index(drop=True)
        
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    df = reduced_models[response]['boxcox_info']
    ax.plot(df['lambdas'],df['ln resid sum squares'],c='k')
    ax.scatter(df['lambdas'],df['ln resid sum squares'],c='k')
    ax.axhline(df['confidence interval'],ls=':')
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Ln (Residual Sum of Squares)')
    ax.set_title(response)
    fig.savefig(os.path.join(boxcox_dir,f"{response}.jpg"))

df_all_models = df_all_models.dropna(axis='columns',how='all')
df_best_models = df_best_models.dropna(axis='columns',how='all')

#%% get models in real units

features_reversed = {}
for key, value in features.items():
    features_reversed[value] = key

real_data = data[list(features.values()) + list(responses)].copy()
real_data.columns = [features_reversed[val] for val in list(features.values())] + list(responses)
non_term_columns = ['response','r2adj','r2press','d_r2s','key_stat','direction','num_terms']

df_all_models_real,df_best_models_real = pd.DataFrame(),pd.DataFrame()
for response in responses:
    df = df_all_models.loc[df_all_models['response']==response]
    df_all_models_real = pd.concat([df_all_models_real,mod_red.encoded_models_to_real(df,
                                                                    term_types,
                                                                    response,real_data,
                                                                    non_term_columns,)])
    df_all_models_real = df_all_models_real.dropna(axis='columns',how='all')
    df = df_best_models.loc[df_best_models['response']==response]
    df_best_models_real = pd.concat([df_best_models_real,mod_red.encoded_models_to_real(df,
                                                                    term_types,
                                                                    response,real_data,
                                                                    non_term_columns,)])
    df_best_models_real = df_best_models_real.dropna(axis='columns',how='all')

#%% export data

with pd.ExcelWriter(os.path.join(output_dir,f"Models summary.xlsx")) as writer:
    df_all_models.to_excel(writer, sheet_name='all models')
    df_best_models.to_excel(writer, sheet_name='best models')
    df_all_models_real[df_all_models.columns].to_excel(writer, sheet_name='all models in real units')
    df_best_models_real[df_best_models.columns].to_excel(writer, sheet_name='best models in real units')
