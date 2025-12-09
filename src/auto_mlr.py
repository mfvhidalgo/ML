#%% USER DEFINED INPUTS

terms_list = ['A', 'B', 'C', 'A:B', 'A:C', 'B:C', 'I(A**2)', 'I(B**2)', 'I(C**2)', 'I(A**3)','I(A**2):I(B**2)', 'I(A**2):I(C**2)',]

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
from functions.model_eval.pred_vs_act import plot_pred_vs_act
import functions.mult_lin_reg_utils.power_transform as power_transform
from functions.helper_funcs import load_data_xlsx


#%%

try:
    src_dir = os.path.dirname(os.path.abspath(__file__))
except:
    src_dir = os.getcwd()
    
output_dir = helper_funcs.create_dir('Output',src_dir)
boxcox_dir = helper_funcs.create_dir('Box-Cox',output_dir)
pred_vs_act_dir = helper_funcs.create_dir('Pred vs Act',output_dir)
    
#%% load data

data_xlsx = load_data_xlsx(os.path.join(src_dir,'Data.xlsx'))

data = data_xlsx['data']
data_test = data_xlsx['data test']
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
    columns = list(term_types.keys())+[response]
    select_data = data[columns].dropna() # only doing .dropna() after selecting the features and responses is preferred over running .dropna() on the complete dataset because some experiments have missing values at different locations
    reduced_models[response] = mod_red.auto_model_reduction(select_data,
                                                            terms_list,
                                                            term_types = term_types,
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

#%% predicted vs actual
for _,row in df_best_models.iterrows():

    lmbda = row['lambda']
    key_stat = row['key_stat']
    direction = row['direction']
    response = row['response']

    columns = list(term_types.keys())+[response]
    select_data_test = data_test[columns].dropna()

    model = reduced_models[response]['models'][lmbda][key_stat][direction]
    pred = model.get_prediction(data).summary_frame(alpha=0.05)
    pred_test = model.get_prediction(select_data_test).summary_frame(alpha=0.05)
    pred_vals = power_transform.box_cox_transform(pred['mean'],lmbda,reverse=True)
    pred_vals_test = power_transform.box_cox_transform(pred_test['mean'],lmbda,reverse=True)
    fig,ax = plot_pred_vs_act(predicted_vals = pred_vals,
                                            actual_vals = data[response],
                                            title = response,
                                            predicted_vals_test = pred_vals_test,
                                            actual_vals_test = select_data_test[response]
                                            )
    plt.tight_layout()
    fig.savefig(os.path.join(pred_vs_act_dir,f"{lmbda}_{response}.jpg"))

#%% get models in real units

features_reversed = {}
for key, value in features.items():
    features_reversed[value] = key

real_data = data[list(features.values()) + list(responses)].copy()
real_data.columns = [features_reversed[val] for val in list(features.values())] + list(responses)
non_term_columns = ['response','lambda','r2adj','r2press','d_r2s','key_stat','direction','num_terms']

df_all_models_real,df_best_models_real = pd.DataFrame(),pd.DataFrame()
for response in responses:
    df = df_all_models.loc[df_all_models['response']==response]
    df_all_models_real = pd.concat([df_all_models_real,mod_red.encoded_models_to_real(df,
                                                                    term_types,
                                                                    response,real_data,
                                                                    non_term_columns)])
    df_all_models_real = df_all_models_real.dropna(axis='columns',how='all')
    df = df_best_models.loc[df_best_models['response']==response]
    df_best_models_real = pd.concat([df_best_models_real,mod_red.encoded_models_to_real(df,
                                                                    term_types,
                                                                    response,real_data,
                                                                    non_term_columns)])
    df_best_models_real = df_best_models_real.dropna(axis='columns',how='all')

#%% export data

with pd.ExcelWriter(os.path.join(output_dir,f"Models summary.xlsx")) as writer:
    df_all_models.to_excel(writer, sheet_name='all models')
    df_best_models.to_excel(writer, sheet_name='best models')
    df_all_models_real[df_all_models.columns].to_excel(writer, sheet_name='all models in real units')
    df_best_models_real[df_best_models.columns].to_excel(writer, sheet_name='best models in real units')
