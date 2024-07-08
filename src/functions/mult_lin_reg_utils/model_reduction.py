from typing import List, Dict
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
from itertools import combinations

from .terms import list_to_orders, list_to_formula
from .statistics import calc_bic_aicc,calc_r2_press
from .hierarchy import get_all_lower_order_terms
from .power_transform import best_boxcox_lambda

def model_reduction(data: pd.DataFrame,
                    terms_list: List,
                    term_types: Dict,
                    response: str,
                    key_stat: str = 'aicc',
                    direction: str = 'forwards') -> OLS:
    '''
    Simplifies the model via model reduction.

    Parameters:
        data (pd.DataFrame): df containing all the features, responses, and their respective values
        terms_list (List): list containing all the terms of the largest model.
        term_types (Dict): dict where the keys are the lowest-order terms and the values are either
                           'Mixture' or 'Process'
        response (str): the column name in data of the response to be modeled.
        key_stat (str): statistic used to determine if a term will be added or not. can be either
                        'aicc' for the corrected Akeike Information Criterion
                        or 'bic' for the Bayesian Information Criterion
        direction (str): direction of model reduction, either 'forwards' or 'backwards'
    
    Returns:   
        OLS: the final reduced and fit model
    '''

    if key_stat != 'aicc' and key_stat != 'bic':
        raise ValueError('key_stat must be either aicc or bic')

    if direction == 'forwards':
        return forward_model_reduction(data,terms_list,term_types,response,key_stat)
    elif direction == 'backwards':
        return backward_model_reduction(data,terms_list,term_types,response,key_stat)
    else:
        raise ValueError('direction must be either forwards or backwards')
    
def forward_model_reduction(data: pd.DataFrame,
                            terms_list: List,
                            term_types: Dict,
                            response: str,
                            key_stat: str) -> OLS:
    '''
    Applies forward model reduction. Mainly a helper function for model_reduction.

    Parameters:
        data (pd.DataFrame): df containing all the features, responses, and their respective values
        terms_list (List): list containing all the terms of the largest model.
        term_types (Dict): dict where the keys are the lowest-order terms and the values are either
                           'Mixture' or 'Process'
        response (str): the column name in data of the response to be modeled.
        key_stat (str): statistic used to determine if a term will be added or not. can be either
                        'aicc' for the corrected Akeike Information Criterion
                        or 'bic' for the Bayesian Information Criterion
    
    Returns:   
        OLS: the final reduced and fit model
    '''
    df = data.copy()
    terms_by_order = list_to_orders(terms_list)

    cur_model_terms, remaining_terms = [], []

    for term in terms_by_order[1]:
        if term_types[term] == 'Mixture':
             cur_model_terms.append(term)

    model = ols(formula=list_to_formula(cur_model_terms,
                                        term_types = term_types,
                                        response=response),data=df).fit()
    best_key_stat = calc_bic_aicc(model,key_stat)

    for order,terms in terms_by_order.items():
        terms_to_try = list(set([term for term in terms + remaining_terms if not(term in cur_model_terms)]))

        while len(terms_to_try) > 0:
            changed_best_ter = False
            for try_term in terms_to_try:
                model = ols(formula=list_to_formula(cur_model_terms + [try_term],
                                                    term_types = term_types,
                                                    response=response),
                            data=df).fit()
                
                key_stat_value = calc_bic_aicc(model,key_stat)

                if key_stat_value < best_key_stat:
                    best_term,best_key_stat = try_term,key_stat_value
                    changed_best_ter = True
            if changed_best_ter:
                cur_model_terms.append(best_term)
                terms_to_try.remove(best_term) 
            else:
                remaining_terms = terms_to_try.copy()
                terms_to_try = []
    
    formula = get_all_lower_order_terms(list_to_formula(cur_model_terms,
                                        term_types = term_types,
                                        response=response))

    return ols(formula=formula,data=df).fit()

def backward_model_reduction(data: pd.DataFrame,
                             terms_list: List,
                             term_types: Dict,
                             response: str,
                             key_stat: str) -> OLS:
    
    '''
    Applies backwards model reduction. Mainly a helper function for model_reduction.

    Parameters:
        data (pd.DataFrame): df containing all the features, responses, and their respective values
        terms_list (List): list containing all the terms of the largest model.
        term_types (Dict): dict where the keys are the lowest-order terms and the values are either
                           'Mixture' or 'Process'
        response (str): the column name in data of the response to be modeled.
        key_stat (str): statistic used to determine if a term will be added or not. can be either
                        'aicc' for the corrected Akeike Information Criterion
                        or 'bic' for the Bayesian Information Criterion
    
    Returns:   
        OLS: the final reduced and fit model
    '''
    
    df = data.copy()
    terms_by_order = list_to_orders(terms_list)
    cur_model_terms = terms_list.copy()

    model = ols(formula=list_to_formula(cur_model_terms,
                                        term_types = term_types,
                                        response=response),data=df).fit()
    best_key_stat = calc_bic_aicc(model,key_stat)

    remove_terms = []

    orders = list(terms_by_order.keys())
    orders.sort(reverse=True)

    for order in orders:
        terms = terms_by_order[order]
        remove_terms.extend(terms)
        changed_best_ter = True
        while changed_best_ter:
            changed_best_ter = False
            for remove_term in remove_terms:
                last_model_terms = cur_model_terms.copy()
                last_model_terms.remove(remove_term)

                removed_model = ols(formula=list_to_formula(last_model_terms,
                                        term_types = term_types,
                                        response=response),
                                    data=df).fit()
                
                removed_key_stat_value = calc_bic_aicc(removed_model,key_stat)

                if removed_key_stat_value < best_key_stat:
                    best_term_remove = remove_term
                    best_key_stat = removed_key_stat_value
                    changed_best_ter = True
                
            if changed_best_ter:
                remove_terms.remove(best_term_remove)
                cur_model_terms.remove(best_term_remove)

    formula = get_all_lower_order_terms(list_to_formula(cur_model_terms,
                                        term_types = term_types,
                                        response=response))

    return ols(formula=formula,data=df).fit()

def get_model_least_terms(data: pd.DataFrame,
                            num_terms_name: str = 'num_terms') -> pd.DataFrame:
    return data.sort_values(by=[num_terms_name],ascending=True).iloc[0,:].to_frame().T

def get_highest_r2_simplest_model(data: pd.DataFrame,
                                    r2_name: str = 'r2adj') -> pd.DataFrame:
    df = data.loc[data[r2_name]==max(data[r2_name])]
    return get_model_least_terms(df)

def get_better_model(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) == 1:
        return data
    
    if len(data) != 2:
        raise ValueError('data does not have exactly 2 rows')
    
    best_df = data.loc[data['d_r2s'] <= 0.2].copy()
    if len(best_df) == 0:
        return get_highest_r2_simplest_model(data, r2_name = 'r2press')
    
    differences = abs(best_df[['r2adj','r2press','d_r2s']].diff().iloc[-1])
    if (differences['r2adj'] <= 0.05) and (differences['r2press'] <= 0.05):
        return get_model_least_terms(best_df)
    if (differences['r2adj'] <= 0.05) and (differences['r2press'] > 0.05):
        return get_highest_r2_simplest_model(best_df, r2_name = 'r2press')
    if (differences['r2adj'] > 0.05) and (differences['r2press'] <= 0.05):
        return get_highest_r2_simplest_model(best_df, r2_name = 'r2adj')
    if (differences['r2adj'] > 0.05) and (differences['r2press'] > 0.05):
        return get_highest_r2_simplest_model(best_df, r2_name = 'r2press')

def round_robin_comparison(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    win_counts = {}
    for key_stat in df['key_stat']:
        win_counts[key_stat] = 0
    
    for comb in combinations(df['key_stat'], 2):
        df_key_stat = pd.concat([df.loc[df['key_stat']==comb[0]],
                                    df.loc[df['key_stat']==comb[1]]])
        best_model = get_better_model(df_key_stat)
        if len(best_model) != 1:
            raise ValueError('More than 1 best model?')
        win_counts[best_model['key_stat'].values[0]] += 1
    
    best_key = max(win_counts, key=lambda k: win_counts[k])
    return df.loc[df['key_stat']==best_key]

def get_best_model(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    
    if len(df) == 1:
        return df
    
    while len(df) != 1:
        if len(set(df['key_stat'])) == len(df['key_stat']):
            return round_robin_comparison(df)
        
        key_stats_list = list(set(df['key_stat']))
        for key_stat in key_stats_list:
            df_key_stat = df.loc[df['key_stat']==key_stat]
            df = df.drop(df_key_stat.index)
            df = pd.concat([df,get_better_model(df_key_stat)])
    
    return df

def auto_model_reduction(data: pd.DataFrame,
                         terms_list: List,
                         term_types: Dict,
                         response: str,
                         key_stat: str = 'aicc_bic',
                         direction: str = 'forwards_backwards',
                         lambdas: List = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]) -> Dict:
    """
    Automatically runs model reduction and selects the best model when comparing different statistics and directions.
    Default is to check both aicc (forwards) vs aicc (backwards) vs bic (forwards) vs bic (backwards).
    Comparisons are made with the following logic:
    - if none have r2adj – r2press >= 0.2
        o	return model with highest r2press
        o	if there are multiple candidates, pick the one simplest model
        o	comment that none are good
    - Order by r2adj and compare top 2. d_adj is the abs difference in R2adj and d_press is the abs difference in R2press
        o if d_adj <= 0.05 and d_press <= 0.05
             pick model with less terms
        o elif d_adj <= 0.05 and d_press > 0.05
             pick model with higher r2press
        o elif d_adj > 0.05 and d_press <= 0.05
             pick model with higher r2adj
        o elif d_adj > 0.05 and d_press > 0.05
             pick model with higher r2press

    - if lambda == 1 is not in CI AND we find a better lambda
        o repeat all above but with best lambda. Include this
          (along with the best from the last analysis) to be exported

    Args:
        data (pd.DataFrame): df containing all the features, responses, and their respective values
        terms_list (List): list containing all the terms of the largest model.
        term_types (Dict): dict where the keys are the lowest-order terms and the values are either
                           'Mixture' or 'Process'
        response (str): the column name in data of the response to be modeled.
        key_stat (str): statistic used to determine if a term will be added or not. can be
                        'aicc' for the corrected Akeike Information Criterion,
                        'bic' for the Bayesian Information Criterion,
                        or 'aicc_bic' for both.
        direction (str): direction of model reduction, 'forwards', 'backwards', or 'forwards_backwards'
        lambdas (List): values of lambda to try power transformations on

    Returns:
        Dict: _description_
    """

    if key_stat == 'aicc':
        key_stats = ['aicc']
    elif key_stat == 'bic':
        key_stats = ['bic']
    elif key_stat == 'aicc_bic':
        key_stats = ['aicc','bic']
    else:
        raise ValueError('key_stat can only be aicc, bic, or aicc_bic')
    
    if direction == 'forwards':
        directions = ['forwards']
    elif direction == 'backwards':
        directions = ['backwards']
    elif direction == 'forwards_backwards':
        directions = ['forwards','backwards']
    else:
        raise ValueError('direction can only be forwards, backwards, or forwards_backwards')
    
    boxcox_info = best_boxcox_lambda(data,
                                     list_to_formula(terms_list,term_types,response),
                                     response,
                                     lambdas
                                     )

    best_lambdas = [1]
    if not(1 in boxcox_info['lambdas in conf int']):
        best_lambdas.extend(boxcox_info['lambdas in conf int'])
    
    models,model_stats,best_models = {},{},{}
    
    for lmbda in best_lambdas:
        models[lmbda] = {}
        r2adjs, r2presses, d_r2s, model_params = [],[],[], []
        key_stats_list,directions_list = [],[]
        for key_stat in key_stats:
            models[lmbda][key_stat] = {}
            for direction in directions:
                models[lmbda][key_stat][direction] = model_reduction(data,
                                                            terms_list,
                                                            term_types,
                                                            response,
                                                            key_stat,
                                                            direction
                                                            )
                r2press = calc_r2_press(models[lmbda][key_stat][direction])
                r2adjs.append(models[lmbda][key_stat][direction].rsquared_adj)
                r2presses.append(r2press)
                d_r2s.append(abs(models[lmbda][key_stat][direction].rsquared_adj - r2press))
                key_stats_list.append(key_stat)
                directions_list.append(direction)
                model_params.append(models[lmbda][key_stat][direction].params)
        
        model_stats[lmbda] = pd.DataFrame({'r2adj':r2adjs,
                                            'r2press':r2presses,
                                            'd_r2s':d_r2s,
                                            'key_stat':key_stats_list,
                                            'direction':directions_list,
                                            'num_terms': [len(param) for param in model_params]
                                            })
        
        model_terms = pd.concat(model_params,axis=1).T
        model_stats[lmbda] = pd.concat([model_stats[lmbda],model_terms],axis=1)
        best_models[lmbda] = get_best_model(model_stats[lmbda])

    return {'models':models,
            'model_stats':model_stats,
            'best_models':best_models}
