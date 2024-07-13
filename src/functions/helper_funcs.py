# functions that run very specific, sometimes one-time functions like loading data in a specific way

from typing import Dict
import pandas as pd
import os

from .mult_lin_reg_utils.preprocessing import get_lambdas
from .math_utils.rescale import rescale

def load_data_xlsx(data_xlsx_file_loc: str) -> Dict:
    '''
    Loads all the parameters from Data.xlsx.
    Provides a useful helper function for any scripts designed to import Data.xlsx.

    Parameters:
        data_xlsx_file_loc (str): directory of Data.xlsx. If Data.xlsx is in the same folder as the script,
                                  just input Data.xlsx

    Returns
        Dict: dict of each key value to be imported
    '''
    # load data from Data.xlsx
    data = pd.read_excel(data_xlsx_file_loc, sheet_name = 'Train Data')
    data_test = pd.read_excel(data_xlsx_file_loc, sheet_name = 'Test Data')
    design_parameters = pd.read_excel(data_xlsx_file_loc, sheet_name = 'Design Parameters').set_index('Code')
    response_parameters = pd.read_excel(data_xlsx_file_loc, sheet_name = 'Responses').set_index('Response')

    # remove spaces and parentheses from the feature and response names
    replacements = {' ':'','(':'',')':'','-':'','+':'','*':'','/':'','°':''}
    data.columns = [col.translate(str.maketrans(replacements)) for col in data.columns]
    data.columns = [f'_{col}' if col[0].isdigit() else col for col in data.columns]
    try:
        data_test.columns = [col.translate(str.maketrans(replacements)) for col in data_test.columns]
        data_test.columns = [f'_{col}' if col[0].isdigit() else col for col in data_test.columns]
    except:
        pass
    design_parameters['Features'] = [row.translate(str.maketrans(replacements)) for row in design_parameters['Features']]
    design_parameters['Features'] = [f'_{row}' if row[0].isdigit() else row for row in design_parameters['Features']]
    response_parameters.index = [row.translate(str.maketrans(replacements)) for row in response_parameters.index]
    response_parameters.index = [f'_{row}' if row[0].isdigit() else row for row in response_parameters.index]

    # prepare dicts from Data.xlsx
    features = design_parameters['Features'].to_dict()
    feature_types = design_parameters['Feature type'].to_dict()
    for code,feature_type in feature_types.items():
        if (feature_type != 'Numerical') and (feature_type != 'Categorical'):
            raise ValueError('Feature type must be either Numerical or Categorical')
        data[features[code]] = data[features[code]].astype(float if feature_type == 'Numerical' else str)

    numerical_features = [feat for feat in feature_types.keys() if feature_types[feat] == 'Numerical']
    categorical_features = [feat for feat in feature_types.keys() if feature_types[feat] == 'Categorical']
    design_parameters.loc[design_parameters.index.isin(categorical_features)]

    levels = {}
    for feat in numerical_features:
        levels[feat] = list(design_parameters.loc[feat,'Min Level':'Max Level'].values)
    for feat in categorical_features:
        levels[feat] = data[features[feat]].unique()

    term_types = {}   
    for feat in numerical_features:
        term_types[feat] = design_parameters.loc[feat,'Term type']
    for feat in categorical_features:
        term_types[feat] = 'Process'

    responses = response_parameters.index
    lambdas = response_parameters['Lambda'].apply(get_lambdas)

    # encode features
    rescalers = {}
    for feature_coded in numerical_features:
        feature = features[feature_coded]
        rescalers[feature_coded] = rescale(levels[feature_coded][0],
                                            levels[feature_coded][1],
                                            -1,1)
        data[feature_coded] = rescalers[feature_coded].transform(data[feature])
        if not(data_test.empty):
            data_test[feature_coded] = rescalers[feature_coded].transform(data_test[feature])

    for feature_coded in categorical_features:
        feature = features[feature_coded]
        data[feature_coded] = data[feature]
        if not(data_test.empty):
            data_test[feature_coded] = data_test[feature]

    return {'data':data,
            'data test':data_test,
            'design_parameters':design_parameters,
            'response_parameters':response_parameters,
            'features':features,
            'feature types':feature_types,
            'levels':levels,
            'term_types':term_types,
            'responses':responses,
            'lambdas':lambdas,
            'rescalers':rescalers
    }

def create_dir(new_folder_name: str,
               new_folder_parent_dir: str
               ) -> str:
    """
    Creates a dir if not yet available

    Args:
        new_folder_name (str): name of new folder to be made
        new_folder_parent_dir (str): parent dir of the folder to be made

    Returns:
        str: path of new folder
    """
    new_dir = os.path.join(new_folder_parent_dir,new_folder_name)
    if not(new_folder_name in os.listdir(new_folder_parent_dir)):
        os.makedirs(new_dir)
        
    return new_dir