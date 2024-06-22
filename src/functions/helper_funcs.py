# functions that run very specific, sometimes one-time functions like loading data in a specific way

from typing import Dict
import pandas as pd

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
    data = pd.read_excel(data_xlsx_file_loc, sheet_name = 'Data')
    design_parameters = pd.read_excel('Data.xlsx', sheet_name = 'Design Parameters').set_index('Code')
    response_parameters = pd.read_excel('Data.xlsx', sheet_name = 'Responses').set_index('Response')

    # remove spaces and parentheses from the feature and response names
    replacements = {' ':'','(':'',')':'','-':'','+':'','*':'','/':''}
    data.columns = [col.translate(str.maketrans(replacements)) for col in data.columns]
    data.columns = [f'_{col}' if col[0].isdigit() else col for col in data.columns]
    design_parameters['Features'] = [row.translate(str.maketrans(replacements)) for row in design_parameters['Features']]
    response_parameters.index = [row.translate(str.maketrans(replacements)) for row in response_parameters.index]

    # prepare dicts from Data.xlsx
    features = design_parameters['Features'].to_dict()
    levels = {'min': design_parameters['Min Level'].to_dict(),
            'max': design_parameters['Max Level'].to_dict()
            }

    term_types = design_parameters['Term type'].to_dict()
    model_orders = response_parameters['Starting model type'].to_dict()

    responses = response_parameters.index
    lambdas = response_parameters['Lambda'].apply(get_lambdas)

    # encode features
    rescalers = {}
    for feature_coded,feature in zip(features.keys(),features.values()):
        rescalers[feature_coded] = rescale(levels['min'][feature_coded],
                                            levels['max'][feature_coded],
                                            -1,1)
        data[feature_coded] = rescalers[feature_coded].transform(data[feature])

    return {'data':data,
            'design_parameters':design_parameters,
            'response_parameters':response_parameters,
            'features':features,
            'levels':levels,
            'term_types':term_types,
            'model_orders':model_orders,
            'responses':responses,
            'lambdas':lambdas,
            'rescalers':rescalers
    }
