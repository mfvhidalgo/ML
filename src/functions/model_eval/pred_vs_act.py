import matplotlib.pyplot as plt
from typing import List, Union
import pandas as pd

from ..math_utils.round_sig import round_sig

def plot_pred_vs_act(predicted_vals: Union[List, pd.Series],
                     actual_vals: Union[List, pd.Series],
                     title: str = '',
                     predicted_vals_test: Union[List, pd.Series] = [],
                     actual_vals_test: Union[List, pd.Series] = [],
                     ) -> List:
    """
    Wrapper function to quickly make a predicted vs actual plots

    Args:
        predicted_vals (Union[List, pd.Series]): predicted values
        actual_vals (Union[List, pd.Series]): actual values
        title (str, optional): title of the plot, usually response or the model used. Defaults to ''.
        predicted_vals_test (Union[List, pd.Series], optional): predicted values from the validation set. Defaults to [].
        actual_vals_test (Union[List, pd.Series], optional): actual values from the validation set. Defaults to [].

    Returns:
        List: [fig, ax] objects
    """
    
    fig = plt.figure(figsize=(7.5,5))
    ax = fig.add_subplot(111)
    ax.scatter(actual_vals,predicted_vals,c='lightgrey')
    try:
        ax.scatter(actual_vals_test,predicted_vals_test,c='r')
    except:
        pass
    all_vals = list(predicted_vals)+list(predicted_vals_test)+list(actual_vals)+list(actual_vals_test)
    min_val = min(all_vals)
    max_val = max(all_vals)
    mid_vals = (min_val+max_val)/2
    ax.plot([min_val,max_val],[min_val,max_val],ls='-',c='k')
    ax.set_title(title,fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel('Actual',fontsize=25)
    ax.set_ylabel('Predicted',fontsize=25)
    ax.set_xticks([round_sig(min_val),round_sig(mid_vals),round_sig(max_val)])
    ax.set_yticks([round_sig(min_val),round_sig(mid_vals),round_sig(max_val)])

    return [fig,ax]
