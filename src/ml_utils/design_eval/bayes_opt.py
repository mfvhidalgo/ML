import numpy as np
from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi

def suggest_next_experiment(x: np.array,
                            y: np.array,
                            model,
                            acq_func_name: str = 'EI'):
    """
    Wrapper to run the acquisition function and suggest the next best experiment.
    Uses skopt.acquisition functions so assumes that the goal is to MINIMIZE the model.

    Args:
        x (np.array): array with shape of n_experiments, n_features
        y (np.array): y-values of the model at points x
        model: sklearn estimator that implements predict with return_std

    Returns:
        List: list containing the acq. func. values, index of the next suggested experiment, then x and y of that suggested experiment.
    """

    if acq_func_name == 'EI':
        ei = gaussian_ei(x, model, np.min(y))
        i = np.argmax(ei)
        print(i)
        return ei, i, x[i],  y[i]
    
    if acq_func_name == 'PI':
        pi = gaussian_pi(x, model, np.min(y))
        i = np.argmax(pi)
        return pi, i, x[i], y[i]
    
    if acq_func_name == 'LCB':
        lcb = gaussian_lcb(x, model)
        i = np.argmin(lcb)
        return lcb, i, x[i], y[i]