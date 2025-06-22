import numpy as np
from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
from sklearn.base import clone

def suggest_next_experiment(X_actual: np.array,
                            y_actual: np.array,
                            X_candidates: np.array,
                            model_actual,
                            goal: str = 'max',
                            acq_func_name: str = 'EI',
                            n_suggestions: int = 1,
                            kriging_believer: str = 'prediction',
                            xi: float = 0.01
                            ) -> list:
    """
    Wrapper to run the acquisition function and suggest the next best experiment.
    Uses skopt.acquisition functions so assumes that the goal is to MINIMIZE the model.

    Args:
        X_actual (np.array): array with shape of n_experiments, n_features representing the actual data.
        y_actual (np.array): y-values of the model at points x.
        X_candidates (np.array): array with shape of n_experiments, n_features representing the candidates for next experiments.
        model_actual: sklearn estimator that implements predict with return_std.
        goal (str, optional): whether the goal is to find the minimum or maximum of the system. Options are ['min', 'max']. Defaults to 'max'.
        acq_func_name (str, optional): acquisition function. Options are ['EI', 'PI', or 'LCB']. Defaults to 'EI'.
        n_suggestions (int, optional): number of experiments to give. Defaults to 1.
        kriging_believer (str, optional): type of believer to use. Options are ['prediction', 'mean', 'min', 'max']. Defaults to 'prediction'.
        xi (float, optional): minimum increase in y to be considered an improvement. Only applicable if acq_func_name is 'EI'. Defaults to 0.01.

    Returns:
        list:   acqs (list): list of the y-values of the acquisition function over the entire design space
                indexes (list): list of indices of the best experiments based on the acq func
                xs (np.array): array of recommended experiments
                ys (np.array): array of recommended experiments
    """

    acqs, indexes, xs = [], [], []
    if goal == 'min':
        X, y = X_actual.copy(), y_actual.copy()
    elif goal == 'max':
        X, y = X_actual.copy(), -y_actual.copy()
    model = clone(model_actual)

    for n_suggestion in range(n_suggestions):
        model.fit(X, y)
        y_pred = model.predict(X_candidates)

        if acq_func_name == 'EI':
            acq = gaussian_ei(X_candidates, model, np.min(y_pred), xi=xi)
            i = np.argmax(acq)
        
        elif acq_func_name == 'PI':
            acq = gaussian_pi(X_candidates, model, np.min(y_pred))
            i = np.argmax(acq)
        
        elif acq_func_name == 'LCB':
            acq = gaussian_lcb(X_candidates, model)
            i = np.argmin(acq)

        else:
            raise ValueError(f'Invalid acq_func_name of {acq_func_name}')

        X_candidate = X_candidates[i][0]
        if kriging_believer == 'prediction':
            y_candidate = y_pred[i]
        elif kriging_believer == 'mean':
            y_candidate = np.mean(y_pred)
        elif kriging_believer == 'min':
            y_candidate = np.min(y_pred)
        elif kriging_believer == 'max':
            y_candidate = np.max(y_pred)
        else:
            raise ValueError(f'Invalid value for kriging_believer of {kriging_believer}')

        if np.mean(acq) == 0:
            print(f'Note that the acquisition function is a flat 0, so be careful when looking at the values for suggested point at index {n_suggestion}')

        acqs.append(acq)
        indexes.append(i)
        xs.append(X_candidate)

        X = np.vstack([X, X_candidate])
        y = np.append(y, y_candidate)

    return acqs, indexes, xs