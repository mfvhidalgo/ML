# mult_lin_reg

## model_reduction
* in get_better_model, added case where one model has d_r2 < 0.2 while the other has d_r2 > 0.2
* encoded_models_to_real now utilizes formulas_col instead of non_term_columns
* TODO: remove non_term_columns and clean-up fxn in encoded_models_to_real

## terms
* fixed issue with calculating the predicted value of the left out case in r2_press.
* added patsy_to_list
* added '[T.' to get_base_exponent

# known issues
* aicc does not match DX results when using categorical terms
    * since aicc matches but bic does not, likely due to issue with len(params)
    * when manually comparing, however, no int value of num_terms results in an aicc which matches DX