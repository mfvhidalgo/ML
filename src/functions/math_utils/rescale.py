from typing import List, Union
import numpy as np
import pandas as pd

class rescale:
    '''
    A class that allows the user to rescale values from one range to another.
    For example, if data is [25,50,75] and the range of the data is from 0 to 100
    and the user wishes to have it in the range of -1 to 1, this class will convert
    the values from [25,50,75] to [-0.5, 0, 0.5].

    The transformation is done by creating a linear fit in the data,
    with x as the actual values and y as the goal values.

    Attributes:
        actual_min (Union[int, float]): actual minimum value. In example above, this is 0
        actual_max (Union[int, float]): actual maximum value. In example above, this is 100
        goal_min (Union[int, float]): goal minimum value. In example above, this is -1
        goal_max (Union[int, float]): goal maximum value. In example above, this is 1
        slope (float): slope of the fit line (actual scale to goal scale)
        intercept (float): intercept of the fit line (actual scale to goal scale)
        reverse_slope (float): slope of the reverse fit line (goal scale to actual scale)
        reverse_intercept (float): intercept of the reverse fit line (goal scale to actual scale)

    Methods:
        transform(values: Union[np.ndarray,pd.Series,pd.DataFrame,List,int,float]
                 ) -> same dtype as input:
            Takes values (which should be in the actual scale) and converts them
            into the goal scale.
        
        reverse_transform(values: Union[np.ndarray,pd.Series,pd.DataFrame,List,int,float]
                         ) -> same dtype as input:
            Takes values (which should be in the goal scale) and converts them
            into the actual scale.

    '''
    def __init__(self,
                actual_min: Union[int, float],
                actual_max: Union[int, float],
                goal_min: Union[int, float],
                goal_max: Union[int, float],
                ) -> Union[np.ndarray,pd.Series,List]:
        self.actual_min = actual_min
        self.actual_max = actual_max
        self.goal_min = goal_min
        self.goal_max = goal_max

        self.slope = (goal_max - goal_min) / (actual_max - actual_min)
        self.intercept = goal_min - actual_min * self.slope

        self.reverse_slope = (actual_max - actual_min) / (goal_max - goal_min)
        self.reverse_intercept = actual_min - goal_min * self.reverse_slope
    
    def transform(self, values: Union[np.ndarray,pd.Series,pd.DataFrame,List,int,float]
                  ) -> Union[np.ndarray,pd.Series,pd.DataFrame,List,int,float]:
        if isinstance(values,list):
            return [val * self.slope + self.intercept for val in values]
        else:
            return values * self.slope + self.intercept
    
    def reverse_transform(self, values: Union[np.ndarray,pd.Series,pd.DataFrame,List,int,float]
                ) -> Union[np.ndarray,pd.Series,pd.DataFrame,List,int,float]:
        if isinstance(values,list):
            return [val * self.reverse_slope + self.reverse_intercept for val in values]
        else:
            return values * self.reverse_slope + self.reverse_intercept
        


