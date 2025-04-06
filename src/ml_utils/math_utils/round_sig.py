from typing import List
from math import log10, floor

def round_sig(x: List,
              sig: int = 3) -> List:
    """
    Round the numbers folloing significant figures rules.

    Args:
        x (List): list of numbers to round
        sig (int, optional): Num of significant figures. Defaults to 3.

    Returns:
        List: x but rounded to significant figures
    """
    return round(x, sig-int(floor(log10(abs(x))))-1)